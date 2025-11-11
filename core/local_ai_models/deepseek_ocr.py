from __future__ import annotations

import os
import tempfile
from typing import Optional, Any

import numpy as np
from PIL import Image

from core.domain.schemas.image_data import ImageData
from core.domain.schemas.ocr_data import OCRData, OCRLine, OCRPage
from core.domain.ports.OCR_provider import OCR_provider

from contextlib import contextmanager

@contextmanager
def _cpu_safe_infer_patch():
    """
    На время model.infer:
      • .cuda(...)        -> no-op
      • .bfloat16()       -> .float()
      • .half()           -> .float()
      • .to(dtype=bf16)   -> dtype=float32
      • .to(device=cuda)  -> device=cpu
    """
    import torch

    orig_cuda = torch.Tensor.cuda
    orig_bf16 = torch.Tensor.bfloat16
    orig_half = torch.Tensor.half
    orig_to   = torch.Tensor.to

    def _safe_to(self, *args, **kwargs):
        new_args = list(args)

        # позиционные: dtype
        if len(new_args) >= 1 and isinstance(new_args[0], torch.dtype):
            if new_args[0] == torch.bfloat16:
                new_args[0] = torch.float32

        # позиционные: device[, dtype]
        if len(new_args) >= 1 and (isinstance(new_args[0], str) or isinstance(new_args[0], torch.device)):
            dev = new_args[0]
            if (isinstance(dev, str) and "cuda" in dev) or (isinstance(dev, torch.device) and dev.type == "cuda"):
                new_args[0] = torch.device("cpu")
            if len(new_args) >= 2 and isinstance(new_args[1], torch.dtype) and new_args[1] == torch.bfloat16:
                new_args[1] = torch.float32

        # именованные
        if kwargs.get("dtype") == torch.bfloat16:
            kwargs["dtype"] = torch.float32
        dev_kw = kwargs.get("device", None)
        if (isinstance(dev_kw, str) and "cuda" in dev_kw) or (isinstance(dev_kw, torch.device) and dev_kw.type == "cuda"):
            kwargs["device"] = torch.device("cpu")

        return orig_to(self, *new_args, **kwargs)

    try:
        torch.Tensor.cuda     = lambda self, *a, **k: self
        torch.Tensor.bfloat16 = lambda self, *a, **k: self.float()
        torch.Tensor.half     = lambda self, *a, **k: self.float()
        torch.Tensor.to       = _safe_to
        yield
    finally:
        torch.Tensor.cuda     = orig_cuda
        torch.Tensor.bfloat16 = orig_bf16
        torch.Tensor.half     = orig_half
        torch.Tensor.to       = orig_to


def _image_from_ndarray(arr: "np.ndarray") -> Image.Image:
    # большинству пайплайнов нужен RGB
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.shape[2] == 3:
        return Image.fromarray(arr[..., ::-1].copy()) if _looks_like_bgr(arr) else Image.fromarray(arr)
    if arr.shape[2] == 4:
        # RGBA -> RGB
        return Image.fromarray(arr[..., :3])
    return Image.fromarray(arr)


def _looks_like_bgr(arr: "np.ndarray") -> bool:
    # эвристика: если средний 0-й канал заметно больше 2-го — вероятно BGR
    c0, c2 = float(arr[..., 0].mean()), float(arr[..., 2].mean())
    return c0 > c2 * 1.2


def _mk_lines_from_text(text: str, w: int, h: int) -> list[OCRLine]:
    lines: list[OCRLine] = []
    bbox = [0, 0, int(w), int(h)]
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        lines.append(OCRLine(text=s, bbox=bbox, conf=0.99))
    # если текст пуст — добавим одну пустую строку, чтобы страница не была без линий
    if not lines:
        lines.append(OCRLine(text="", bbox=bbox, conf=0.0))
    return lines


class DeepSeekOCRProvider(OCR_provider):
    """
    Провайдер на базе deepseek-ai/DeepSeek-OCR.
    Поддерживает два backend'а:
      - HF (transformers, model.infer) — DS_BACKEND=hf
      - vLLM — DS_BACKEND=vllm (батчинги, быстрый inference)

    Примечание по разметке:
      DeepSeek-OCR часто возвращает готовый текст/markdown без боксов.
      Мы маппим строки в OCRLine с bbox на всю страницу: совместимо с твоими Pydantic-моделями.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        prompt: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        vllm_ngram_size: int = 30,
        vllm_window_size: int = 90,
        vllm_whitelist_token_ids: Optional[set[int]] = None,
    ) -> None:
        self.model_name = model_name or os.getenv("DS_MODEL", "deepseek-ai/DeepSeek-OCR")
        self.backend = (backend or os.getenv("DS_BACKEND", "hf")).lower()
        self.prompt = prompt or os.getenv("DS_PROMPT", "<image>\nFree OCR.")
        self.base_size = int(os.getenv("DS_BASE_SIZE", str(base_size)))
        self.image_size = int(os.getenv("DS_IMAGE_SIZE", str(image_size)))
        self.crop_mode = (os.getenv("DS_CROP_MODE", "1") if os.getenv("DS_CROP_MODE") is not None else crop_mode)
        if isinstance(self.crop_mode, str):
            self.crop_mode = self.crop_mode not in ("0", "false", "False")

        self._hf_model = None
        self._hf_tokenizer = None

        self._vllm_llm = None
        self._vllm_sampling = None
        self._vllm_ngram_size = vllm_ngram_size
        self._vllm_window_size = vllm_window_size
        self._vllm_whitelist_token_ids = vllm_whitelist_token_ids or {128821, 128822}  # <td>, </td>

        if self.backend == "hf":
            self._init_hf()
        elif self.backend == "vllm":
            self._init_vllm()
        else:
            raise ValueError(f"Unknown DS_BACKEND={self.backend}. Use 'hf' or 'vllm'.")

    # ---------- HF backend ----------
    def _init_hf(self) -> None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, use_safetensors=True)

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        if getattr(model, "generation_config", None) is not None and tok.pad_token_id is not None:
            model.generation_config.pad_token_id = tok.pad_token_id

        self._hf_tokenizer = tok
        self._hf_model = model.eval().to("cpu").to(torch.float32)  # CPU + fp32


    # ---------- vLLM backend ----------
    def _init_vllm(self) -> None:
        try:
            from vllm import LLM, SamplingParams
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        except Exception as e:
            raise RuntimeError(f"vLLM unavailable: {e}")

        self._vllm_llm = LLM(
            model=self.model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
        )
        self._vllm_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=self._vllm_ngram_size,
                window_size=self._vllm_window_size,
                whitelist_token_ids=self._vllm_whitelist_token_ids,
            ),
            skip_special_tokens=False,
        )


    # ---------- OCR_provider API ----------
    def get_text(self, data: ImageData) -> OCRData:
        pages: list[OCRPage] = []
        for idx, page in enumerate(data.pages, start=1):
            mat = page.ensure_array()
            pages.append(self._extract_page(mat, idx))
        return OCRData(language=None, has_text_layer=False, pages=pages)

    def _extract_page(self, image: "np.ndarray", page_number: int) -> OCRPage:
        h, w = image.shape[:2]
        text = ""

        if self.backend == "hf":
            # Use eval-mode runner to get text back from DeepSeek-OCR
            text = self._run_hf_eval(image)
        else:
            text = self._run_vllm(image)

        lines = _mk_lines_from_text(text, w, h)
        return OCRPage(num=page_number, width=w, height=h, rotation=0, lines=lines)

    # ---------- Backend runners ----------
    def _run_hf_eval(self, image: "np.ndarray") -> str:
        import os, tempfile, shutil

        pil = _image_from_ndarray(image).convert("RGB")

        def _as_bool(v: object) -> bool:
            return str(v or "").lower() in ("1", "true", "yes")

        eval_mode = _as_bool(os.getenv("DS_EVAL_MODE", "1"))
        save_results = _as_bool(os.getenv("DS_SAVE_RESULTS", "0"))
        test_compress = _as_bool(os.getenv("DS_TEST_COMPRESS", "0"))
        out_dir_env = os.getenv("DS_OUT_DIR")

        tmp_img_path = None
        tmp_out_dir = None
        try:
            fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil.save(tmp_img_path)

            if out_dir_env:
                os.makedirs(out_dir_env, exist_ok=True)
                tmp_out_dir = out_dir_env
            else:
                tmp_out_dir = tempfile.mkdtemp(prefix="dsocr_")

            with _cpu_safe_infer_patch():
                res = self._hf_model.infer(
                    self._hf_tokenizer,
                    prompt=self.prompt,
                    image_file=tmp_img_path,
                    output_path=tmp_out_dir,
                    base_size=self.base_size,
                    image_size=self.image_size,
                    crop_mode=bool(self.crop_mode),
                    save_results=save_results,
                    test_compress=test_compress,
                    eval_mode=eval_mode,
                )

            text = ""
            if isinstance(res, str):
                text = res
            elif isinstance(res, dict):
                for k in ("text", "output", "result"):
                    v = res.get(k)
                    if isinstance(v, str):
                        text = v
                        break
            elif isinstance(res, (list, tuple)) and res and isinstance(res[0], str):
                text = "\n".join(res)
            else:
                text = str(res) if res is not None else ""

            return self._clean_generated_text(text)

        finally:
            if tmp_img_path and os.path.exists(tmp_img_path):
                try:
                    os.remove(tmp_img_path)
                except Exception:
                    pass
            keep_tmp = _as_bool(os.getenv("DS_KEEP_TMP", "0"))
            preserve = keep_tmp or bool(out_dir_env) or save_results
            if not preserve and tmp_out_dir and os.path.isdir(tmp_out_dir):
                try:
                    shutil.rmtree(tmp_out_dir, ignore_errors=True)
                except Exception:
                    pass

    def _run_hf(self, image: "np.ndarray") -> str:
        import os, tempfile, shutil

        pil = _image_from_ndarray(image).convert("RGB")

        tmp_img_path = None
        tmp_out_dir = None
        try:
            # --- временный PNG (важно закрыть дескриптор на Windows) ---
            fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil.save(tmp_img_path)

            # --- временная папка для DeepSeek-OCR (infer требует непустой path) ---
            tmp_out_dir = tempfile.mkdtemp(prefix="dsocr_")

            # === КЛЮЧЕВОЕ: отключаем .cuda() внутри их infer() на время вызова ===
            from contextlib import nullcontext
            with _cpu_safe_infer_patch():
                res = self._hf_model.infer(
                    self._hf_tokenizer,
                    prompt=self.prompt,
                    image_file=tmp_img_path,
                    output_path=tmp_out_dir,
                    base_size=self.base_size,
                    image_size=self.image_size,
                    crop_mode=bool(self.crop_mode),
                    save_results=False,
                    test_compress=True,
                )

            return self._extract_text(res)

        finally:
            if tmp_img_path and os.path.exists(tmp_img_path):
                try: os.remove(tmp_img_path)
                except Exception: pass
            keep = os.getenv("DS_KEEP_TMP", "0").lower() in ("1", "true", "yes")
            if not keep and tmp_out_dir and os.path.isdir(tmp_out_dir):
                try: shutil.rmtree(tmp_out_dir, ignore_errors=True)
                except Exception: pass



    def _run_vllm(self, image: "np.ndarray") -> str:
        pil = _image_from_ndarray(image).convert("RGB")
        model_input = [{"prompt": self.prompt, "multi_modal_data": {"image": pil}}]
        outs = self._vllm_llm.generate(model_input, self._vllm_sampling)
        if outs and outs[0].outputs:
            return outs[0].outputs[0].text or ""
        return ""

    # ---------- Helpers ----------
    @staticmethod
    def _clean_generated_text(text: str) -> str:
        if not text:
            return ""
        out: list[str] = []
        for raw in text.splitlines():
            s = (raw or "").strip()
            if not s:
                continue
            if s.startswith("<|") and "|>" in s:
                continue
            if s.startswith("[[") and s.endswith("]]"):
                continue
            low = s.lower()
            if s.startswith("====") or low.startswith("image size:") or low.startswith("valid image tokens:") or low.startswith("output texts tokens") or low.startswith("compression ratio"):
                continue
            out.append(s)
        return "\n".join(out)
    @staticmethod
    def _extract_text(res: Any) -> str:
        # DeepSeek-OCR infer обычно возвращает строку; делаем бережный парсинг
        if res is None:
            return ""
        if isinstance(res, str):
            return res
        # иногда могут прилетать словари или списки с ключом/полем text
        if isinstance(res, dict):
            for k in ("text", "output", "result"):
                if k in res and isinstance(res[k], str):
                    return res[k]
        if isinstance(res, (list, tuple)) and res and isinstance(res[0], str):
            return "\n".join(res)
        # fallback
        return str(res)
