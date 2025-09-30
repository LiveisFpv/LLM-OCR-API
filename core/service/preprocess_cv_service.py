from __future__ import annotations

from io import BytesIO
from typing import List, Optional
import os
import cv2


import numpy as np

from core.lib.logger import get_logger
from core.domain.schemas.image_data import ImageData, ImagePage
from core.domain.schemas.input_data import InputData
from core.domain.schemas.preprocess_type import Preprocess_type
from core.domain.ports.Preprocess_provider import Preprocess_cv


class PreprocessCVService(Preprocess_cv):
    """CV-based preprocessing: PDF/images -> ImageData with basic cleaning.

    - If input is a PDF, converts pages to images via PyMuPDF (no external deps).
    - If input is an image, wraps it as a single ImagePage.
    - Applies simple OpenCV steps: grayscale, denoise, binarize, deskew.
    """

    def get_image(self, input: InputData) -> ImageData:
        logger = get_logger("preprocess.cv")
        content, filename, content_type = self._load_bytes(input)
        is_pdf = self._is_pdf(filename, content_type, content)
        logger.info("source: name=%s content_type=%s is_pdf=%s size=%d", filename, content_type, is_pdf, len(content))
        pages = self._pdf_to_images(content) if is_pdf else [content]

        max_pages = input.options.max_pages or len(pages)
        pages = pages[:max_pages]

        image_pages: List[ImagePage] = []
        os.makedirs("tmp", exist_ok=True)
        for index, raw_bytes in enumerate(pages, start=1):
            page = ImagePage(content=raw_bytes)
            mat = page.ensure_array()
            h, w = mat.shape[:2]
            logger.info("page[%d]: decoded %dx%d", index, w, h)

            # Color-friendly pipeline for detection-first OCR engines (Paddle/Rapid)
            color = self._ensure_color(mat)
            # 1) выправляем перспективу (если найдём 4 угла)
            color = self._correct_perspective_image(color)
            # 2) устраняем перекос
            rotated, angle = self._deskew_color(color)
            # 3) апскейлим до удобного размера
            scaled = self._resize_longest(rotated, target_long=2400)
            # 4) выравниваем освещение (фон) + CLAHE
            lit = self._illumination_correct(scaled)
            enhanced = self._enhance_color(lit)  # твой CLAHE-проход по Y
            # 5) лёгкое повышение резкости
            final = self._unsharp(enhanced)
            # 6) аккуратное обрезание чёрных рамок
            # final = self._autocrop_borders(final)

            logger.info("page[%d]: deskew angle=%.2f", index, angle)

            gray_dbg = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray_dbg, cv2.CV_64F).var()
            mean = float(gray_dbg.mean())
            logger.info("page[%d]: sharpness=%.1f mean=%.1f", index, sharp, mean)

            cv2.imwrite(f"tmp/preprocessed_color_{index}.png", final)
            binary_dbg = self._binarize_for_debug(gray_dbg)
            cv2.imwrite(f"tmp/preprocessed_binary_{index}.png", binary_dbg)

            page.set_array(final, encode=True, format=".png")

            image_pages.append(page)

        return ImageData(pages=image_pages, source=filename or ("url" if input.document.url else "inline"))

    # --------- helpers ---------
    def _load_bytes(self, input: InputData) -> tuple[bytes, Optional[str], Optional[str]]:
        if input.document.data is not None:
            return input.document.data, input.document.filename, input.document.content_type
        if input.document.url:
            try:
                import requests
                resp = requests.get(str(input.document.url), timeout=15)
                resp.raise_for_status()
            except Exception as e:  # pragma: no cover - network
                raise RuntimeError(f"failed to download document: {e}")
            ct = resp.headers.get("Content-Type")
            return resp.content, input.document.filename, ct
        raise ValueError("no document data provided")

    def _is_pdf(self, filename: Optional[str], content_type: Optional[str], data: bytes) -> bool:
        if content_type and "pdf" in content_type.lower():
            return True
        if filename and filename.lower().endswith(".pdf"):
            return True
        # Fallback: PDF files start with %PDF
        return data[:4] == b"%PDF"

    def _pdf_to_images(self, data: bytes, dpi: int = 300) -> List[bytes]:
        try:
            import fitz  # PyMuPDF
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pymupdf is required to rasterize pdf pages") from e

        images: List[bytes] = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                images.append(pix.tobytes("png"))
        return images

    # --------- cv steps ---------
    def _deskew_image(self, image: "np.ndarray") -> "np.ndarray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return image
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _denoise_image(self, image: "np.ndarray") -> "np.ndarray":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.medianBlur(image, 3)

    def _binarize_image(self, image: "np.ndarray") -> "np.ndarray":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def _correct_perspective_image(self, image: "np.ndarray") -> "np.ndarray":
        # Placeholder; real perspective correction would detect corners/contours.
        return image

    # --------- color-friendly helpers for Paddle/Rapid ---------
    def _ensure_color(self, image: "np.ndarray") -> "np.ndarray":
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _enhance_color(self, image: "np.ndarray") -> "np.ndarray":
        # Gentle denoise while preserving edges
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        # CLAHE on luminance
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        merged = cv2.merge((y, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def _deskew_color(self, image: "np.ndarray") -> tuple["np.ndarray", float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return image, 0.0
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    def _binarize_for_debug(self, gray: "np.ndarray") -> "np.ndarray":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def _correct_perspective_image(self, image: "np.ndarray") -> "np.ndarray":
        img = image.copy()
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Сильное сглаживание больших перепадов + Canny
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)

        # Немного расширяем границы, чтобы контур листа замкнулся
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, k, iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return image

        # самый большой контур по площади
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 0.2 * w * h:
            # слишком маленький — вероятно, не лист
            return image

        # аппроксим до 4 точек
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            return image

        pts = approx.reshape(4, 2).astype(np.float32)

        # упорядочиваем – tl, tr, br, bl
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        rect = np.array([tl, tr, br, bl], dtype=np.float32)

        # размеры целевого прямоугольника
        def dist(a, b): return np.linalg.norm(a - b)
        widthA = dist(br, bl); widthB = dist(tr, tl)
        heightA = dist(tr, br); heightB = dist(tl, bl)
        maxW = int(max(widthA, widthB))
        maxH = int(max(heightA, heightB))
        if maxW < 100 or maxH < 100:
            return image

        dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        return warped

    def _deskew_color(self, image: "np.ndarray") -> tuple["np.ndarray", float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Бинаризация и усиление горизонтальных кластеров текста
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = 255 - thr
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
        mor = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)

        coords = np.column_stack(np.where(mor > 0))
        if coords.size == 0:
            return image, 0.0

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated, float(angle)

    def _resize_longest(self, image: "np.ndarray", target_long: int = 2400) -> "np.ndarray":
        h, w = image.shape[:2]
        cur_long = max(h, w)
        if cur_long >= target_long:
            return image
        scale = target_long / float(cur_long)
        return cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    def _illumination_correct(self, image: "np.ndarray") -> "np.ndarray":
        # работаем в оттенках серого, затем возвращаемся в BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # «фон» — сильно размытый
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
        # деление на фон выравнивает освещённость
        gray32 = gray.astype(np.float32) + 1.0
        bg32 = bg.astype(np.float32) + 1.0
        norm = cv2.divide(gray32, bg32, scale=128.0)
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    def _unsharp(self, image: "np.ndarray") -> "np.ndarray":
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)
        return cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    
    def _autocrop_borders(self,img: np.ndarray, margin: int = 6) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts,_ = cv2.findContours(255-bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return img
        x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        x = max(0, x - margin); y = max(0, y - margin)
        xe = min(img.shape[1], x + w + margin); ye = min(img.shape[0], y + h + margin)
        return img[y:ye, x:xe]