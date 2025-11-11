# LLM-OCR-API
Stateless сервис для распознавания сканов/фото/документов и извлечения структурированных данных в JSON.
Пайплайн: препроцессинг → OCR/парсинг → layout (IR) → извлечение полей (якоря/регексы) → LLM-нормализация → валидация → JSON-ответ.

Сервис ничего не хранит — возвращает результат в ответе HTTP. Логи обезличены (без ПДн).

# Возможности
- Принимает PDF/JPG/PNG/TIFF/DOCX/RTF (в зависимости от подключённых провайдеров).
- Авто-байпас OCR для PDF с текст-слоем.
- Единое Layout IR (intermediate representation) с bbox, блоками и порядком чтения.
- Двухступенчатое извлечение полей: якоря/регексы → LLM-нормализация.
- Валидаторы (маски дат/номеров, обязательность полей и т. п.).
- Таймауты/лимиты на запрос.

## Локальный запуск (без Docker)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## HTTP API (Swagger)
- Запуск сервера через env (загружает .env автоматически):
```bash
# в .env: SERVE=true, DOMAIN=0.0.0.0, PORT=8080
python main.py
```
- Альтернатива через uvicorn:
```bash
uvicorn core.transport.http.server:app --host 0.0.0.0 --port 8080
```
- Swagger UI: http://localhost:8080/docs (вкл/выкл через `SWAGGER_ENABLED` в `.env`).
- Методы:
  - `POST /api/recognize` (multipart `file` или query `url`), опции: `text_first`, `max_pages`.
  - `GET /api/settings` — посмотреть активные настройки окружения.
  - `GET /health` — проверка живости.

## Как работает пайплайн
1. Вход: файл (или URL), базовые опции.
2. Препроцессинг:
- для фото/сканов — dewarp/deskew/denoise;
- для PDF — извлечение страниц/текст-слоя.
3. Определение текст-слоя:
- если в PDF есть текст — OCR пропускается;
- иначе — запускается OCR (с координатами строк, conf).
4. Layout (IR):
- построение блоков (title, org_header, meta_kv, list_block, table, signature_stamp, barcode_qr…),
- сохранение bbox, conf, reading_order.
- Пример структуры — docs/sample_layout.json.
5. Извлечение полей:
- словари якорей/синонимов + регексы + близость по bbox,
- сбор сырых значений → ParsedDataDraft.
6. LLM-нормализация (опционально):
- приведение дат/номеров/ФИО к строгим форматам согласно JSON-схеме.
7. Валидация:
- маски (ИНН/СНИЛС/ОМС), даты ≤ сегодня, обязательность полей и пр.
8. Ответ:
- meta (версии, тайминги),
- опционально layout_ir,
- result (fields{value,conf,source_block}, errors[]).
```json
{
  "meta": { "request_id":"...", "timings_ms": { /* ... */ } },
  "layout_ir": {},
  "result": {
    "doc_type": "napravlenie_med_osmotr",
    "fields": {
      "person.last_name": { "value": "Иванов", "conf": 0.94, "source_block": "b3" }
    },
    "errors": []
  }
}
```
