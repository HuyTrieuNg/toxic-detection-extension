# VOZ Toxic Detector

Backend Django REST API phục vụ Chrome Extension phát hiện bình luận toxic trên VOZ.vn.

## Cấu trúc

```
backend/
├── ai-models/
│   ├── LearnedEmb_BiLSTM_Attention/
│   │   ├── best_model.keras       # BiLSTM Keras model
│   │   ├── vocab.json             # Vocabulary mapping
│   │   └── stop_words.txt         # Stop words list
│   └── Phobert_binary/
│       ├── model.safetensors      # PhoBERT fine-tuned weights
│       ├── config.json
│       └── tokenizer_config.json
├── config/
│   ├── settings.py
│   └── urls.py
├── detector/                      # Django app chính
│   ├── apps.py
│   ├── model_loader.py            # Singleton model registry
│   ├── serializers.py
│   ├── views.py
│   └── urls.py
└── pyproject.toml
```

## API Endpoints

| Method | URL | Mô tả |
|--------|-----|--------|
| `GET`  | `/api/health/` | Health check + model status |
| `GET`  | `/api/models/` | Danh sách models |
| `POST` | `/api/predict/` | Phân loại 1 bình luận |
| `POST` | `/api/predict/batch/` | Phân loại batch (tối đa 100) |

## Khởi chạy

```bash
# Cài dependencies (chỉ lần đầu)
uv sync

# Chạy server
uv run python manage.py migrate
uv run python manage.py runserver
```

Server chạy tại `http://localhost:8000`

## Test API

```bash
# Health check
curl http://localhost:8000/api/health/

# Predict single
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "nội dung bình luận", "model": "bilstm"}'

# Predict batch
curl -X POST http://localhost:8000/api/predict/batch/ \
  -H "Content-Type: application/json" \
  -d '{"texts": ["text 1", "text 2"], "model": "phobert"}'
```

## Lưu ý

- **Python 3.12** được yêu cầu (TensorFlow chưa hỗ trợ 3.14)
- CORS đã được cấu hình cho phép Chrome Extension gọi API
