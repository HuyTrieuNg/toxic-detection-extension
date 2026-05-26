"""
Model Loader — Singleton registry để load và cache 2 model AI.

BiLSTM: Keras model + vocab.json
  → Preprocessing: lowercase → clean_text → remove_stopwords → pyvi.ViTokenizer → text_to_sequence
PhoBERT: HuggingFace RobertaForSequenceClassification + PhobertTokenizer
    → Preprocessing: lowercase → clean_text_for_transformer → MAX_LENGTH=128
"""

import json
import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM Preprocessing — khớp với notebooks/bilstm-train.ipynb
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns (từ notebook cell 3)
_URL_PATTERN     = re.compile(r'http\S+|www\S+')
_TAG_PATTERN     = re.compile(r'\[.*?\]')
_HTML_PATTERN    = re.compile(r'<.*?>')
_MENTION_PATTERN = re.compile(r'@\w+')
_HASHTAG_PATTERN = re.compile(r'#(\w+)')
_REPEAT_PATTERN  = re.compile(r'(.)\1{2,}')
_SPACE_PATTERN   = re.compile(r'\s+')
_PUNCT_PATTERN   = re.compile(r'[^\w\s]', flags=re.UNICODE)
_EMOJI_PATTERN   = re.compile(
    '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+',
    flags=re.UNICODE,
)
_FOREIGN_PATTERN = re.compile(
    '[\u0370-\u04FF\u0600-\u06FF\u0E00-\u0E7F\u1780-\u17FF'
    '\u3040-\u30FF\u3130-\u318F\u4E00-\u9FFF\uAC00-\uD7AF]+',
    flags=re.UNICODE,
)


def _clean_text(text: str) -> str:
    """Làm sạch text — giống hệt hàm clean_text() trong notebook."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFC', text)
    text = _FOREIGN_PATTERN.sub('', text)
    text = _MENTION_PATTERN.sub('', text)
    text = _HASHTAG_PATTERN.sub(r'\1', text)
    text = _URL_PATTERN.sub('', text)
    text = _TAG_PATTERN.sub('', text)
    text = _HTML_PATTERN.sub('', text)
    text = _EMOJI_PATTERN.sub('', text)
    text = _PUNCT_PATTERN.sub(' ', text)
    text = _REPEAT_PATTERN.sub(r'\1\1', text)
    text = _SPACE_PATTERN.sub(' ', text).strip()
    return text


def _clean_text_for_transformer(text: str) -> str:
    """Tiền xử lý cho transformer (PhoBERT). Không loại bỏ dấu câu."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFC', text)
    text = _FOREIGN_PATTERN.sub('', text)
    text = _MENTION_PATTERN.sub('', text)
    text = _HASHTAG_PATTERN.sub(r'\1', text)
    text = _URL_PATTERN.sub('', text)
    text = _TAG_PATTERN.sub('', text)
    text = _HTML_PATTERN.sub('', text)
    text = _EMOJI_PATTERN.sub('', text)
    text = _REPEAT_PATTERN.sub(r'\1\1', text)
    text = _SPACE_PATTERN.sub(' ', text).strip()
    return text


def _remove_stopwords(text: str, stopwords: set) -> str:
    if not isinstance(text, str):
        return ''
    return ' '.join(w for w in text.split() if w not in stopwords)


def _tokenize_vi(text: str) -> str:
    """Word-segment tiếng Việt bằng pyvi.ViTokenizer — giống notebook."""
    if not isinstance(text, str) or not text.strip():
        return ''
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except ImportError:
        logger.warning("[BiLSTM] pyvi chưa cài, bỏ qua word-segment. Kết quả có thể kém hơn.")
        return text


def bilstm_preprocess(text: str, stopwords: Optional[set] = None) -> str:
    """
    Pipeline tiền xử lý cho BiLSTM — khớp hoàn toàn với nlp_processing_pipeline() trong notebook:
      1. lowercase
      2. clean_text (xóa URL, emoji, HTML, ký tự lặp, ...)
      3. remove_stopwords (nếu có)
      4. pyvi.ViTokenizer.tokenize
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = _clean_text(text)
    if stopwords:
        text = _remove_stopwords(text, stopwords)
    if text.strip():
        text = _tokenize_vi(text)
    return text


def phobert_preprocess(text: str) -> str:
    """
    Pipeline tiền xử lý cho PhoBERT — khớp notebook:
      1. lowercase
      2. clean_text_for_transformer (giữ dấu câu)
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = _clean_text_for_transformer(text)
    return text


def _load_stopwords(stopwords_path: Path) -> set:
    if not stopwords_path.exists():
        logger.warning("[BiLSTM] Không tìm thấy stopwords tại %s", stopwords_path)
        return set()
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return {w.strip() for w in f if w.strip()}


def _text_to_sequence(text: str, vocab: dict, max_len: int) -> np.ndarray:
    """
    Chuyển text đã preprocess thành sequence int — khớp với text_to_sequence() trong notebook.
    PAD=0, UNK=1, vocab indices bắt đầu từ 2.
    """
    tokens = str(text).strip().split()
    seq = [vocab.get(t, 1) for t in tokens]   # 1 = UNK
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [0] * (max_len - len(seq))      # 0 = PAD
    return np.array(seq, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Keras Layer — SimpleAttention
# ─────────────────────────────────────────────────────────────────────────────
def _make_simple_attention_class():
    """Tạo class SimpleAttention — dùng với custom_objects khi load model."""
    import tensorflow as tf
    from keras import layers

    class SimpleAttention(layers.Layer):
        def __init__(self, **kwargs):
            super(SimpleAttention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name='attn_W',
                shape=(input_shape[-1], 1),
                initializer='glorot_uniform',
                trainable=True,
            )
            self.b = self.add_weight(
                name='attn_b',
                shape=(1,),
                initializer='zeros',
                trainable=True,
            )
            super(SimpleAttention, self).build(input_shape)

        def call(self, x, mask=None):
            e = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
            a = tf.nn.softmax(e, axis=1)
            if mask is not None:
                mask_f = tf.cast(mask, x.dtype)
                mask_f = tf.expand_dims(mask_f, axis=-1)
                a = a * mask_f
                a = a / (tf.reduce_sum(a, axis=1, keepdims=True) + 1e-7)
            return tf.reduce_sum(x * a, axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

        def get_config(self):
            return super(SimpleAttention, self).get_config()

    return SimpleAttention


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM Predictor
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMPredictor:
    """Predictor wrapper cho BiLSTM Keras model."""

    def __init__(self, model_path: Path, vocab_path: Path, max_len: int = 100):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.max_len = max_len
        self.model = None
        self.vocab: dict = {}
        self._loaded = False
        self._stopwords: Optional[set] = None

    def load(self):
        if self._loaded:
            return
        logger.info("[BiLSTM] Đang load model từ %s", self.model_path)
        import tensorflow as tf

        SimpleAttention = _make_simple_attention_class()
        self.model = tf.keras.models.load_model(
            str(self.model_path),
            custom_objects={'SimpleAttention': SimpleAttention},
        )
        logger.info("[BiLSTM] Đang load vocab từ %s", self.vocab_path)
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        stopwords_path = self.vocab_path.parent / 'stop_words.txt'
        self._stopwords = _load_stopwords(stopwords_path)
        self._loaded = True
        logger.info(
            "[BiLSTM] Load xong. Vocab size: %d, STOPWORDS: %d, MAX_LEN: %d",
            len(self.vocab),
            len(self._stopwords or []),
            self.max_len,
        )

    def predict_batch(self, texts: list[str], threshold: float = 0.5) -> list[dict]:
        if not self._loaded:
            self.load()

        logger.info("[BiLSTM] ── Nhận batch %d texts ──", len(texts))

        preprocessed = []
        for i, text in enumerate(texts):
            processed = bilstm_preprocess(text, self._stopwords)
            logger.debug("[BiLSTM] [%d] raw   : %s", i, text[:80])
            logger.debug("[BiLSTM] [%d] clean : %s", i, processed[:80])
            preprocessed.append(processed)

        # Chuyển thành sequences
        X = np.stack([
            _text_to_sequence(t, self.vocab, self.max_len)
            for t in preprocessed
        ])
        logger.info("[BiLSTM] Input shape: %s | dtype: %s", X.shape, X.dtype)
        logger.debug("[BiLSTM] Sequence mẫu [0]: %s", X[0][:20])

        preds = self.model.predict(X, verbose=0)
        logger.info("[BiLSTM] Output shape: %s", preds.shape)

        results = []
        for i, text in enumerate(texts):
            raw = preds[i]
            confidence = float(raw[0]) if raw.shape[-1] == 1 else float(raw[1])
            is_toxic = confidence >= threshold
            logger.info(
                "[BiLSTM] [%d] \"%s\" → %s (%.4f)",
                i, text[:50], 'TOXIC' if is_toxic else 'non-toxic', confidence
            )
            results.append({
                'text': text,
                'label': 'toxic' if is_toxic else 'non-toxic',
                'confidence': round(confidence, 4),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# PhoBERT Predictor
# ─────────────────────────────────────────────────────────────────────────────
class PhoBERTPredictor:
    """
    Predictor wrapper cho PhoBERT HuggingFace model.
    MAX_LENGTH=128 khớp với notebook finetune-phobert-model-v2.ipynb.
    Input: cleaned_comment_transformer — văn bản đã được làm sạch theo notebook.
    """

    def __init__(self, model_dir: Path, max_len: int = 128):
        self.model_dir = model_dir
        self.max_len = max_len       # 128 — khớp notebook (MAX_LENGTH = 128)
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.device = None

    def load(self):
        if self._loaded:
            return
        logger.info("[PhoBERT] Đang load model từ %s", self.model_dir)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            use_fast=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
        )
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self._loaded = True
        logger.info("[PhoBERT] Load xong. Device: %s | MAX_LENGTH: %d", self.device, self.max_len)

    def predict_batch(self, texts: list[str], threshold: float = 0.5) -> list[dict]:
        if not self._loaded:
            self.load()

        import torch
        import torch.nn.functional as F

        logger.info("[PhoBERT] ── Nhận batch %d texts ──", len(texts))

        processed_texts = []
        for i, text in enumerate(texts):
            processed = phobert_preprocess(text)
            logger.debug("[PhoBERT] [%d] raw      : %s", i, text[:80])
            logger.debug("[PhoBERT] [%d] cleaned  : %s", i, processed[:80])
            processed_texts.append(processed)

        # Tokenize với PhobertTokenizer, max_length=128
        encodings = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )
        logger.info(
            "[PhoBERT] input_ids shape: %s | attention_mask shape: %s",
            tuple(encodings['input_ids'].shape),
            tuple(encodings['attention_mask'].shape),
        )

        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits          # [batch, 2]
            probs = F.softmax(logits, dim=-1)  # [batch, 2]
            toxic_probs = probs[:, 1].cpu().numpy()

        logger.info("[PhoBERT] logits shape: %s", tuple(outputs.logits.shape))

        results = []
        for i, text in enumerate(texts):
            confidence = float(toxic_probs[i])
            is_toxic = confidence >= threshold
            logger.info(
                "[PhoBERT] [%d] \"%s\" → %s (%.4f)",
                i, text[:50], 'TOXIC' if is_toxic else 'non-toxic', confidence
            )
            results.append({
                'text': text,
                'label': 'toxic' if is_toxic else 'non-toxic',
                'confidence': round(confidence, 4),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry — Singleton, thread-safe
# ─────────────────────────────────────────────────────────────────────────────
class ModelRegistry:
    """Singleton registry quản lý cả 2 model. Lazy load, thread-safe."""

    _instance: Optional['ModelRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _init_predictors(self):
        if self._initialized:
            return
        from django.conf import settings
        self.bilstm = BiLSTMPredictor(
            model_path=settings.BILSTM_MODEL_PATH,
            vocab_path=settings.BILSTM_VOCAB_PATH,
            max_len=settings.BILSTM_MAX_LEN,
        )
        self.phobert = PhoBERTPredictor(
            model_dir=settings.PHOBERT_MODEL_PATH,
            max_len=settings.PHOBERT_MAX_LEN,
        )
        self.threshold = settings.TOXIC_THRESHOLD
        self._initialized = True

    def get_predictor(self, model_name: str):
        self._init_predictors()
        if model_name == 'bilstm':
            return self.bilstm
        elif model_name == 'phobert':
            return self.phobert
        else:
            raise ValueError(f"Unknown model: '{model_name}'. Chọn 'bilstm' hoặc 'phobert'.")

    def available_models(self) -> list[dict]:
        return [
            {
                'id': 'bilstm',
                'name': 'BiLSTM + Attention',
                'description': 'BiLSTM với Attention, preprocessing bằng pyvi.ViTokenizer',
                'loaded': self.bilstm._loaded if self._initialized else False,
            },
            {
                'id': 'phobert',
                'name': 'PhoBERT',
                'description': 'PhoBERT fine-tuned, preprocessing theo notebook',
                'loaded': self.phobert._loaded if self._initialized else False,
            },
        ]


# Global singleton instance
model_registry = ModelRegistry()
