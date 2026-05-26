import logging

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle

from detector.model_loader import model_registry
from detector.serializers import PredictBatchSerializer, PredictSingleSerializer

logger = logging.getLogger(__name__)


@api_view(['GET'])
def health_check(request):
    """
    GET /api/health/
    Kiểm tra trạng thái server và các model.
    """
    models_status = {}
    try:
        for m in model_registry.available_models():
            models_status[m['id']] = m['loaded']
    except Exception:
        pass

    return Response({
        'status': 'ok',
        'models': models_status,
    })


@api_view(['GET'])
def list_models(request):
    """
    GET /api/models/
    Trả về danh sách các model có sẵn và thông tin.
    """
    try:
        models = model_registry.available_models()
        return Response({
            'models': models,
            'threshold': settings.TOXIC_THRESHOLD,
        })
    except Exception as e:
        logger.exception("Error listing models")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(['POST'])
def predict_single(request):
    """
    POST /api/predict/
    Phân loại một bình luận.

    Request body:
        {
            "text": "nội dung bình luận",
            "model": "bilstm" | "phobert"
        }

    Response:
        {
            "text": "...",
            "label": "toxic" | "non-toxic",
            "confidence": 0.92,
            "model": "bilstm"
        }
    """
    serializer = PredictSingleSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    text = serializer.validated_data['text'].strip()
    model_name = serializer.validated_data['model']

    if not text:
        return Response(
            {'error': 'text không được để trống'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        predictor = model_registry.get_predictor(model_name)
        results = predictor.predict_batch([text], threshold=settings.TOXIC_THRESHOLD)
        result = results[0]
        return Response({
            'text': result['text'],
            'label': result['label'],
            'confidence': result['confidence'],
            'model': model_name,
        })
    except ValueError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.exception("Prediction error for model=%s", model_name)
        return Response(
            {'error': f'Lỗi dự đoán: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(['POST'])
def predict_batch(request):
    """
    POST /api/predict/batch/
    Phân loại một batch các bình luận.

    Request body:
        {
            "texts": ["text 1", "text 2", ...],
            "model": "bilstm" | "phobert"
        }

    Response:
        {
            "results": [
                {"id": 0, "text": "...", "label": "toxic", "confidence": 0.87},
                ...
            ],
            "model": "bilstm",
            "total": 3,
            "toxic_count": 1
        }
    """
    serializer = PredictBatchSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    texts = [t.strip() for t in serializer.validated_data['texts']]
    model_name = serializer.validated_data['model']

    # Filter empty texts but keep original indices
    indexed_texts = [(i, t) for i, t in enumerate(texts) if t]
    if not indexed_texts:
        return Response(
            {'error': 'Tất cả texts đều rỗng'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        predictor = model_registry.get_predictor(model_name)
        original_indices, clean_texts = zip(*indexed_texts)
        predictions = predictor.predict_batch(
            list(clean_texts),
            threshold=settings.TOXIC_THRESHOLD,
        )

        results = []
        for idx, pred in zip(original_indices, predictions):
            results.append({
                'id': idx,
                'text': pred['text'],
                'label': pred['label'],
                'confidence': pred['confidence'],
            })

        toxic_count = sum(1 for r in results if r['label'] == 'toxic')

        return Response({
            'results': results,
            'model': model_name,
            'total': len(results),
            'toxic_count': toxic_count,
        })
    except ValueError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.exception("Batch prediction error for model=%s", model_name)
        return Response(
            {'error': f'Lỗi dự đoán batch: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
