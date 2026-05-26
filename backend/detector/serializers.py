from rest_framework import serializers


class PredictSingleSerializer(serializers.Serializer):
    """Request serializer cho single text prediction."""
    text = serializers.CharField(
        required=True,
        max_length=5000,
        help_text="Nội dung bình luận cần phân loại",
    )
    model = serializers.ChoiceField(
        choices=['bilstm', 'phobert'],
        default='bilstm',
        help_text="Model sử dụng: 'bilstm' hoặc 'phobert'",
    )


class PredictBatchSerializer(serializers.Serializer):
    """Request serializer cho batch prediction."""
    texts = serializers.ListField(
        child=serializers.CharField(max_length=5000),
        min_length=1,
        max_length=100,
        help_text="Danh sách bình luận cần phân loại (tối đa 100)",
    )
    model = serializers.ChoiceField(
        choices=['bilstm', 'phobert'],
        default='bilstm',
        help_text="Model sử dụng: 'bilstm' hoặc 'phobert'",
    )


class PredictionResultSerializer(serializers.Serializer):
    """Response cho một kết quả dự đoán."""
    text = serializers.CharField()
    label = serializers.CharField()
    confidence = serializers.FloatField()


class BatchResultItemSerializer(serializers.Serializer):
    """Response item trong batch prediction."""
    id = serializers.IntegerField()
    text = serializers.CharField()
    label = serializers.CharField()
    confidence = serializers.FloatField()
