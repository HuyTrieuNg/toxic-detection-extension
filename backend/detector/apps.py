import logging
import os
import threading

from django.apps import AppConfig
from django.conf import settings


logger = logging.getLogger(__name__)

_preload_lock = threading.Lock()
_preload_started = False


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    verbose_name = 'Toxic Comment Detector'

    def ready(self):
        """
        Hook gọi khi Django khởi động.
        Preload model trong nền để request đầu tiên không phải chờ load.
        """
        global _preload_started

        if os.environ.get('PRELOAD_MODELS', '1') != '1':
            logger.info('Model preload is disabled by PRELOAD_MODELS=0')
            return

        if settings.DEBUG and os.environ.get('RUN_MAIN') != 'true':
            return

        with _preload_lock:
            if _preload_started:
                return
            _preload_started = True

        def _preload_models():
            try:
                from detector.model_loader import model_registry

                model_registry.get_predictor('bilstm').load()
                model_registry.get_predictor('phobert').load()
                logger.info('Model preload completed in background.')
            except Exception:
                logger.exception('Background model preload failed')

        threading.Thread(
            target=_preload_models,
            name='model-preload-thread',
            daemon=True,
        ).start()
