from django.urls import path
from detector import views

urlpatterns = [
    path('health/', views.health_check, name='health-check'),
    path('models/', views.list_models, name='list-models'),
    path('predict/', views.predict_single, name='predict-single'),
    path('predict/batch/', views.predict_batch, name='predict-batch'),
]
