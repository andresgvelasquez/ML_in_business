from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DatasetViewSet, DatasetList, DatasetHeadView, upload_file

# Configura el router para las vistas de tu API
router = DefaultRouter()
router.register(r'datasets', DatasetViewSet, basename='dataset')

urlpatterns = [
    path('', include(router.urls)),
    path('datasets/list/', DatasetList.as_view(), name='dataset-list'),
    path('datasets/head/', DatasetHeadView.as_view(), name='dataset-head'),
    path('upload/', upload_file, name='upload_file'),
]