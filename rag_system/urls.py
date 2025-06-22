from django.urls import path
from . import views

urlpatterns = [
    path('send/', views.RAGAPIView.as_view(), name='rag-endpoint'),
    path('query/', views.QueryAPIView.as_view(), name='query-endpoint'),
    path('history/', views.HistoryAPIView.as_view(), name='history-endpoint'),
] 