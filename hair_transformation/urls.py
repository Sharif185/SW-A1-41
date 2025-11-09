from django.urls import path
from .views import HomeView, ProcessingView, ResultsView, AjaxProcessingView

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('upload/', HomeView.as_view(), name='upload'),
    path('processing/<str:session_id>/', ProcessingView.as_view(), name='processing_view'),
    path('results/<str:session_id>/', ResultsView.as_view(), name='results_view'),
    path('ajax/status/<str:session_id>/', AjaxProcessingView.as_view(), name='ajax_status'),
]