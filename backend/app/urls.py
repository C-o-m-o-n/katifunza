from django.http import HttpResponse
from django.urls import path
from . import views

urlpatterns = [
    path('prompt/', views.AgentPromptView.as_view(), name='agent-prompt'),
]
