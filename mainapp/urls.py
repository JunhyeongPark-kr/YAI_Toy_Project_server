from django.urls import path
from .views import input_text_view

urlpatterns = [
    path('', input_text_view, name='input-text'),
]
