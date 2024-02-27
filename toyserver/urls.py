from django.contrib import admin
from django.urls import path, include  # include 함수를 임포트

urlpatterns = [
    path('admin/', admin.site.urls),
    path('mainapp/', include('mainapp.urls')),  # mainapp의 urls.py 파일을 연동
]
