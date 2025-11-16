# C:\Users\shmha\goldmap\core\urls.py

from django.contrib import admin
from django.urls import path, include
from api.views import map_view   # الصفحة الرئيسية

urlpatterns = [
    # لوحة الإدارة
    path("admin/", admin.site.urls),

    # الصفحة الرئيسية (الخريطة)
    path("", map_view, name="map"),

    # جميع مسارات API تأتي من api/urls.py
    path("api/", include("api.urls")),
]
