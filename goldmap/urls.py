# C:\Users\shmha\goldmap\goldmap\urls.py

from django.contrib import admin
from django.urls import path, include
from api.views import map_view  # صفحة الخريطة الرئيسية من تطبيق api

urlpatterns = [
    # لوحة الإدارة
    path("admin/", admin.site.urls),

    # الصفحة الرئيسية = خريطة GoldMap AI
    path("", map_view, name="map"),

    # جميع واجهات الـ API تحت المسار /api/
    path("api/", include("api.urls")),
]
