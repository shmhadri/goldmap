# C:\Users\shmha\goldmap\api\urls.py

from django.urls import path
from . import views

urlpatterns = [
    # تحليل مضلع مرسوم
    path("predict/", views.predict, name="api_predict"),

    # تحليل نافذة (BBox)
    # (إعادة التسمية لتطابق الواجهة الأمامية)
    path("analyze_bbox_tiles/", views.analyze_bbox_tiles, name="api_analyze_bbox_tiles"),
    path("analyze-bbox-tiles/", views.analyze_bbox_tiles, name="api_analyze_bbox_tiles_dash"),

    # البحث عن المدن/القرى/الأودية
    path("analyze_place/", views.analyze_place, name="api_analyze_place"),

    # تصدير GeoJSON
    path("export_geojson/", views.export_geojson, name="api_export_geojson"),

    # حفظ النتائج (اختياري)
    path("save_hotspots/", views.save_hotspots, name="api_save_hotspots"),

    # فحص النظام
    path("health/", views.health_check, name="api_health"),

    # مسح كاش (اختياري)
    path("clear-cache/", views.clear_cache, name="api_clear_cache"),
]
