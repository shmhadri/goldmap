# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # تحليل مضلع مرسوم على الخريطة (Polygon)
    path("predict/", views.predict, name="predict"),

    # تحليل مربّع/نافذة (BBox) على مستوى أوسع
    path("analyze_bbox_tiles/", views.analyze_bbox_tiles, name="analyze_bbox_tiles"),

    # البحث عن مكان (مدينة/وادي/منجم) وإرجاع مركزه وحدوده
    path("analyze_place/", views.analyze_place, name="analyze_place"),

    # تصدير نتائج التحليل بصيغة GeoJSON قابلة للتحميل
    path("export_geojson/", views.export_geojson, name="export_geojson"),

    # حفظ النقاط الساخنة (hotspots) في قاعدة البيانات أو أي مخزن
    path("save_hotspots/", views.save_hotspots, name="save_hotspots"),
]
