from django.urls import path
from . import views

urlpatterns = [
    path("", views.map_view, name="map"),
    path("api/predict/", views.predict, name="api_predict"),
    path("api/analyze_bbox_tiles/", views.analyze_bbox_tiles, name="api_analyze_bbox_tiles"),
    path("api/analyze_place/", views.analyze_place, name="api_analyze_place"),
    path("api/export_geojson/", views.export_geojson, name="api_export_geojson"),
    path("api/save_hotspots/", views.save_hotspots, name="api_save_hotspots"),
]
