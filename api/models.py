# api/models.py
from django.db import models


class AnalysisRegion(models.Model):
    """
    يمثل منطقة تحليل تم تنفيذ خوارزميات GoldMap عليها،
    ويتم تخزين نتائج التحليل بصيغة GeoJSON.
    """

    name = models.CharField(
        max_length=200,
        blank=True,
        help_text="اسم اختياري للمنطقة (مثلاً: وادي الذهب – تحليل 1)"
    )

    polygon = models.JSONField(
        help_text="مضلع التحليل (قائمة إحداثيات lat/lon).",
        blank=True,
        null=True
    )

    geojson = models.JSONField(
        help_text="نتائج التحليل كاملة بصيغة GeoJSON.",
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="وقت إنشاء التحليل."
    )

    def __str__(self):
        return self.name or f"تحليل #{self.pk}"
