# api/views.py
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


def _fc(features: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """
    يبني هيكل FeatureCollection بسيط متوافق مع GeoJSON.
    يُستخدم كاستجابة افتراضية أو هيكل أساسي لنتائج التحليل.
    """
    return {"type": "FeatureCollection", "features": features or []}


@csrf_exempt
def predict(request):
    """
    API تجريبية لتحليل مضلع مرسوم على الخريطة.
    حاليًا ترجع FeatureCollection فارغة (بدون تحليل حقيقي).
    لاحقًا سيتم ربطها بمحرك التحليل الجيولوجي (GoldMap Analyzer).
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام POST لهذه الواجهة")

    try:
        body = request.body.decode("utf-8") or "{}"
        payload = json.loads(body)

        # هنا لاحقًا: قراءة payload["coordinates"], ["patterns"], ... وتمريرها للمحلّل
        # الآن نرجع فقط هيكل فارغ مع ميتا بسيطة
        return JsonResponse(
            {
                **_fc(),
                "metadata": {
                    "message": "نقطة دخل صحيحة، ولكن التحليل غير مفعل بعد.",
                    "received_keys": list(payload.keys()),
                },
            }
        )
    except json.JSONDecodeError:
        return HttpResponseBadRequest("تنسيق JSON غير صالح")
    except Exception as e:
        logger.exception("خطأ غير متوقع في واجهة predict")
        return HttpResponseBadRequest(f"error: {e}")


@csrf_exempt
def analyze_bbox_tiles(request):
    """
    API تجريبية لتحليل نافذة (BBox) من الخريطة.
    حاليًا ترجع FeatureCollection فارغة.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام POST لهذه الواجهة")

    try:
        body = request.body.decode("utf-8") or "{}"
        payload = json.loads(body)

        # لاحقًا: استخدام north/south/east/west/zoom من payload للتحليل
        return JsonResponse(
            {
                **_fc(),
                "metadata": {
                    "message": "تحليل المربّع غير مفعّل بعد (نسخة تجريبية).",
                    "received_keys": list(payload.keys()),
                },
            }
        )
    except json.JSONDecodeError:
        return HttpResponseBadRequest("تنسيق JSON غير صالح")
    except Exception as e:
        logger.exception("خطأ غير متوقع في واجهة analyze_bbox_tiles")
        return HttpResponseBadRequest(f"error: {e}")


def analyze_place(request):
    """
    API تجريبية للبحث عن مكان (مدينة/وادي/منجم).
    حاليًا ترجع قيمة found=False بشكل ثابت.
    لاحقًا يمكن ربطها بـ Nominatim أو خدمة أخرى للبحث الجغرافي.
    """
    q = (request.GET.get("q") or "").strip()
    if not q:
        return JsonResponse(
            {
                "found": False,
                "error": "يجب تمرير معامل q في الاستعلام، مثلاً: ?q=أبها",
            },
            status=400,
        )

    # لاحقًا: طلب حقيقي لـ Nominatim وإرجاع center, bbox, ...
    return JsonResponse(
        {
            "found": False,
            "center": {"lat": 0.0, "lon": 0.0},
            "message": "البحث الحقيقي غير مفعّل بعد (نسخة تجريبية).",
            "query": q,
        }
    )


@csrf_exempt
def export_geojson(request):
    """
    تصدير GeoJSON من الواجهة.
    الآن: نتأكد أن الـ JSON القادم سليم، ثم نرجعه كما هو.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام POST لهذه الواجهة")

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")

        # لو ما فيه type=FeatureCollection نعيد خطأ بسيط
        if payload.get("type") != "FeatureCollection":
            return JsonResponse(
                {
                    "ok": False,
                    "error": "يجب أن يكون type = 'FeatureCollection'",
                },
                status=400,
            )

        data_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        resp = HttpResponse(
            data_bytes,
            content_type="application/geo+json; charset=utf-8",
        )
        resp["Content-Disposition"] = 'attachment; filename="goldmap_results.geojson"'
        return resp
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "تنسيق JSON غير صالح"}, status=400)
    except Exception as e:
        logger.exception("خطأ في export_geojson")
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


@csrf_exempt
def save_hotspots(request):
    """
    حفظ النقاط الساخنة (نتائج التحليل).
    حاليًا: لا يوجد حفظ حقيقي في قاعدة بيانات، فقط استجابة وهمية.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام POST لهذه الواجهة")

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
        features = (payload.get("data") or {}).get("features", [])

        # لاحقًا: إنشاء سجل AnalysisRegion وربطه بهذه الميزات
        saved_count = len(features)

        return JsonResponse(
            {
                "ok": True,
                "region_id": 1,  # لاحقًا: ID حقيقي من قاعدة البيانات
                "saved": saved_count,
                "message": "واجهة حفظ تجريبية - لم يتم الحفظ فعليًا في قاعدة البيانات.",
            }
        )
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "تنسيق JSON غير صالح"}, status=400)
    except Exception as e:
        logger.exception("خطأ في save_hotspots")
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
