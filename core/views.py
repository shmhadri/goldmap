from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import json
import math
import random
import time
import threading
import logging

import requests

# ===================== إعداد التسجيل (Logging) =====================

logger = logging.getLogger(__name__)

# ===================== صفحة الخريطة =====================

def map_view(request):
    """
    عرض صفحة الخريطة الرئيسية (واجهة GoldMap).
    تمثل HTML في templates/demo/map.html
    """
    return render(request, "demo/map.html")


# ===================== أدوات هندسية =====================

def point_in_poly(lat, lon, poly):
    """
    التحقق من وجود نقطة داخل مضلع باستخدام خوارزمية ray casting.
    poly = [[lat, lon], ...]
    """
    x, y = lon, lat
    inside = False
    n = len(poly)

    for i in range(n):
        x1, y1 = poly[i][1], poly[i][0]
        x2, y2 = poly[(i + 1) % n][1], poly[(i + 1) % n][0]

        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ( (y2 - y1) + 1e-12 ) + x1
        ):
            inside = not inside

    return inside


def _grid_points_in_poly(poly, count):
    """
    إنشاء نقاط باستخدام تقسيم شبكي احتياطي (fallback)
    في حال random_points ما أعطت العدد الكافي.
    """
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    grid_points = []
    grid_size = max(2, int(math.sqrt(count * 4)))

    for i in range(grid_size):
        for j in range(grid_size):
            lat = min_lat + (max_lat - min_lat) * (i + 0.5) / grid_size
            lon = min_lon + (max_lon - min_lon) * (j + 0.5) / grid_size

            if point_in_poly(lat, lon, poly):
                grid_points.append((lat, lon))

    return grid_points[:count]


def random_points_in_poly(poly, count=200):
    """
    إنشاء نقاط عشوائية داخل مضلع مع تحسين الكفاءة
    (محاولات عشوائية + شبكة احتياطية).
    """
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    pts = []
    trials = 0
    max_trials = count * 100  # نسمح بمحاولات أكثر قليلاً

    while len(pts) < count and trials < max_trials:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        if point_in_poly(lat, lon, poly):
            pts.append((lat, lon))
        trials += 1

    # لو ما وصلنا العدد المطلوب نكمل بشبكة
    if len(pts) < count:
        pts.extend(_grid_points_in_poly(poly, count - len(pts)))

    return pts[:count]


# ===================== محرك تحليل جيولوجي (GeologyAnalyzer) =====================

class GeologyAnalyzer:
    """
    محرك تحليل جيولوجي رياضي (تجريبي) يحاكي:
    - النمط الشجري tree
    - النمط الورقي leaf
    - الأودية الداكنة wadis
    - تعدد الألوان multi (أكسدة)
    - الكسور fractures
    مع تعزيز إضافي للدرع العربي والمناطق الجبلية.
    """

    def __init__(self):
        self.weights = {
            "tree": 0.40,
            "leaf": 0.25,
            "wadis": 0.15,
            "multi": 0.12,
            "fractures": 0.08,
        }

    # -------- أنماط التضاريس / الأودية / الأكسدة --------

    def _terrain_roughness(self, lat, lon):
        """حساب خشونة التضاريس (قيمة 0..1)."""
        base = abs(math.sin(lat * 7.3) * math.cos(lon * 5.7))
        detail = math.sin(lat * 13.2) * math.cos(lon * 11.5) * 0.3
        return max(0.0, min(1.0, base + detail))

    def _dendritic_pattern(self, lat, lon):
        """النمط الشجري - تشعبات متفرعة (جذع + أغصان)."""
        main_branches = math.sin((lat + lon) * 6.0) * math.cos(lat * 3.1)
        sub_branches = math.cos((lat - lon) * 8.2) * math.sin(lon * 4.7)
        density = math.sin(lat * 15.0) * 0.2

        score = (
            abs(main_branches) * 0.5
            + max(0.0, sub_branches) * 0.3
            + density * 0.2
        )
        return max(0.0, min(1.0, score))

    def _leafy_pattern(self, lat, lon):
        """النمط الورقي - تشعبات دقيقة متعددة الاتجاهات."""
        pattern1 = math.cos(lat * 9.0) * math.cos(lon * 8.0)
        pattern2 = math.sin(lat * 5.0) * math.sin(lon * 6.0)
        fine_details = math.sin(lat * 18.0 + lon * 12.0) * 0.4

        score = (pattern1 * 0.4 + pattern2 * 0.4 + fine_details * 0.2 + 1.0) / 2.0
        return max(0.0, min(1.0, score))

    def _wadis_pattern(self, lat, lon):
        """أنماط الأودية في المناطق الخشنة."""
        roughness = self._terrain_roughness(lat, lon)
        valley_pattern = math.sin((lat * 12.0) - (lon * 7.0))
        drainage = math.cos(lat * 8.5 + lon * 6.3) * 0.3

        score = roughness * max(0.0, valley_pattern) + drainage
        return max(0.0, min(1.0, abs(score)))

    def _oxidation_pattern(self, lat, lon):
        """أنماط الأكسدة المتعددة الألوان (أصفر/أحمر/برتقالي)."""
        yellow_zones = (math.sin(lat * 11.0) + 1.0) / 2.0
        red_zones = (math.cos(lon * 13.0) + 1.0) / 2.0
        orange_zones = math.sin(lat * 7.0 + lon * 9.0) * 0.5 + 0.5

        score = yellow_zones * 0.4 + red_zones * 0.35 + orange_zones * 0.25
        return max(0.0, min(1.0, score))

    def _fractures_pattern(self, lat, lon):
        """أنماط الكسور والصدوع (خطية ومتقاطعة)."""
        linear_features = abs(math.sin(lon * 10.0)) * 0.6
        cross_features = abs(math.cos(lat * 10.0)) * 0.4
        fault_density = math.sin(lat * 6.0 + lon * 8.0) * 0.3 + 0.3

        score = linear_features + cross_features + fault_density
        return max(0.0, min(1.0, score / 1.5))

    # -------- الدرع العربي + الجبال --------

    def _arabian_shield_bias(self, lat, lon):
        """تحديد هل النقطة داخل نطاق الدرع العربي في السعودية."""
        in_longitude = 36.0 <= lon <= 44.5
        in_latitude = 16.0 <= lat <= 28.5
        return 1.0 if (in_longitude and in_latitude) else 0.0

    def _mountain_analysis(self, lat, lon):
        """تحليل الخصائص الجبلية / الانحدار / الوعورة."""
        roughness = self._terrain_roughness(lat, lon)
        hillshade = (math.cos(lat * 3.3) * math.sin(lon * 2.9)) > 0.25

        return {
            "mountain": roughness > 0.35,
            "hillshade": hillshade,
            "slope": roughness > 0.45,
            "rugged": (roughness + (1.0 if roughness > 0.6 else 0.0)) > 0.8,
            "roughness_value": roughness,
        }

    # -------- تحليل نقطة واحدة --------

    def _generate_reasons(self, scores, flags, shield_bias, strong_traits):
        """توليد أسباب نصية للتحليل (تظهر في الـ popup على الخريطة)."""
        reasons = []

        if shield_bias > 0.0:
            reasons.append("داخل نطاق الدرع العربي - بيئة مواتية للتمعدن")

        if flags["mountain"]:
            reasons.append("منطقة جبلية - توفر الصخور الأساسية")
        if flags["slope"]:
            reasons.append("انحدار مرتفع - يساهم في تركيز المعادن")
        if flags["hillshade"]:
            reasons.append("تباين تضاريسي واضح - مؤشر على التعقيد الجيولوجي")
        if flags["rugged"]:
            reasons.append("تضاريس وعرة - بيئة مناسبة للتمعدن")

        if scores.get("tree", 0) > 0.50:
            reasons.append("ملامح شجرية - أنماط تصريف متفرعة")
        if scores.get("leaf", 0) > 0.50:
            reasons.append("ملامح ورقية - تشعبات دقيقة متعددة")
        if scores.get("wadis", 0) > 0.50:
            reasons.append("تشعبات أودية - أنماط تصريف في مناطق خشنة")
        if scores.get("multi", 0) > 0.55:
            reasons.append("ألوان أكسدة متعددة - مؤشر على تفاعلات كيميائية")
        if scores.get("fractures", 0) > 0.55:
            reasons.append("كسور خطية - هياكل جيولوجية مواتية")

        if strong_traits >= 3:
            reasons.append("تعدد السمات الجيولوجية - مؤشر قوي على الإمكانات")

        return reasons

    def analyze_point(self, lat, lon, patterns):
        """
        تحليل نقطة واحدة:
          - scores لكل نمط
          - flags جبلية
          - confidence 0..1
          - أسباب نصية
          - strong_traits عدد السمات القوية
        """
        requested = set(patterns)

        scores = {}
        if "tree" in requested:
            scores["tree"] = self._dendritic_pattern(lat, lon)
        if "leaf" in requested:
            scores["leaf"] = self._leafy_pattern(lat, lon)
        if "wadis" in requested:
            scores["wadis"] = self._wadis_pattern(lat, lon)
        if "multi" in requested:
            scores["multi"] = self._oxidation_pattern(lat, lon)
        if "fractures" in requested:
            scores["fractures"] = self._fractures_pattern(lat, lon)

        pattern = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        flags = self._mountain_analysis(lat, lon)
        shield_bias = self._arabian_shield_bias(lat, lon)

        # ثقة أساسية
        base_confidence = sum(
            self.weights.get(k, 0.0) * v for k, v in scores.items()
        )
        confidence = base_confidence

        # تعزيز الدرع العربي
        if shield_bias > 0.0:
            confidence = min(1.0, confidence * 1.10 + 0.06)

        # تعزيز المناطق الجبلية
        if flags["mountain"] or flags["slope"] or flags["hillshade"]:
            confidence = min(1.0, confidence * 1.12 + 0.05)

        # حساب السمات القوية
        strong_traits = 0
        strong_traits += 1 if scores.get("tree", 0) > 0.50 else 0
        strong_traits += 1 if scores.get("leaf", 0) > 0.50 else 0
        strong_traits += 1 if scores.get("wadis", 0) > 0.50 else 0
        strong_traits += 1 if scores.get("multi", 0) > 0.55 else 0
        strong_traits += 1 if scores.get("fractures", 0) > 0.55 else 0

        if strong_traits >= 3:
            confidence = min(1.0, confidence * 1.15 + 0.08)
        elif strong_traits >= 2:
            confidence = min(1.0, confidence * 1.08 + 0.04)
        else:
            confidence = max(0.0, confidence * 0.85 - 0.02)

        confidence = max(0.0, min(1.0, confidence))
        reasons = self._generate_reasons(scores, flags, shield_bias, strong_traits)

        return {
            "scores": scores,
            "flags": flags,
            "pattern": pattern,
            "confidence": float(confidence),
            "reasons": reasons,
            "strong_traits": strong_traits,
        }


# مثيل المحلل
analyzer = GeologyAnalyzer()


# ===================== كاشف المناطق المأهولة / المياه (UrbanAreaDetector) =====================

class UrbanAreaDetector:
    """
    كاشف المناطق المأهولة والمياه باستخدام Overpass API + كاش
    - يستبعد: طرق رئيسة، مدن/بلدات، استعمالات سكنية/تجارية/صناعية، مجاري مياه، مسطحات مائية.
    """

    def __init__(self):
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 10 * 60  # 10 دقائق

    def _bbox_key(self, south, north, west, east, zoom):
        return (
            round(south, 3),
            round(north, 3),
            round(west, 3),
            round(east, 3),
            int(max(8, min(zoom, 14))),
        )

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000.0
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        return 2 * R * math.asin(math.sqrt(a))

    def _point_to_segment_distance(self, lat, lon, seg_lat1, seg_lon1, seg_lat2, seg_lon2):
        """
        مسافة نقطة إلى قطعة مستقيمة (تقريبًا بالإسقاط المتري المحلي).
        """
        lat_scale = 111320.0
        lon_scale = 111320.0 * math.cos(math.radians(lat))

        x, y = lon * lon_scale, lat * lat_scale
        x1, y1 = seg_lon1 * lon_scale, seg_lat1 * lat_scale
        x2, y2 = seg_lon2 * lon_scale, seg_lat2 * lat_scale

        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            return math.hypot(x - x1, y - y1)

        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
        x_proj = x1 + t * dx
        y_proj = y1 + t * dy
        return math.hypot(x - x_proj, y - y_proj)

    def _fetch_osm_data(self, south, north, west, east):
        """
        جلب بيانات OSM (طرق + مياه + مدن + مساكن) من Overpass API.
        """
        overpass_url = "https://overpass-api.de/api/interpreter"
        padding = 0.08
        s, n = south - padding, north + padding
        w, e = west - padding, east + padding

        query = f"""
        [out:json][timeout:30];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]({s},{w},{n},{e});
          way["waterway"~"^(river|stream|canal)$"]({s},{w},{n},{e});
          way["natural"="water"]({s},{w},{n},{e});
          way["landuse"="reservoir"]({s},{w},{n},{e});
          relation["natural"="water"]({s},{w},{n},{e});
          node["place"~"^(city|town|village)$"]({s},{w},{n},{e});
          way["landuse"~"^(residential|commercial|industrial)$"]({s},{w},{n},{e});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            response = requests.post(
                overpass_url,
                data=query.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                timeout=25,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"فشل في جلب بيانات OSM: {e}")
            return {"elements": []}

    def _process_osm_data(self, osm_data):
        """
        تحويل بيانات Overpass إلى قوائم:
        - roads: قائمة خطوط
        - waters: قائمة خطوط/مضلعات
        - cities: نقاط
        - urban_areas: مضلعات landuse حضرية
        """
        nodes = {}
        roads = []
        waters = []
        cities = []
        urban_areas = []

        for el in osm_data.get("elements", []):
            if el.get("type") == "node":
                nodes[el["id"]] = (el["lat"], el["lon"])

        for el in osm_data.get("elements", []):
            etype = el.get("type")
            tags = el.get("tags", {})

            if etype == "way":
                coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
                if not coords:
                    continue

                if "highway" in tags:
                    roads.append(coords)
                elif (
                    tags.get("waterway")
                    or tags.get("natural") == "water"
                    or tags.get("landuse") == "reservoir"
                ):
                    waters.append({"line": coords})
                elif tags.get("landuse") in ("residential", "commercial", "industrial"):
                    urban_areas.append(coords)

            elif etype == "node" and tags.get("place") in ("city", "town", "village"):
                cities.append((el["lat"], el["lon"]))

            elif etype == "relation" and tags.get("natural") == "water":
                members = el.get("members", [])
                all_coords = []
                for m in members:
                    if m.get("type") == "way":
                        # نحتاج nodes of that way، غالباً غير متوفرة كاملة هنا
                        # نكتفي بتبسيط: نتجاهل العلاقات المعقدة إن لم تكن مغطاة في الطرق
                        pass
                # يمكن توسيعها لاحقاً

        return {
            "roads": roads,
            "waters": waters,
            "cities": cities,
            "urban_areas": urban_areas,
        }

    def get_urban_mask(self, south, north, west, east, zoom=12):
        """
        الحصول على قناع المناطق الحضرية/المائية مع كاش.
        """
        key = self._bbox_key(south, north, west, east, zoom)
        now = time.time()

        with self.cache_lock:
            cached = self.cache.get(key)
            if cached and now - cached["timestamp"] < self.cache_ttl:
                return cached["data"]

        osm_data = self._fetch_osm_data(south, north, west, east)
        mask = self._process_osm_data(osm_data)

        with self.cache_lock:
            self.cache[key] = {"timestamp": now, "data": mask}

        return mask

    def is_urban_area(
        self,
        lat,
        lon,
        urban_mask,
        road_distance=500,
        city_distance=3000,
        water_distance=200,
    ):
        """
        True لو كانت النقطة قرب:
        - مدينة/بلدة
        - طريق رئيسي
        - مسطح مائي أو مجرى نهر
        - landuse حضري
        """
        # مدن / بلدات
        for c_lat, c_lon in urban_mask.get("cities", []):
            if self._haversine_distance(lat, lon, c_lat, c_lon) <= city_distance:
                return True

        # طرق رئيسية
        for road in urban_mask.get("roads", []):
            for (a_lat, a_lon), (b_lat, b_lon) in zip(road[:-1], road[1:]):
                d = self._point_to_segment_distance(lat, lon, a_lat, a_lon, b_lat, b_lon)
                if d <= road_distance:
                    return True

        # مياه (خطوط/مضلعات مبسّطة)
        for water in urban_mask.get("waters", []):
            if "line" in water:
                for (a_lat, a_lon), (b_lat, b_lon) in zip(
                    water["line"][:-1], water["line"][1:]
                ):
                    d = self._point_to_segment_distance(
                        lat, lon, a_lat, a_lon, b_lat, b_lon
                    )
                    if d <= water_distance:
                        return True
            elif "poly" in water and len(water["poly"]) >= 3:
                if point_in_poly(lat, lon, water["poly"]):
                    return True

        # landuse حضري كمضلع
        for urban_poly in urban_mask.get("urban_areas", []):
            if len(urban_poly) >= 3 and point_in_poly(lat, lon, urban_poly):
                return True

        return False


urban_detector = UrbanAreaDetector()


# ===================== دوال مساعدة GeoJSON =====================

def create_geojson_feature(lon, lat, analysis_result):
    """إنشاء عنصر GeoJSON لنقطة واحدة."""
    return {
        "type": "Feature",
        "properties": {
            "confidence": analysis_result["confidence"],
            "pattern": analysis_result["pattern"],
            "scores": analysis_result["scores"],
            "flags": analysis_result["flags"],
            "reasons": analysis_result["reasons"],
            "strong_traits": analysis_result["strong_traits"],
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon), float(lat)],
        },
    }


def create_feature_collection(features=None):
    """إنشاء FeatureCollection قياسي."""
    return {
        "type": "FeatureCollection",
        "features": features or [],
    }


# ===================== APIs: تحليل المضلع =====================

@csrf_exempt
def predict(request):
    """
    POST /predict
    body: {
      "coordinates": [[lat,lon], ...],
      "patterns": ["tree","leaf","wadis","multi","fractures"],
      "mountainsOnly": true/false,
      "excludeUrban": true/false,
      "count": 200
    }
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST فقط")

    try:
        start_time = time.time()
        payload = json.loads(request.body.decode("utf-8"))

        coords = payload.get("coordinates") or []
        patterns = payload.get("patterns") or [
            "tree",
            "leaf",
            "wadis",
            "multi",
            "fractures",
        ]
        mountains_only = bool(payload.get("mountainsOnly", True))
        exclude_urban = bool(payload.get("excludeUrban", True))
        count = min(500, int(payload.get("count", 250)))

        if not coords or len(coords) < 3:
            return JsonResponse(
                {
                    "error": "مضلع غير صالح",
                    "details": "يجب توفير مضلع بثلاث نقاط على الأقل",
                },
                status=400,
            )

        lats = [p[0] for p in coords]
        lons = [p[1] for p in coords]
        bbox = {
            "south": min(lats),
            "north": max(lats),
            "west": min(lons),
            "east": max(lons),
        }

        # قناع حضري/مائي للمنطقة
        if exclude_urban:
            urban_mask = urban_detector.get_urban_mask(
                bbox["south"], bbox["north"], bbox["west"], bbox["east"], zoom=12
            )
        else:
            urban_mask = {"roads": [], "waters": [], "cities": [], "urban_areas": []}

        analysis_points = random_points_in_poly(coords, count)
        features = []
        analyzed_points = 0

        for lat, lon in analysis_points:
            analyzed_points += 1

            # استبعاد مناطق حضرية أو مياه
            if exclude_urban and urban_detector.is_urban_area(lat, lon, urban_mask):
                continue

            analysis_result = analyzer.analyze_point(lat, lon, patterns)

            # شرط الجبال فقط
            if mountains_only and not (
                analysis_result["flags"]["mountain"]
                or analysis_result["flags"]["slope"]
                or analysis_result["flags"]["hillshade"]
            ):
                continue

            # إسقاط النتائج الضعيفة
            if analysis_result["confidence"] < 0.35:
                continue

            features.append(create_geojson_feature(lon, lat, analysis_result))

            if len(features) >= 600:
                break

        features.sort(
            key=lambda f: f["properties"]["confidence"],
            reverse=True,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"predict: تم تحليل {analyzed_points} نقطة، احتفظنا بـ {len(features)} في {processing_time:.2f}s"
        )

        return JsonResponse(
            {
                "type": "FeatureCollection",
                "features": features,
                "metadata": {
                    "total_points": analyzed_points,
                    "filtered_points": len(features),
                    "processing_time": processing_time,
                    "bbox": bbox,
                },
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "بيانات JSON غير صالحة"}, status=400)
    except Exception as e:
        logger.error(f"خطأ في تحليل المضلع: {e}")
        return JsonResponse(
            {"error": "خطأ في المعالجة", "details": str(e)},
            status=500,
        )


# ===================== APIs: تحليل BBOX =====================

@csrf_exempt
def analyze_bbox_tiles(request):
    """
    POST /analyze-bbox-tiles
    body: {
      "north": .., "south": .., "east": .., "west": ..,
      "zoom": 12,
      "patterns": [...],
      "mountainsOnly": true/false,
      "excludeUrban": true/false,
      "limit": 500
    }
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST فقط")

    try:
        start_time = time.time()
        payload = json.loads(request.body.decode("utf-8"))

        north = float(payload.get("north", 0))
        south = float(payload.get("south", 0))
        east = float(payload.get("east", 0))
        west = float(payload.get("west", 0))
        zoom = int(payload.get("zoom", 12))
        patterns = payload.get("patterns") or [
            "tree",
            "leaf",
            "wadis",
            "multi",
            "fractures",
        ]
        mountains_only = bool(payload.get("mountainsOnly", True))
        exclude_urban = bool(payload.get("excludeUrban", True))
        limit = min(800, int(payload.get("limit", 500)))

        if not (south < north and west < east):
            return JsonResponse(
                {
                    "error": "مربع محيط غير صالح",
                    "details": "يجب أن تكون القيم متناسقة (شمال > جنوب، شرق > غرب)",
                },
                status=400,
            )

        if exclude_urban:
            urban_mask = urban_detector.get_urban_mask(south, north, west, east, zoom)
        else:
            urban_mask = {"roads": [], "waters": [], "cities": [], "urban_areas": []}

        base_density = max(8, min(20, zoom - 8))
        cols = max(10, int((east - west) * base_density))
        rows = max(8, int((north - south) * base_density))

        step_lat = (north - south) / max(rows, 1)
        step_lon = (east - west) / max(cols, 1)

        features = []
        analyzed_points = 0

        lat = south + step_lat / 2.0
        while lat < north and len(features) < limit:
            lon = west + step_lon / 2.0
            while lon < east and len(features) < limit:
                analyzed_points += 1

                if exclude_urban and urban_detector.is_urban_area(lat, lon, urban_mask):
                    lon += step_lon
                    continue

                analysis_result = analyzer.analyze_point(lat, lon, patterns)

                if mountains_only and not (
                    analysis_result["flags"]["mountain"]
                    or analysis_result["flags"]["slope"]
                    or analysis_result["flags"]["hillshade"]
                ):
                    lon += step_lon
                    continue

                if analysis_result["strong_traits"] < 2 or analysis_result[
                    "confidence"
                ] < 0.4:
                    lon += step_lon
                    continue

                features.append(create_geojson_feature(lon, lat, analysis_result))
                lon += step_lon

            lat += step_lat

        features.sort(
            key=lambda f: f["properties"]["confidence"],
            reverse=True,
        )
        features = features[:limit]

        processing_time = time.time() - start_time
        logger.info(
            f"analyze_bbox_tiles: تم تحليل {analyzed_points} نقطة، احتفظنا بـ {len(features)} في {processing_time:.2f}s"
        )

        return JsonResponse(
            {
                "type": "FeatureCollection",
                "features": features,
                "metadata": {
                    "total_points": analyzed_points,
                    "filtered_points": len(features),
                    "processing_time": processing_time,
                    "bbox": {
                        "south": south,
                        "north": north,
                        "west": west,
                        "east": east,
                    },
                },
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "بيانات JSON غير صالحة"}, status=400)
    except Exception as e:
        logger.error(f"خطأ في تحليل المربع المحيط: {e}")
        return JsonResponse(
            {"error": "خطأ في المعالجة", "details": str(e)},
            status=500,
        )


# ===================== API: البحث عن مكان (Nominatim) =====================

@csrf_exempt
def analyze_place(request):
    """
    GET /analyze-place?q=اسم
    يرجع مركز المكان + bbox عبر Nominatim (داخل السعودية).
    """
    query = (request.GET.get("q") or "").strip()
    if not query:
        return JsonResponse(
            {
                "error": "معامل البحث مطلوب",
                "details": "يجب توفير معامل 'q' للبحث",
            },
            status=400,
        )

    try:
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "format": "json",
            "q": query,
            "accept-language": "ar",
            "countrycodes": "sa",
            "limit": 1,
        }
        headers = {
            "User-Agent": "GoldMap-Pro/1.0 (https://example.com/goldmap)"  # عدّلها حسب مشروعك
        }

        response = requests.get(
            nominatim_url,
            params=params,
            headers=headers,
            timeout=15,
        )

        if not response.ok:
            return JsonResponse(
                {
                    "found": False,
                    "error": "فشل في الاتصال بخدمة البحث",
                }
            )

        data = response.json()
        if not data:
            return JsonResponse({"found": False})

        place = data[0]
        bbox_vals = [float(c) for c in place["boundingbox"]]

        return JsonResponse(
            {
                "found": True,
                "name": place.get("display_name", ""),
                "center": {
                    "lat": float(place["lat"]),
                    "lon": float(place["lon"]),
                },
                "bbox": {
                    "south": bbox_vals[0],
                    "north": bbox_vals[1],
                    "west": bbox_vals[2],
                    "east": bbox_vals[3],
                },
                "type": place.get("type", ""),
                "importance": float(place.get("importance", 0)),
            }
        )

    except requests.Timeout:
        return JsonResponse(
            {
                "found": False,
                "error": "انتهت مهلة البحث",
            }
        )
    except Exception as e:
        logger.error(f"خطأ في البحث عن المكان: {e}")
        return JsonResponse(
            {
                "found": False,
                "error": "حدث خطأ أثناء البحث",
            }
        )


# ===================== API: تصدير GeoJSON =====================

@csrf_exempt
def export_geojson(request):
    """
    POST /export-geojson
    body: FeatureCollection
    يرجع ملف goldmap_analysis_results.geojson للتنزيل.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST فقط")

    try:
        geojson_data = json.loads(request.body.decode("utf-8"))

        if not isinstance(geojson_data, dict) or geojson_data.get("type") != "FeatureCollection":
            return JsonResponse(
                {
                    "error": "بيانات GeoJSON غير صالحة",
                    "details": "يجب أن تكون البيانات بصيغة FeatureCollection",
                },
                status=400,
            )

        response = HttpResponse(
            json.dumps(geojson_data, ensure_ascii=False, indent=2),
            content_type="application/geo+json; charset=utf-8",
        )
        response["Content-Disposition"] = (
            'attachment; filename="goldmap_analysis_results.geojson"'
        )
        return response

    except json.JSONDecodeError:
        return JsonResponse({"error": "بيانات JSON غير صالحة"}, status=400)
    except Exception as e:
        logger.error(f"خطأ في تصدير GeoJSON: {e}")
        return JsonResponse(
            {"error": "خطأ في التصدير", "details": str(e)},
            status=500,
        )


# ===================== API: حفظ النتائج (وهمية الآن) =====================

@csrf_exempt
def save_hotspots(request):
    """
    POST /save-hotspots
    body: {"data": FeatureCollection}
    حاليًا: يحسب عدد النقاط ويعيد رقم منطقة وهمي.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST فقط")

    try:
        payload = json.loads(request.body.decode("utf-8"))
        features = (payload.get("data") or {}).get("features", [])

        return JsonResponse(
            {
                "success": True,
                "region_id": random.randint(1000, 9999),
                "saved_points": len(features),
                "message": "تم حفظ النتائج بنجاح",
            }
        )

    except Exception as e:
        logger.error(f"خطأ في حفظ النقاط الساخنة: {e}")
        return JsonResponse(
            {
                "success": False,
                "error": "فشل في حفظ البيانات",
            },
            status=500,
        )


# ===================== واجهات إضافية: health / clear cache =====================

def health_check(request):
    """
    GET /health
    لفحص صحة الخدمة.
    """
    return JsonResponse(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {
                "geology_analyzer": "active",
                "urban_detector": "active",
                "cache": "active",
            },
        }
    )


@csrf_exempt
def clear_cache(request):
    """
    POST /clear-cache
    لمسح كاش UrbanAreaDetector.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST فقط")

    try:
        with urban_detector.cache_lock:
            cleared = len(urban_detector.cache)
            urban_detector.cache.clear()

        return JsonResponse(
            {
                "success": True,
                "message": "تم مسح ذاكرة التخزين المؤقت بنجاح",
                "cleared_entries": cleared,
            }
        )

    except Exception as e:
        logger.error(f"خطأ في مسح الذاكرة المؤقتة: {e}")
        return JsonResponse(
            {
                "success": False,
                "error": "فشل في مسح الذاكرة المؤقتة",
            },
            status=500,
        )
