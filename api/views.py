from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import json
import math
import random
import time
import threading
import logging

from typing import List, Dict, Any

import requests
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ===================== صفحة الخريطة (الواجهة الأمامية) =====================


def map_view(request):
    """إرجاع صفحة الخريطة الرئيسية (واجهة GoldMap)."""
    return render(request, "demo/map.html")


# ===================== أدوات GeoJSON مساعدة =====================


def _fc(features: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """FeatureCollection بسيط."""
    return {"type": "FeatureCollection", "features": features or []}


def create_geojson_feature(
    lon: float, lat: float, analysis_result: Dict[str, Any]
) -> Dict[str, Any]:
    """إنشاء عنصر GeoJSON لنقطة واحدة مع تفاصيل الثقة والأسباب."""
    confidence = float(analysis_result.get("confidence", 0.0))
    return {
        "type": "Feature",
        "properties": {
            "confidence": confidence,
            "confidence_percent": int(round(confidence * 100)),
            "pattern": analysis_result.get("pattern"),
            "scores": analysis_result.get("scores", {}),
            "flags": analysis_result.get("flags", {}),
            "reasons": analysis_result.get("reasons", []),
            "reason_details": analysis_result.get("reason_details", []),
            "strong_traits": analysis_result.get("strong_traits", 0),
            "gold_signature_score": float(
                analysis_result.get("gold_signature_score", 0.0)
            ),
            "gold_signature_like": bool(
                analysis_result.get("gold_signature_like", False)
            ),
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon), float(lat)],
        },
    }


def create_feature_collection(
    features: List[Dict[str, Any]] | None = None
) -> Dict[str, Any]:
    """إنشاء FeatureCollection قياسي."""
    return {"type": "FeatureCollection", "features": features or []}


# ===================== أدوات هندسية (نقاط داخل مضلع) =====================


def point_in_poly(lat: float, lon: float, poly: List[List[float]]) -> bool:
    """التحقق من وجود نقطة داخل مضلع (ray casting). poly = [[lat, lon], ...]."""
    x, y = lon, lat
    inside = False
    n = len(poly)

    for i in range(n):
        x1, y1 = poly[i][1], poly[i][0]
        x2, y2 = poly[(i + 1) % n][1], poly[(i + 1) % n][0]

        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
        ):
            inside = not inside

    return inside


def _grid_points_in_poly(poly: List[List[float]], count: int) -> List[tuple]:
    """شبكة احتياطية لنقاط داخل المضلع لو العشوائي لم يكفِ."""
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    grid_points: List[tuple] = []
    grid_size = max(2, int(math.sqrt(count * 3)))

    for i in range(grid_size):
        for j in range(grid_size):
            lat = min_lat + (max_lat - min_lat) * (i + 0.5) / grid_size
            lon = min_lon + (max_lon - min_lon) * (j + 0.5) / grid_size

            if point_in_poly(lat, lon, poly):
                grid_points.append((lat, lon))

    return grid_points[:count]


def random_points_in_poly(poly: List[List[float]], count: int = 120) -> List[tuple]:
    """
    نقاط عشوائية داخل مضلع مع محاولات محدودة + شبكة احتياطية.
    العدد الافتراضي أقل لتخفيف التحليل.
    """
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    pts: List[tuple] = []
    trials = 0
    max_trials = count * 60  # أقل من السابق لتقليل الوقت

    while len(pts) < count and trials < max_trials:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        if point_in_poly(lat, lon, poly):
            pts.append((lat, lon))
        trials += 1

    if len(pts) < count:
        pts.extend(_grid_points_in_poly(poly, count - len(pts)))

    return pts[:count]


# ===================== كاشف البحر (تقريبي) =====================


def is_sea(lat: float, lon: float) -> bool:
    """
    استبعاد البحر الأحمر والخليج (تقريبياً):
    - البحر الأحمر: غرب السعودية تقريباً (lon < 38.8)
    - الخليج العربي: شرق السعودية تقريباً (lon > 50.5)
    هذا يحمي من التحليل فوق البحر بدون استبعاد اليابسة.
    """
    # البحر الأحمر
    if 16.0 <= lat <= 28.8 and lon < 38.8:
        return True

    # الخليج العربي
    if 24.0 <= lat <= 29.5 and lon > 50.5:
        return True

    return False


# ===================== محرك التحليل الجيولوجي (مبسّط) =====================


class GeologyAnalyzer:
    """
    محرك تحليلي مبسّط يعتمد على:
    - نمط شجري / ورقي / أودية / أكسدة / كسور.
    - تفضيل الدرع العربي وحواف الجبال.
    """

    def __init__(self):
        self.weights = {
            "tree": 0.40,
            "leaf": 0.25,
            "wadis": 0.15,
            "multi": 0.12,
            "fractures": 0.08,
        }

    # -------- أنماط سطحية مبسّطة --------

    def _terrain_roughness(self, lat: float, lon: float) -> float:
        base = abs(math.sin(lat * 7.3) * math.cos(lon * 5.7))
        detail = math.sin(lat * 13.2) * math.cos(lon * 11.5) * 0.3
        return max(0.0, min(1.0, base + detail))

    def _dendritic_pattern(self, lat: float, lon: float) -> float:
        main_branches = math.sin((lat + lon) * 6.0) * math.cos(lat * 3.1)
        sub_branches = math.cos((lat - lon) * 8.2) * math.sin(lon * 4.7)
        density = math.sin(lat * 15.0) * 0.2
        score = (
            abs(main_branches) * 0.5 + max(0.0, sub_branches) * 0.3 + density * 0.2
        )
        return max(0.0, min(1.0, score))

    def _leafy_pattern(self, lat: float, lon: float) -> float:
        pattern1 = math.cos(lat * 9.0) * math.cos(lon * 8.0)
        pattern2 = math.sin(lat * 5.0) * math.sin(lon * 6.0)
        fine_details = math.sin(lat * 18.0 + lon * 12.0) * 0.4
        score = (pattern1 * 0.4 + pattern2 * 0.4 + fine_details * 0.2 + 1.0) / 2.0
        return max(0.0, min(1.0, score))

    def _wadis_pattern(self, lat: float, lon: float) -> float:
        roughness = self._terrain_roughness(lat, lon)
        valley_pattern = math.sin((lat * 12.0) - (lon * 7.0))
        drainage = math.cos(lat * 8.5 + lon * 6.3) * 0.3
        score = roughness * max(0.0, valley_pattern) + drainage
        return max(0.0, min(1.0, abs(score)))

    def _oxidation_pattern(self, lat: float, lon: float) -> float:
        yellow_zones = (math.sin(lat * 11.0) + 1.0) / 2.0
        red_zones = (math.cos(lon * 13.0) + 1.0) / 2.0
        orange_zones = math.sin(lat * 7.0 + lon * 9.0) * 0.5 + 0.5
        score = yellow_zones * 0.4 + red_zones * 0.35 + orange_zones * 0.25
        return max(0.0, min(1.0, score))

    def _fractures_pattern(self, lat: float, lon: float) -> float:
        linear_features = abs(math.sin(lon * 10.0)) * 0.6
        cross_features = abs(math.cos(lat * 10.0)) * 0.4
        fault_density = math.sin(lat * 6.0 + lon * 8.0) * 0.3 + 0.3
        score = linear_features + cross_features + fault_density
        return max(0.0, min(1.0, score / 1.5))

    # -------- الدرع العربي + الجبال --------

    def _arabian_shield_zone(self, lat: float, lon: float) -> str:
        """inside / margin / outside بالنسبة للدرع العربي (تقريبي)."""
        inside_lon_min, inside_lon_max = 36.0, 44.5
        inside_lat_min, inside_lat_max = 16.0, 28.5

        margin_lon_min, margin_lon_max = 35.0, 46.0
        margin_lat_min, margin_lat_max = 15.0, 30.0

        if (inside_lon_min <= lon <= inside_lon_max) and (
            inside_lat_min <= lat <= inside_lat_max
        ):
            return "inside"

        if (margin_lon_min <= lon <= margin_lon_max) and (
            margin_lat_min <= lat <= margin_lat_max
        ):
            return "margin"

        return "outside"

    def _arabian_shield_bias(self, lat: float, lon: float) -> float:
        zone = self._arabian_shield_zone(lat, lon)
        if zone == "inside":
            return 1.0
        if zone == "margin":
            return 0.4
        return 0.0

    def _mountain_analysis(self, lat: float, lon: float) -> Dict[str, Any]:
        roughness = self._terrain_roughness(lat, lon)
        hillshade = (math.cos(lat * 3.3) * math.sin(lon * 2.9)) > 0.25
        transition_zone = 0.32 < roughness < 0.62
        return {
            "mountain": roughness > 0.35,
            "hillshade": hillshade,
            "slope": roughness > 0.45,
            "rugged": (roughness + (1.0 if roughness > 0.6 else 0.0)) > 0.8,
            "transition_zone": transition_zone,
            "roughness_value": roughness,
        }

    # -------- توقيع ذهبي مبسّط --------

    def _gold_signature_score(
        self,
        lat: float,
        lon: float,
        scores: Dict[str, float],
        flags: Dict[str, Any],
        shield_bias: float,
    ) -> float:
        s_tree = scores.get("tree", 0.0)
        s_wadis = scores.get("wadis", 0.0)
        s_multi = scores.get("multi", 0.0)
        s_fract = scores.get("fractures", 0.0)

        score = 0.0

        if shield_bias > 0.0:
            score += 0.25 * shield_bias

        if s_tree > 0.55 and s_wadis > 0.45:
            score += 0.25

        if s_multi > 0.45:
            score += 0.20

        if s_fract > 0.50:
            score += 0.20

        if flags.get("rugged") and flags.get("transition_zone"):
            score += 0.10

        return max(0.0, min(1.0, score))

    # -------- أسباب مبسّطة --------

    def _generate_reasons(
        self,
        scores: Dict[str, float],
        flags: Dict[str, Any],
        shield_bias: float,
        strong_traits: int,
        gold_signature: float,
    ) -> tuple[list[str], list[Dict[str, Any]]]:
        reasons: List[str] = []
        details: List[Dict[str, Any]] = []

        def add_reason(label: str, score: float):
            s = float(max(0.0, min(1.0, score)))
            reasons.append(label)
            details.append(
                {"label": label, "score": round(s, 3), "percent": int(round(s * 100))}
            )

        if shield_bias > 0.0:
            add_reason("داخل نطاق الدرع العربي", 0.85 * shield_bias)

        if flags.get("mountain"):
            add_reason("منطقة جبلية", 0.7)
        if flags.get("slope"):
            add_reason("انحدار مرتفع", 0.65)
        if flags.get("hillshade"):
            add_reason("تباين تضاريسي واضح", 0.6)
        if flags.get("rugged"):
            add_reason("تضاريس وعرة", 0.75)
        if flags.get("transition_zone"):
            add_reason("منطقة انتقالية جبل/وادي", 0.8)

        if scores.get("tree", 0) > 0.50:
            add_reason("ملامح شجرية قوية", scores["tree"])
        if scores.get("leaf", 0) > 0.50:
            add_reason("ملامح ورقية دقيقة", scores["leaf"])
        if scores.get("wadis", 0) > 0.50:
            add_reason("شبكة أودية كثيفة", scores["wadis"])
        if scores.get("multi", 0) > 0.55:
            add_reason("هالات أكسدة حمراء/صفراء", scores["multi"])
        if scores.get("fractures", 0) > 0.55:
            add_reason("كسور وفوالق خطية", scores["fractures"])

        if gold_signature >= 0.6:
            add_reason("نمط يشبه مواقع ذهب حراري معروفة", gold_signature)

        if strong_traits >= 3:
            add_reason("تعدد سمات قوية في نفس النقطة", 0.85)
        elif strong_traits >= 2:
            add_reason("وجود أكثر من سمة متوسطة القوة", 0.65)

        return reasons, details

    # -------- تحليل نقطة واحدة --------

    def analyze_point(
        self, lat: float, lon: float, patterns: List[str]
    ) -> Dict[str, Any]:
        """
        تحليل نقطة واحدة:
        - scores لكل نمط
        - flags جبلية/انتقالية
        - gold_signature_score
        - confidence 0..1
        - أسباب + تفاصيل
        """
        requested = set(patterns)

        scores: Dict[str, float] = {}
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
        shield_zone = self._arabian_shield_zone(lat, lon)
        flags["shield_zone"] = shield_zone

        base_confidence = sum(self.weights.get(k, 0.0) * v for k, v in scores.items())
        gold_signature = self._gold_signature_score(lat, lon, scores, flags, shield_bias)

        confidence = base_confidence + 0.15 * gold_signature

        if shield_bias > 0.0:
            confidence = min(
                1.0, confidence * (1.05 + 0.05 * shield_bias) + 0.05 * shield_bias
            )

        if flags["mountain"] or flags["slope"] or flags["hillshade"]:
            confidence = min(1.0, confidence * 1.10 + 0.04)
        if flags["transition_zone"]:
            confidence = min(1.0, confidence * 1.06 + 0.03)

        strong_traits = 0
        strong_traits += 1 if scores.get("tree", 0) > 0.50 else 0
        strong_traits += 1 if scores.get("leaf", 0) > 0.50 else 0
        strong_traits += 1 if scores.get("wadis", 0) > 0.50 else 0
        strong_traits += 1 if scores.get("multi", 0) > 0.55 else 0
        strong_traits += 1 if scores.get("fractures", 0) > 0.55 else 0

        if strong_traits >= 3:
            confidence = min(1.0, confidence * 1.12 + 0.06)
        elif strong_traits >= 2:
            confidence = min(1.0, confidence * 1.05 + 0.03)
        else:
            confidence = max(0.0, confidence * 0.80 - 0.03)

        confidence = max(0.0, min(1.0, confidence))

        reasons, reason_details = self._generate_reasons(
            scores, flags, shield_bias, strong_traits, gold_signature
        )

        return {
            "scores": scores,
            "flags": flags,
            "pattern": pattern,
            "confidence": float(confidence),
            "reasons": reasons,
            "reason_details": reason_details,
            "strong_traits": strong_traits,
            "gold_signature_score": float(gold_signature),
            "gold_signature_like": bool(gold_signature >= 0.6),
        }


analyzer = GeologyAnalyzer()


def _accept_geology_point(analysis_result: Dict[str, Any]) -> bool:
    """
    فلتر نهائي:
    - داخل الدرع: شروط متوسطة.
    - الهامش: أعلى.
    - خارج الدرع: صارم جداً (نادر).
    """
    zone = analysis_result.get("flags", {}).get("shield_zone", "outside")
    conf = float(analysis_result.get("confidence", 0.0))
    strong = int(analysis_result.get("strong_traits", 0))
    scores = analysis_result.get("scores", {}) or {}
    tree = float(scores.get("tree", 0.0))
    leaf = float(scores.get("leaf", 0.0))

    if zone == "inside":
        if conf < 0.50 or strong < 3:
            return False
        if max(tree, leaf) < 0.50:
            return False
        return True

    if zone == "margin":
        if conf < 0.60 or strong < 3:
            return False
        if max(tree, leaf) < 0.55:
            return False
        return True

    if conf < 0.80 or strong < 4:
        return False
    if tree < 0.65 or leaf < 0.55:
        return False
    return True


# ===================== كاشف المناطق المأهولة / المزارع / المياه =====================


class UrbanAreaDetector:
    """
    كاشف المدن / الطرق / المياه / المزارع باستخدام Overpass.
    يُستخدم لمنع ظهور نقاط على المدن والبحر والمزارع.
    """

    def __init__(self):
        self.cache: Dict[Any, Any] = {}
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

    def _point_to_segment_distance(
        self, lat, lon, seg_lat1, seg_lon1, seg_lat2, seg_lon2
    ):
        """مسافة نقطة إلى قطعة مستقيمة (تقريب متري محلي)."""
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
        """جلب بيانات OSM من Overpass."""
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
          way["natural"="coastline"]({s},{w},{n},{e});
          relation["natural"="water"]({s},{w},{n},{e});
          relation["natural"="coastline"]({s},{w},{n},{e});
          node["place"="sea"]({s},{w},{n},{e});
          way["place"="sea"]({s},{w},{n},{e});
          relation["place"="sea"]({s},{w},{n},{e});

          node["place"~"^(city|town|village)$"]({s},{w},{n},{e});

          way["landuse"~"^(residential|commercial|industrial)$"]({s},{w},{n},{e});
          way["landuse"~"^(farmland|farm|orchard|vineyard|meadow)$"]({s},{w},{n},{e});
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
        """تحويل بيانات Overpass إلى قوائم بسيطة."""
        nodes: Dict[int, tuple] = {}
        roads: List[List[tuple]] = []
        waters: List[Dict[str, Any]] = []
        cities: List[tuple] = []
        urban_areas: List[List[tuple]] = []

        for el in osm_data.get("elements", []):
            if el.get("type") == "node":
                nodes[el["id"]] = (el["lat"], el["lon"])

        for el in osm_data.get("elements", []):
            etype = el.get("type")
            tags = el.get("tags", {}) or {}

            if etype == "way":
                coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
                if not coords:
                    continue

                if "highway" in tags:
                    roads.append(coords)
                elif (
                    tags.get("waterway")
                    or tags.get("natural") in ("water", "coastline")
                    or tags.get("landuse") == "reservoir"
                    or tags.get("place") == "sea"
                ):
                    waters.append({"line": coords})
                elif tags.get("landuse") in (
                    "residential",
                    "commercial",
                    "industrial",
                    "farmland",
                    "farm",
                    "orchard",
                    "vineyard",
                    "meadow",
                ):
                    urban_areas.append(coords)

            elif etype == "node" and tags.get("place") in ("city", "town", "village"):
                cities.append((el["lat"], el["lon"]))

        return {
            "roads": roads,
            "waters": waters,
            "cities": cities,
            "urban_areas": urban_areas,
        }

    def get_urban_mask(self, south, north, west, east, zoom=12):
        """قناع المناطق الحضرية/المائية/الزراعية مع كاش."""
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
        road_distance=800,
        city_distance=10000,
        water_distance=700,
    ):
        """
        True لو كانت النقطة قرب:
        - مدينة/بلدة (مسافة أكبر لحمايتها)
        - طريق رئيسي
        - مسطح مائي / بحر / مجرى نهر
        - landuse حضري أو زراعي
        """
        for c_lat, c_lon in urban_mask.get("cities", []):
            if self._haversine_distance(lat, lon, c_lat, c_lon) <= city_distance:
                return True

        for road in urban_mask.get("roads", []):
            for (a_lat, a_lon), (b_lat, b_lon) in zip(road[:-1], road[1:]):
                d = self._point_to_segment_distance(lat, lon, a_lat, a_lon, b_lat, b_lon)
                if d <= road_distance:
                    return True

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

        for urban_poly in urban_mask.get("urban_areas", []):
            if len(urban_poly) >= 3 and point_in_poly(lat, lon, urban_poly):
                return True

        return False


urban_detector = UrbanAreaDetector()


# ===================== بلاطات صور الأقمار الصناعية (ESRI Imagery) =====================
# (موجودة كخيار إضافي فقط، ولن تُستخدم افتراضيًا لتخفيف التحليل)


ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def latlon_to_tilexy(lat, lon, z):
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = (lon + 180.0) / 360.0 * (1 << z)
    y = (
        1.0
        - math.log(
            math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))
        )
        / math.pi
    ) / 2.0 * (1 << z)
    return int(x), int(y)


def num2deg(x, y, z):
    n = 2.0**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def bbox_to_tiles(n, s, e, w, z):
    tx_min, ty_min = latlon_to_tilexy(n, w, z)
    tx_max, ty_max = latlon_to_tilexy(s, e, z)
    tx0, tx1 = min(tx_min, tx_max), max(tx_min, tx_max)
    ty0, ty1 = min(ty_min, ty_max), max(ty_min, ty_max)
    return [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]


def fetch_tile(tx, ty, z, timeout=15):
    url = ESRI_URL.format(z=z, x=tx, y=ty)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    return img


def fetch_tiles_bbox(n, s, e, w, z=15, max_tiles=32):
    tiles_xy = bbox_to_tiles(n, s, e, w, z)
    if len(tiles_xy) > max_tiles:
        z = max(8, z - 1)
        tiles_xy = bbox_to_tiles(n, s, e, w, z)
    out = []
    for tx, ty in tiles_xy:
        try:
            img = fetch_tile(tx, ty, z)
            out.append({"tx": tx, "ty": ty, "z": z, "image": img})
        except requests.RequestException:
            continue
    return out


def _summarize_single_tile(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return {
            "dark_fraction": 0.0,
            "light_fraction": 0.0,
            "edge_fraction": 0.0,
            "oxidation_fraction": 0.0,
            "pits_score": 0.0,
            "tree_like_score": 0.0,
            "dark_fan_score": 0.0,
            "overall_hotspot": 0.0,
        }

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    dark_mask = v < 80
    light_mask = v > 180
    dark_fraction = float(dark_mask.mean())
    light_fraction = float(light_mask.mean())

    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]

    yellow_mask = (h_ch >= 20) & (h_ch <= 35) & (s_ch > 80) & (v > 80)
    orange_mask = (h_ch >= 10) & (h_ch < 20) & (s_ch > 80) & (v > 80)
    red_mask1 = (h_ch <= 5) & (s_ch > 80) & (v > 60)
    red_mask2 = (h_ch >= 170) & (s_ch > 80) & (v > 60)
    oxidation_mask = yellow_mask | orange_mask | red_mask1 | red_mask2
    oxidation_fraction = float(oxidation_mask.mean())

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_fraction = float((edges > 0).mean())

    pits_score = 0.0
    try:
        dark_binary = dark_mask.astype("uint8") * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_binary, connectivity=8
        )
        small_pits = 0
        for i in range(1, num_labels):
            a = stats[i, cv2.CC_STAT_AREA]
            if 5 < a < 300:
                small_pits += 1
        pits_score = min(1.0, small_pits / 400.0)
    except Exception:
        pits_score = 0.0

    tree_like_score = min(1.0, edge_fraction * 4.0)

    if light_fraction > 0.1:
        dark_fan_score = float(max(0.0, min(1.0, (dark_fraction - 0.2) * 3.0)))
    else:
        dark_fan_score = 0.0

    overall_hotspot = max(
        0.0,
        min(
            1.0,
            0.3 * dark_fan_score
            + 0.25 * tree_like_score
            + 0.25 * min(1.0, oxidation_fraction * 3.0)
            + 0.2 * pits_score,
        ),
    )

    return {
        "dark_fraction": dark_fraction,
        "light_fraction": light_fraction,
        "edge_fraction": edge_fraction,
        "oxidation_fraction": oxidation_fraction,
        "pits_score": pits_score,
        "tree_like_score": tree_like_score,
        "dark_fan_score": dark_fan_score,
        "overall_hotspot": overall_hotspot,
    }


def summarize_tiles_for_bbox(tiles: List[Dict[str, Any]]) -> Dict[str, float]:
    if not tiles:
        return {
            "has_imagery": False,
            "overall_hotspot": 0.0,
            "tree_like_score": 0.0,
            "dark_fan_score": 0.0,
            "pits_score": 0.0,
            "oxidation_fraction": 0.0,
        }

    summaries = [_summarize_single_tile(t["image"]) for t in tiles]

    def avg(key: str) -> float:
        return float(sum(s[key] for s in summaries) / len(summaries))

    return {
        "has_imagery": True,
        "overall_hotspot": avg("overall_hotspot"),
        "tree_like_score": avg("tree_like_score"),
        "dark_fan_score": avg("dark_fan_score"),
        "pits_score": avg("pits_score"),
        "oxidation_fraction": avg("oxidation_fraction"),
    }


def imagery_hints_for_bbox(
    north, south, east, west, z=15, max_tiles=32
) -> Dict[str, Any]:
    try:
        tiles = fetch_tiles_bbox(north, south, east, west, z=z, max_tiles=max_tiles)
        if not tiles:
            return {
                "has_imagery": False,
                "overall_hotspot": 0.0,
                "tree_like_score": 0.0,
                "dark_fan_score": 0.0,
                "pits_score": 0.0,
                "oxidation_fraction": 0.0,
            }
        return summarize_tiles_for_bbox(tiles)
    except Exception as e:
        logger.warning(f"فشل في تحليل صور الأقمار الصناعية: {e}")
        return {
            "has_imagery": False,
            "overall_hotspot": 0.0,
            "tree_like_score": 0.0,
            "dark_fan_score": 0.0,
            "pits_score": 0.0,
            "oxidation_fraction": 0.0,
        }


# ===================== API: تحليل مضلع مرسوم على الخريطة =====================


@csrf_exempt
def predict(request):
    """
    POST /predict
    body: {
      "coordinates": [[lat,lon], ...],
      "patterns": ["tree","leaf","wadis","multi","fractures"],
      "mountainsOnly": true/false,
      "excludeUrban": (سيتم تجاهله، الفلتر مفعّل دائماً),
      "count": 150
    }
    - استبعاد المدن/البحر/المزارع.
    - نقاط أقل وتحليل أخف.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST فقط")

    try:
        start_time = time.time()
        payload = json.loads(request.body.decode("utf-8") or "{}")

        coords = payload.get("coordinates") or []
        patterns = payload.get("patterns") or [
            "tree",
            "leaf",
            "wadis",
            "multi",
            "fractures",
        ]
        mountains_only = bool(payload.get("mountainsOnly", True))
        # فلتر حضري دائماً (لحماية المدن والمزارع والبحر)
        exclude_urban = True
        count = min(300, int(payload.get("count", 180)))

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

        if exclude_urban:
            urban_mask = urban_detector.get_urban_mask(
                bbox["south"], bbox["north"], bbox["west"], bbox["east"], zoom=12
            )
        else:
            urban_mask = {"roads": [], "waters": [], "cities": [], "urban_areas": []}

        analysis_points = random_points_in_poly(coords, count)
        features: List[Dict[str, Any]] = []
        analyzed_points = 0

        for lat, lon in analysis_points:
            analyzed_points += 1

            # منع التحليل فوق البحر
            if is_sea(lat, lon):
                continue

            # منع المدن/المزارع/المياه
            if exclude_urban and urban_detector.is_urban_area(lat, lon, urban_mask):
                continue

            analysis_result = analyzer.analyze_point(lat, lon, patterns)

            if mountains_only and not (
                analysis_result["flags"]["mountain"]
                or analysis_result["flags"]["slope"]
                or analysis_result["flags"]["hillshade"]
                or analysis_result["flags"]["transition_zone"]
            ):
                continue

            if not _accept_geology_point(analysis_result):
                continue

            features.append(create_geojson_feature(lon, lat, analysis_result))

            if len(features) >= 400:
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


# ===================== API: تحليل BBOX شبكي + ارتباط اختياري بالصور =====================


@csrf_exempt
def analyze_bbox_tiles(request):
    """
    POST /analyze-bbox-tiles
    body: {
      "north": .., "south": .., "east": .., "west": ..,
      "zoom": 12,
      "patterns": [...],
      "mountainsOnly": true/false,
      "excludeUrban": (يُتجاهل، الفلتر مفعّل دائماً),
      "useImagery": false,   # افتراضيًا لا يستخدم الصور (أخف)
      "limit": 350
    }
    """
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST لهذه الواجهة")

    try:
        start_time = time.time()
        payload = json.loads(request.body.decode("utf-8") or "{}")

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
        # فلتر حضري دائماً
        exclude_urban = True
        limit = min(500, int(payload.get("limit", 350)))
        use_imagery = bool(payload.get("useImagery", False))

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

        # الصور الفضائية اختيارية الآن (لتخفيف الضغط)
        if use_imagery:
            imagery_hints = imagery_hints_for_bbox(
                north=north, south=south, east=east, west=west, z=zoom, max_tiles=32
            )
        else:
            imagery_hints = {
                "has_imagery": False,
                "overall_hotspot": 0.0,
                "tree_like_score": 0.0,
                "dark_fan_score": 0.0,
                "pits_score": 0.0,
                "oxidation_fraction": 0.0,
            }

        base_density = max(5, min(12, zoom - 9))  # أقل من السابق لتخفيف النقاط
        cols = max(8, int((east - west) * base_density))
        rows = max(6, int((north - south) * base_density))

        step_lat = (north - south) / max(rows, 1)
        step_lon = (east - west) / max(cols, 1)

        features: List[Dict[str, Any]] = []
        analyzed_points = 0

        hotspot_boost = float(imagery_hints.get("overall_hotspot", 0.0))
        tree_like_hint = float(imagery_hints.get("tree_like_score", 0.0))
        dark_fan_hint = float(imagery_hints.get("dark_fan_score", 0.0))
        pits_hint = float(imagery_hints.get("pits_score", 0.0))
        oxidation_hint = float(imagery_hints.get("oxidation_fraction", 0.0))

        lat = south + step_lat / 2.0
        while lat < north and len(features) < limit:
            lon = west + step_lon / 2.0
            while lon < east and len(features) < limit:
                analyzed_points += 1

                # منع البحر
                if is_sea(lat, lon):
                    lon += step_lon
                    continue

                # منع المدن/المزارع/المياه
                if exclude_urban and urban_detector.is_urban_area(lat, lon, urban_mask):
                    lon += step_lon
                    continue

                analysis_result = analyzer.analyze_point(lat, lon, patterns)

                if imagery_hints.get("has_imagery", False):
                    img_factor = 1.0 + 0.30 * hotspot_boost
                    analysis_result["confidence"] = float(
                        max(0.0, min(1.0, analysis_result["confidence"] * img_factor))
                    )

                    if oxidation_hint > 0.18:
                        analysis_result["reasons"].append(
                            "صور الأقمار الصناعية تُظهر ألوان أكسدة داخل هذا المربع."
                        )
                    if pits_hint > 0.12:
                        analysis_result["reasons"].append(
                            "توجد نقاط داكنة صغيرة كثيرة (حُفر/أنشطة تعدين) في الصور."
                        )
                    if dark_fan_hint > 0.25:
                        analysis_result["reasons"].append(
                            "كتل جبلية داكنة مع أودية فاتحة حولها (نمط شبيه لمناجم معروفة)."
                        )
                    if tree_like_hint > 0.30:
                        analysis_result["reasons"].append(
                            "الشبكة التصريفية في الصور تبدو شجرية/ورقية قوية."
                        )

                if mountains_only and not (
                    analysis_result["flags"]["mountain"]
                    or analysis_result["flags"]["slope"]
                    or analysis_result["flags"]["hillshade"]
                    or analysis_result["flags"]["transition_zone"]
                ):
                    lon += step_lon
                    continue

                if not _accept_geology_point(analysis_result):
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
                    "imagery": imagery_hints,
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
    """GET /analyze-place?q=اسم – بحث عن مكان داخل السعودية."""
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
        headers = {"User-Agent": "GoldMap-Pro/1.0 (https://example.com/goldmap)"}

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
    """POST /export-geojson – يرجع ملف GeoJSON للتنزيل."""
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST لهذه الواجهة")

    try:
        geojson_data = json.loads(request.body.decode("utf-8") or "{}")

        if (
            not isinstance(geojson_data, dict)
            or geojson_data.get("type") != "FeatureCollection"
        ):
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


# ===================== API: حفظ النتائج (تجريبي) =====================


@csrf_exempt
def save_hotspots(request):
    """POST /save-hotspots – حفظ تجريبي بدون قاعدة بيانات."""
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST لهذه الواجهة")

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
        features = (payload.get("data") or {}).get("features", [])

        return JsonResponse(
            {
                "success": True,
                "region_id": random.randint(1000, 9999),
                "saved_points": len(features),
                "message": "تم حفظ النتائج بنجاح (نسخة تجريبية بدون قاعدة بيانات).",
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


# ===================== Health / Clear Cache =====================


def health_check(request):
    """GET /health – فحص صحة الخدمة."""
    return JsonResponse(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.1",
            "services": {
                "geology_analyzer": "active",
                "urban_detector": "active",
                "cache": "active",
            },
        }
    )


@csrf_exempt
def clear_cache(request):
    """POST /clear-cache – مسح كاش UrbanAreaDetector."""
    if request.method != "POST":
        return HttpResponseBadRequest("يجب استخدام طريقة POST لهذه الواجهة")

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
