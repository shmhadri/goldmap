# vision_tile.py
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
import cv2

# =================== أوزان الأنماط (متوافقة مع الواجهة) ===================

@dataclass
class Weights:
    tree: float = 0.40      # زيادة الوزن للنمط الشجري
    leaf: float = 0.25      # زيادة الوزن للنمط الورقي
    wadis: float = 0.15     # تشعّبات أودية في جبال داكنة
    multi: float = 0.12     # تعدد لوني (أصفر/أحمر/برتقالي)
    fractures: float = 0.08

# =================== أدوات عامة ===================

def _img_diag(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    return float(np.hypot(h, w))

def _auto_canny_thresh(gray: np.ndarray, low: float, high: float) -> Tuple[int, int]:
    """
    low/high هي نسب من الحد الذاتي (median) وتُحوّل إلى عتبات صحيحة.
    """
    med = float(np.median(gray))
    t1 = int(max(0, (1.0 - low) * med))
    t2 = int(min(255, (1.0 + high) * med))
    if t1 >= t2:
        t1, t2 = max(0, t2 - 40), min(255, t2 + 40)
    return t1, t2

# =================== أقنعة لونية مع تحسينات ===================

def mask_yellow(hsv: np.ndarray) -> np.ndarray:
    # أصفر قوي (S,V مرتفعة) - تحسين النطاق للذهب
    return cv2.inRange(hsv, (18, 85, 130), (42, 255, 255))

def mask_red(hsv: np.ndarray) -> np.ndarray:
    # أحمر مرتبط بأكسدة الذهب
    a = cv2.inRange(hsv, (0, 75, 80), (12, 255, 255))
    b = cv2.inRange(hsv, (165, 75, 80), (180, 255, 255))
    return cv2.bitwise_or(a, b)

def mask_orange(hsv: np.ndarray) -> np.ndarray:
    # برتقالي مرتبط بالمعادن
    return cv2.inRange(hsv, (8, 95, 110), (22, 255, 255))

def mask_dark(hsv: np.ndarray) -> np.ndarray:
    """
    جبال داكنة: قيمة V منخفضة + تشبّع كافٍ
    نحاول استبعاد الطرق الإسفلتية الرمادية قدر الإمكان.
    """
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    dark_v = cv2.inRange(v, 0, 85)   # تحسين النطاق
    sat_ok = cv2.inRange(s, 45, 255) # استبعاد السطوح الباهتة
    return cv2.bitwise_and(dark_v, sat_ok)

def mask_water(bgr: np.ndarray) -> np.ndarray:
    """
    كشف المسطحات المائية (بحر/بحيرات) حتى لا تُحلّل ولا تُرسم عليها نقاط.
    نعتمد على نطاقات الأزرق/السماوي مع توسعة بسيطة.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # أزرق عميق + سماوي قرب السواحل
    deep_blue = cv2.inRange(hsv, (90, 40, 20), (140, 255, 220))
    cyan = cv2.inRange(hsv, (75, 25, 40), (100, 255, 255))
    water = cv2.bitwise_or(deep_blue, cyan)

    # توسعة بسيطة للمناطق المائية لتغطية الحواف
    kernel = np.ones((7, 7), np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel)
    water = cv2.dilate(water, kernel, iterations=1)

    return water

def mask_urban_areas(bgr: np.ndarray) -> np.ndarray:
    """
    كشف المناطق المأهولة (طرق، مباني، مدن).
    نحاول أن نكون عدوانيين قليلاً حتى لا نضع نقاط فوق المدن/الطرق.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # 1. كشف الأسطح الملساء (طرق إسفلت/أسطح كبيرة)
    smooth_areas = cv2.GaussianBlur(gray, (15, 15), 0)
    smooth_mask = cv2.inRange(smooth_areas, 85, 235)

    # 2. كشف الألوان الصناعية (مباني/أسطح فاتحة متجانسة)
    # قيم S منخفضة + V أعلى من حد معيّن
    urban_colors = cv2.inRange(hsv, (0, 0, 140), (180, 70, 255))
    
    # 3. كشف الخطوط المستقيمة (طرق)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=45,
                            minLineLength=40, maxLineGap=10)
    
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
    
    # الجمع بين جميع المؤشرات
    urban_mask = cv2.bitwise_or(smooth_mask, urban_colors)
    urban_mask = cv2.bitwise_or(urban_mask, line_mask)

    # توسعة المناطق المكتشفة أكثر لمنع النقاط القريبة من العمران
    kernel = np.ones((11, 11), np.uint8)
    urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_CLOSE, kernel)
    urban_mask = cv2.dilate(urban_mask, kernel, iterations=1)
    
    return urban_mask

# =================== أدوات شبكية (Skeleton/Branch) ===================

def skeletonize(bin_img: np.ndarray, max_iter: int = 256) -> np.ndarray:
    img = (bin_img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    it = 0
    while True:
        it += 1
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0 or it >= max_iter:
            break
    return skel

def branch_points(skel: np.ndarray) -> int:
    sk = (skel > 0).astype(np.uint8)
    k = np.array([[1, 1, 1],
                  [1,10, 1],
                  [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(sk, -1, k)
    # نطاق يهزّ الأفرع ويقلل الضجيج
    return int(np.sum((conv >= 13) & (conv <= 18)))

# =================== مقاييس الأنماط ===================

def score_tree(gray: np.ndarray) -> float:
    """
    تحسين النمط الشجري - يبحث عن أنماط تشعبية تشبه الجذور أو الأغصان
    """
    t1, t2 = _auto_canny_thresh(gray, 0.30, 0.35)  # تحسين العتبات
    edges = cv2.Canny(gray, t1, t2)
    
    # تنظيف الحواف
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    
    skel = skeletonize(edges)
    nodes = branch_points(skel)
    length = float(np.count_nonzero(skel))
    
    if length <= 10.0:  # زيادة الحد الأدنى
        return 0.0
    
    # حساب الكثافة التشعبية
    branch_density = nodes / (length / 800.0 + 1e-6)
    
    # تحسين النتيجة النهائية
    score = float(np.clip(branch_density, 0.0, 1.0))
    
    # تعزيز النتائج الجيدة
    if score > 0.3:
        score = min(1.0, score * 1.2)
    
    return score

def score_leaf(gray: np.ndarray) -> float:
    """
    تحسين النمط الورقي - يبحث عن تشعبات دقيقة متعددة الاتجاهات
    """
    t1, t2 = _auto_canny_thresh(gray, 0.25, 0.40)  # عتبات أكثر حساسية
    edges = cv2.Canny(gray, t1, t2)
    
    # تنظيف الحواف للأنماط الدقيقة
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70,  # تخفيض العتبة
                            minLineLength=15, maxLineGap=6)      # خطوط أقصر
    
    if lines is None:
        return 0.0
    
    lines = lines[:800]  # زيادة الحد الأقصى للخطوط
    lengths = np.hypot(lines[:,0,2]-lines[:,0,0], lines[:,0,3]-lines[:,0,1])
    
    # كثافة الخطوط (أكثر حساسية)
    density = float(min(1.0, len(lines)/350.0))
    
    # متوسط الأطوال (نفضل الخطوط القصيرة المتعددة)
    avg_length = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
    length_score = 1.0 - min(1.0, avg_length/100.0)  # عكس العلاقة
    
    # اتجاهات الخطوط (التنوع في الاتجاهات)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        angles.append(angle % 180)
    
    angle_diversity = len(set(np.round(angles).astype(int))) / 180.0
    
    # النتيجة المركبة
    score = 0.5*density + 0.3*length_score + 0.2*angle_diversity
    return float(min(1.0, score))

def score_wadis(gray: np.ndarray, hsv: np.ndarray) -> float:
    """
    تحسين اكتشاف الأودية في المناطق الجبلية الداكنة
    """
    dark = mask_dark(hsv)
    
    # التأكد من وجود كتلة داكنة كافية
    dark_ratio = np.count_nonzero(dark) / dark.size
    if dark_ratio < 0.15:  # زيادة الحد الأدنى
        return 0.0
    
    t1, t2 = _auto_canny_thresh(gray, 0.28, 0.38)  # عتبات محسنة
    edges = cv2.Canny(gray, t1, t2)
    edges = cv2.bitwise_and(edges, edges, mask=dark)
    
    # تنظيف الحواف في المناطق الداكنة
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    skel = skeletonize(edges)
    nodes = branch_points(skel)
    length = float(np.count_nonzero(skel))
    
    if length <= 15.0:
        return 0.0
    
    # حساب النتيجة مع مراعاة نسبة المنطقة الداكنة
    base_score = nodes / (length / 650.0 + 1e-6)
    score = float(np.clip(base_score, 0.0, 1.0))
    
    # تعزيز النتيجة في المناطق الداكنة الكبيرة
    if dark_ratio > 0.3:
        score = min(1.0, score * 1.15)
    
    return score

def score_multi(hsv: np.ndarray) -> float:
    """
    تحسين اكتشاف تعدد الألوان (أصفر/أحمر/برتقالي)
    """
    y = mask_yellow(hsv)
    r = mask_red(hsv)
    o = mask_orange(hsv)
    
    # حساب المساحة مع مراعاة التوزيع
    total_area = (np.count_nonzero(y) + np.count_nonzero(r) + np.count_nonzero(o))
    total_pixels = hsv.shape[0] * hsv.shape[1]
    area_ratio = total_area / total_pixels
    
    # تنوع الألوان
    color_variety = 0.0
    if np.count_nonzero(y) > 0: color_variety += 0.4
    if np.count_nonzero(r) > 0: color_variety += 0.35
    if np.count_nonzero(o) > 0: color_variety += 0.25
    
    # النتيجة المركبة
    base_score = area_ratio * 7.0  # زيادة الحساسية قليلاً
    variety_boost = color_variety * 0.3
    
    score = min(1.0, base_score + variety_boost)
    return float(score)

def score_fractures(gray: np.ndarray) -> float:
    """
    تحسين اكتشاف الكسور والصدوع الخطية
    """
    t1, t2 = _auto_canny_thresh(gray, 0.32, 0.42)  # عتبات محسنة
    edges = cv2.Canny(gray, t1, t2)
    
    # تنظيف الحواف للخطوط الطويلة
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=85,  # تخفيض العتبة
                            minLineLength=60, maxLineGap=12)     # تحسين المعاملات
    
    if lines is None:
        return 0.0
    
    lengths = np.hypot(lines[:,0,2]-lines[:,0,0], lines[:,0,3]-lines[:,0,1])
    
    # عدّ القطاعات الطويلة
    long_segments = np.sum(lengths > 75)  # تخفيض الحد الأدنى للطول
    
    # حساب اتجاهات الخطوط للكشف عن الانتظام
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2-y1, x2-x1)
        angles.append(angle)
    
    # تنوع الاتجاهات (الكسور عادة ما تكون متعددة الاتجاهات)
    angle_std = np.std(angles) if len(angles) > 1 else 0.0
    direction_variety = min(1.0, angle_std / (np.pi/2))
    
    # النتيجة المركبة
    segment_score = min(1.0, long_segments / 45.0)
    score = 0.7 * segment_score + 0.3 * direction_variety
    
    return float(score)

# =================== Hotspots لونية مع فلترة بحر/طرق/منازل ===================

def extract_hotspots(bgr: np.ndarray, ignore_urban: bool = True) -> List[Tuple[float, float, float]]:
    """
    يرجّع [(cx, cy, k)] داخل التايل مع فلترة المناطق المأهولة والبحر.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y = mask_yellow(hsv)
    r = mask_red(hsv)
    o = mask_orange(hsv)
    mask = cv2.bitwise_or(cv2.bitwise_or(y, r), o)

    # فلترة الضوضاء + اشتراط الظلام النسبي (جبال داكنة)
    dark = mask_dark(hsv)
    mask = cv2.bitwise_and(mask, dark)
    
    # تنظيف الشوائب
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)

    # فلترة المناطق المأهولة إذا طُلب
    if ignore_urban:
        urban_mask = mask_urban_areas(bgr)
        inv_urban = cv2.bitwise_not(urban_mask)
        mask = cv2.bitwise_and(mask, mask, mask=inv_urban)

    # فلترة البحر/المسطحات المائية نهائياً
    water_mask = mask_water(bgr)
    inv_water = cv2.bitwise_not(water_mask)
    mask = cv2.bitwise_and(mask, mask, mask=inv_water)

    num, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts: List[Tuple[float, float, float]] = []
    
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if 8 <= area <= 150:   # توسيع النطاق قليلاً
            cx, cy = cent[i]
            k = 0.0
            
            # تحسين تحديد الثقة بناءً على اللون والموقع
            cx_i, cy_i = int(round(cx)), int(round(cy))
            if 0 <= cy_i < hsv.shape[0] and 0 <= cx_i < hsv.shape[1]:
                if y[cy_i, cx_i] > 0: 
                    k = 0.90  # زيادة ثقة الأصفر
                elif r[cy_i, cx_i] > 0: 
                    k = 0.65  # زيادة ثقة الأحمر
                elif o[cy_i, cx_i] > 0: 
                    k = 0.45  # زيادة ثقة البرتقالي
            
            # تعزيز الثقة في المناطق الداكنة (جبال)
            if k > 0.0 and dark[cy_i, cx_i] > 0:
                k = min(1.0, k * 1.1)
                
            if k > 0.0:
                pts.append((float(cx), float(cy), float(k)))
    
    return pts

# =================== تصنيف التايل ===================

def classify_tile(
    bgr: np.ndarray,
    patterns: Sequence[str],
    w: Optional[Weights] = None,
    with_hotspots: bool = False,
    ignore_urban: bool = True
) -> Tuple[float, Dict[str, float], Dict[str, bool], List[str], Optional[List[Tuple[float, float, float]]]]:
    """
    يُرجع:
      - total: درجة 0..1
      - scores: {'tree':..,'leaf':..,'wadis':..,'multi':..,'fractures':..}
      - flags:  {'dark_mass':..,'water_area':..,'urban_area':..} + مؤشرات مفيدة
      - reasons: أسباب نصية مختصرة
      - hotspots: [(cx,cy,k)] إن طُلِبَت
    """
    pset = set((patterns or []))
    w = w or Weights()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # فحص البحر/المسطحات المائية أولاً
    water_mask = mask_water(bgr)
    water_ratio = np.count_nonzero(water_mask) / water_mask.size

    # فحص المناطق المأهولة
    urban_mask = mask_urban_areas(bgr)
    urban_ratio = np.count_nonzero(urban_mask) / urban_mask.size
    
    # إذا كانت التايل بحر في الغالب -> تخطي التحليل بالكامل
    if water_ratio > 0.35:
        flags = {
            "water_area": True,
            "water_ratio": water_ratio,
            "urban_area": urban_ratio > 0.4,
            "urban_ratio": urban_ratio,
            "mountain_dark_mass": False,
            "any_edges": False,
        }
        return 0.0, {}, flags, ["منطقة بحرية/مائية - تم تخطي التحليل"], None

    # إذا كانت المنطقة مأهولة بشكل كبير، نتخطى التحليل
    if ignore_urban and urban_ratio > 0.4:
        flags = {
            "water_area": water_ratio > 0.35,
            "water_ratio": water_ratio,
            "urban_area": True,
            "urban_ratio": urban_ratio,
            "mountain_dark_mass": False,
            "any_edges": False,
        }
        return 0.0, {}, flags, ["منطقة مأهولة (مدن/طرق كثيفة) - تم تخطي التحليل"], None

    scores: Dict[str, float] = {}
    if "tree" in pset:       scores["tree"] = score_tree(gray)
    if "leaf" in pset:       scores["leaf"] = score_leaf(gray)
    if "wadis" in pset:      scores["wadis"] = score_wadis(gray, hsv)
    if "multi" in pset:      scores["multi"] = score_multi(hsv)
    if "fractures" in pset:  scores["fractures"] = score_fractures(gray)

    # أعلام مساعدة
    dark = mask_dark(hsv)
    dark_ratio = float(np.count_nonzero(dark)) / float(dark.size)
    
    flags = {
        "mountain_dark_mass": dark_ratio > 0.25,
        "any_edges": any(v > 0.05 for v in scores.values()),
        "urban_area": urban_ratio > 0.4,
        "urban_ratio": urban_ratio,
        "water_area": water_ratio > 0.35,
        "water_ratio": water_ratio,
    }

    # دمج الأوزان
    total = (
        w.tree * scores.get("tree", 0.0) +
        w.leaf * scores.get("leaf", 0.0) +
        w.wadis * scores.get("wadis", 0.0) +
        w.multi * scores.get("multi", 0.0) +
        w.fractures * scores.get("fractures", 0.0)
    )

    # شدّة إضافية: داخل جبال داكنة؟ وزد شرط ≥3 سمات
    traits = 0
    traits += 1 if scores.get("tree", 0.0)      > 0.50 else 0  # تخفيض العتبة
    traits += 1 if scores.get("leaf", 0.0)      > 0.50 else 0
    traits += 1 if scores.get("wadis", 0.0)     > 0.50 else 0
    traits += 1 if scores.get("multi", 0.0)     > 0.55 else 0
    traits += 1 if scores.get("fractures", 0.0) > 0.55 else 0

    if flags["mountain_dark_mass"]:
        total = min(1.0, total * 1.10 + 0.04)  # زيادة التعزيز

    if traits >= 3:
        total = min(1.0, total * 1.15 + 0.08)   # زيادة التعزيز للسمات المتعددة
    elif traits >= 2:
        total = min(1.0, total * 1.05 + 0.03)   # تعزيز متوسط لسمتين
    else:
        total = max(0.0, total * 0.80 - 0.03)   # تخفيض أقل للنتائج المنفردة

    total = float(np.clip(total, 0.0, 1.0))

    # أسباب
    reasons: List[str] = []
    if flags["mountain_dark_mass"]:
        reasons.append("كتلة داكنة واسعة (بيئة جبلية)")
    if scores.get("wadis", 0.0) > 0.50:
        reasons.append("تشعّبات أودية داكنة")
    if scores.get("tree", 0.0) > 0.50:
        reasons.append("ملامح شجرية (جذع/أغصان)")
    if scores.get("leaf", 0.0) > 0.50:
        reasons.append("ملامح ورقية/تفرعات قصيرة")
    if scores.get("multi", 0.0) > 0.55:
        reasons.append("ألوان أكسدة متعدّدة")
    if scores.get("fractures", 0.0) > 0.55:
        reasons.append("شقوق/كسور خطية")
    
    if traits >= 3:
        reasons.append("تعدد السمات الجيولوجية")
    
    if urban_ratio > 0.1:
        reasons.append(f"منطقة تحتوي على معالم عمرانية ({urban_ratio:.1%})")
    if water_ratio > 0.05:
        reasons.append(f"نسبة مسطحات مائية داخل التايل ({water_ratio:.1%})")

    hs = extract_hotspots(bgr, ignore_urban) if with_hotspots else None
    return total, scores, flags, reasons, hs
