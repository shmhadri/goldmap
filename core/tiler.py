import math, requests
import numpy as np, cv2

def latlon_to_tilexy(lat, lon, z):
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = (lon + 180.0) / 360.0 * (1 << z)
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * (1 << z)
    return int(x), int(y)

def num2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon

def tilexy_bounds(tx, ty, z):
    lat1, lon1 = num2deg(tx, ty, z)
    lat2, lon2 = num2deg(tx+1, ty+1, z)
    north, south = max(lat1, lat2), min(lat1, lat2)
    west, east = min(lon1, lon2), max(lon1, lon2)
    return north, south, east, west

def bbox_to_tiles(n, s, e, w, z):
    tx_min, ty_min = latlon_to_tilexy(n, w, z)
    tx_max, ty_max = latlon_to_tilexy(s, e, z)
    tx0, tx1 = min(tx_min, tx_max), max(tx_min, tx_max)
    ty0, ty1 = min(ty_min, ty_max), max(ty_min, ty_max)
    return [(tx, ty) for ty in range(ty0, ty1+1) for tx in range(tx0, tx1+1)]

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def fetch_tile(tx, ty, z, timeout=15):
    url = ESRI_URL.format(z=z, x=tx, y=ty)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)  # BGR
    return img

def fetch_tiles_bbox(n, s, e, w, z=15, max_tiles=64):
    tiles_xy = bbox_to_tiles(n, s, e, w, z)
    if len(tiles_xy) > max_tiles:
        z = max(8, z-1)
        tiles_xy = bbox_to_tiles(n, s, e, w, z)
    out = []
    for tx, ty in tiles_xy:
        try:
            img = fetch_tile(tx, ty, z)
            out.append({"tx": tx, "ty": ty, "z": z, "image": img})
        except requests.RequestException:
            continue
    return out
