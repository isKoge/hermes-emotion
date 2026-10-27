"""Friday Weather Sense — Ackem-style dispatched weather detection.

Detects location + time context in user messages and automatically fetches
weather data for injection into the system prompt.

Architecture:
  _detect_weather_intent(msg) → keyword + regex check (<1ms)
  → _fetch_weather(lat, lon, days_offset) → Open-Meteo API (free, no key)
  → inject into prompt context block

Matches Ackem's `tool/weather-sense/` dispatched mode behavior.
"""

import re
import json
import logging
from typing import Optional, Dict, List, Tuple
from urllib.request import Request, urlopen
from datetime import datetime, timedelta, date

logger = logging.getLogger(__name__)

# ── City → Coordinates ──────────────────────────────────────────────
# Common cities + fuzzy matching
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "广州": (23.13, 113.26),
    "深圳": (22.54, 114.06),
    "北京": (39.90, 116.41),
    "上海": (31.23, 121.47),
    "杭州": (30.27, 120.15),
    "成都": (30.57, 104.07),
    "重庆": (29.56, 106.55),
    "武汉": (30.59, 114.31),
    "南京": (32.06, 118.80),
    "西安": (34.26, 108.94),
    "长沙": (28.23, 112.94),
    "厦门": (24.48, 118.09),
    "珠海": (22.27, 113.58),
    "东莞": (23.02, 113.75),
    "佛山": (23.02, 113.12),
    "中山": (22.52, 113.38),
    "惠州": (23.11, 114.42),
    "汕头": (23.37, 116.68),
    "湛江": (21.27, 110.36),
    "三亚": (18.25, 109.51),
    "海口": (20.02, 110.35),
    "桂林": (25.27, 110.29),
    "昆明": (25.04, 102.68),
    "贵阳": (26.65, 106.63),
    "天津": (39.13, 117.20),
    "苏州": (31.30, 120.62),
    "无锡": (31.49, 120.31),
    "宁波": (29.87, 121.55),
    "青岛": (36.07, 120.38),
    "大连": (38.91, 121.61),
    "沈阳": (41.80, 123.43),
    "哈尔滨": (45.80, 126.53),
    "郑州": (34.75, 113.63),
    "济南": (36.67, 116.98),
    "合肥": (31.82, 117.23),
    "南昌": (28.68, 115.86),
    "福州": (26.07, 119.30),
    "南宁": (22.82, 108.37),
    "拉萨": (29.65, 91.14),
    "乌鲁木齐": (43.83, 87.62),
    "呼和浩特": (40.84, 111.75),
    "银川": (38.47, 106.27),
    "兰州": (36.06, 103.83),
    "西宁": (36.62, 101.78),
    "石家庄": (38.04, 114.51),
    "太原": (37.87, 112.55),
    "长春": (43.88, 125.32),
    "香港": (22.28, 114.17),
    "澳门": (22.20, 113.55),
    "台北": (25.03, 121.57),
    "东京": (35.68, 139.76),
    "首尔": (37.57, 126.98),
    "新加坡": (1.35, 103.82),
    "曼谷": (13.75, 100.50),
    "纽约": (40.71, -74.01),
    "London": (51.51, -0.13),
    "巴黎": (48.86, 2.35),
}

# ── Detection patterns ───────────────────────────────────────────────
# Time keywords
TIME_PATTERNS = [
    (re.compile(r"明天|明日"), 1),
    (re.compile(r"后天|後天"), 2),
    (re.compile(r"大后天"), 3),
    (re.compile(r"下周[一二三四五六日]"), None),  # need to calculate
    (re.compile(r"这周末|这个周末|本周末"), None),
    (re.compile(r"下周|下星期"), 7),
    (re.compile(r"今天|今日"), 0),
]

# Travel/planning intent
TRAVEL_PATTERNS = [
    re.compile(r"(去|到|去一趟|出发去|飞|出差去|旅行去)"),
    re.compile(r"(穿什么|带什么衣服|什么天气|冷不冷|热不热|下雨吗|带伞)"),
]

# Cache
_weather_cache: Dict[str, Tuple[float, str]] = {}
CACHE_TTL = 1800  # 30 minutes


# ── Detection ─────────────────────────────────────────────────────────

def _find_city(text: str) -> Optional[Tuple[str, Tuple[float, float]]]:
    """Find city mention in text, return (name, (lat, lon)) or None."""
    for city, coords in CITY_COORDS.items():
        if city in text:
            return (city, coords)
    return None


def _find_days_offset(text: str) -> Optional[int]:
    """Detect time offset in days from today."""
    for pattern, offset in TIME_PATTERNS:
        if pattern.search(text):
            if offset is not None:
                return offset
            # For complex patterns like "下周三", skip for now
            return 3  # default to 3 days if unclear
    return None


def _has_travel_intent(text: str) -> bool:
    """Check if the message suggests travel/planning."""
    for pat in TRAVEL_PATTERNS:
        if pat.search(text):
            return True
    return False


# ── Weather fetch (Open-Meteo, free, no API key required) ───────────

def _fetch_weather(lat: float, lon: float, days_offset: int = 0) -> Optional[str]:
    """Fetch weather from Open-Meteo API.

    Returns a human-readable weather string, or None.
    """
    target_date = date.today() + timedelta(days=days_offset)
    date_str = target_date.strftime("%Y-%m-%d")

    cache_key = f"{lat:.2f}_{lon:.2f}_{date_str}"
    now_ts = datetime.now().timestamp()
    if cache_key in _weather_cache:
        cached_ts, cached_result = _weather_cache[cache_key]
        if now_ts - cached_ts < CACHE_TTL:
            return cached_result

    result = _fetch_openmeteo(lat, lon, days_offset)

    if result:
        _weather_cache[cache_key] = (now_ts, result)
    return result


def _fetch_openmeteo(lat: float, lon: float, days_offset: int = 0) -> Optional[str]:
    """Fetch from Open-Meteo API (free, no key required)."""
    target_date = date.today() + timedelta(days=days_offset)
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_probability_max"
            f"&timezone=Asia/Shanghai"
            f"&start_date={date_str}&end_date={date_str}"
        )
        req = Request(url, headers={"User-Agent": "HermesEmotion/1.0"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        daily = data.get("daily", {})
        if not daily:
            return None
        temp_max = daily.get("temperature_2m_max", [None])[0]
        temp_min = daily.get("temperature_2m_min", [None])[0]
        wc = daily.get("weathercode", [None])[0]
        if temp_max is None:
            return None
        desc = _openmeteo_code(wc)
        result = f"{temp_max:.0f}°C / {temp_min:.0f}°C · {desc}"
        return result
    except Exception as e:
        logger.debug("[weather] open-meteo failed: %s", e)
        return None


def _openmeteo_code(code: int) -> str:
    if code is None: return "未知"
    if code <= 1: return "晴"
    if code <= 3: return "多云"
    if code <= 48: return "雾"
    if code <= 57: return "毛毛雨"
    if code <= 67: return "雨"
    if code <= 77: return "雪"
    if code <= 82: return "阵雨"
    if code <= 86: return "阵雪"
    if code <= 99: return "雷暴"
    return "未知"


# ── Main API ──────────────────────────────────────────────────────────

def detect_and_fetch(text: str) -> Optional[str]:
    """Detect weather intent and fetch weather data.

    Returns weather string for injection, or None.
    """
    if not text or len(text) < 3:
        return None

    city_result = _find_city(text)
    if not city_result:
        return None

    # Must have BOTH city AND time intent (or strong travel intent)
    days = _find_days_offset(text)
    has_travel = _has_travel_intent(text)

    if days is None and not has_travel:
        return None

    if days is None:
        days = 0  # default to today

    city_name, (lat, lon) = city_result
    weather = _fetch_weather(lat, lon, days)
    if not weather:
        return None

    day_label = {0: "今天", 1: "明天", 2: "后天"}.get(days, f"{days}天后")
    return f"{city_name} {day_label} {weather}"
