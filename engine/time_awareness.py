"""Friday Time Awareness — Ackem-compatible temporal perception.

Features:
  - Holiday detection (Gregorian fixed + Lunar precomputed + floating)
  - Time depth calculation (natural language: "we've known each other for X")
  - Special date detection (birthday, anniversary, milestone, holiday)
  - Reunion boost (affection bonus after long offline)

References:
  Ackem src/main/engine/temporalAwareness/ — 100% algorithm parity.
"""

from datetime import datetime, date
from typing import Dict, Optional, List

# ═══════════════════════════════════════════════════════════════════
# ① Holiday Detection — 3-strategy, pure function, <0.2ms
# ═══════════════════════════════════════════════════════════════════

STATIC_HOLIDAYS: Dict[str, str] = {
    "01-01": "元旦",
    "02-14": "情人节",
    "03-08": "国际妇女节",
    "04-01": "愚人节",
    "05-01": "劳动节",
    "05-20": "520",
    "05-21": "521",
    "06-01": "儿童节",
    "10-01": "国庆节",
    "11-11": "光棍节",
    "12-24": "平安夜",
    "12-25": "圣诞节",
    "12-31": "跨年夜",
}

LUNAR_HOLIDAYS: Dict[str, Dict[str, str]] = {
    "2026": {"02-17": "春节", "02-22": "元宵节", "05-31": "端午节", "08-19": "七夕", "10-04": "中秋节"},
    "2027": {"02-06": "春节", "02-11": "元宵节", "05-20": "端午节", "08-08": "七夕", "09-23": "中秋节"},
    "2028": {"01-26": "春节", "01-31": "元宵节", "05-08": "端午节", "07-28": "七夕", "09-11": "中秋节"},
    "2029": {"02-13": "春节", "02-18": "元宵节", "05-28": "端午节", "08-17": "七夕", "10-01": "中秋节"},
    "2030": {"02-03": "春节", "02-08": "元宵节", "05-17": "端午节", "08-06": "七夕", "09-19": "中秋节"},
}

HOLIDAY_CATEGORY: Dict[str, str] = {
    "元旦": "traditional", "国庆节": "traditional",
    "情人节": "western", "圣诞节": "western", "平安夜": "western", "跨年夜": "western",
    "520": "social", "521": "social", "光棍节": "social",
    "母亲节": "family", "父亲节": "family", "儿童节": "family", "国际妇女节": "family",
    "春节": "traditional", "元宵节": "traditional", "端午节": "traditional",
    "七夕": "traditional", "中秋节": "traditional",
}


def _nth_sunday_of_month(year: int, month: int, n: int) -> int:
    """Return the day-of-month of the nth Sunday of a given month."""
    first = date(year, month, 1)
    day_of_week = first.weekday()  # 0=Mon, 6=Sun
    return 1 + 7 * (n - 1) + ((6 - day_of_week) % 7)


def _get_floating_holiday(year: int, month: int, day: int) -> Optional[str]:
    """Detect floating Gregorian holidays (Mother's Day, Father's Day)."""
    if month == 5 and day == _nth_sunday_of_month(year, 5, 2):
        return "母亲节"
    if month == 6 and day == _nth_sunday_of_month(year, 6, 3):
        return "父亲节"
    return None


def detect_holiday(today: date) -> Optional[Dict]:
    """Detect if today is a holiday. Returns {key, category} or None."""
    mmdd = today.strftime("%m-%d")
    year_str = str(today.year)

    # ① Static Gregorian
    if mmdd in STATIC_HOLIDAYS:
        key = STATIC_HOLIDAYS[mmdd]
        return {"key": key, "category": HOLIDAY_CATEGORY.get(key, "traditional")}

    # ② Floating Gregorian
    floating = _get_floating_holiday(today.year, today.month, today.day)
    if floating:
        return {"key": floating, "category": "family"}

    # ③ Lunar precomputed
    lunar_year = LUNAR_HOLIDAYS.get(year_str, {})
    if mmdd in lunar_year:
        key = lunar_year[mmdd]
        return {"key": key, "category": "traditional"}

    return None


# ═══════════════════════════════════════════════════════════════════
# ② Time Depth Calculator
# ═══════════════════════════════════════════════════════════════════

LABELS = {
    "just_met": "刚认识不久",
    "over_month": "认识一个多月了",
    "half_year": "认识快半年了",
    "over_half_year": "认识大半年了",
    "exact_year": "整整 {} 年了",
    "just_over_year": "刚过一周年",
    "just_over_years": "刚满 {} 周年",
    "almost_next_year": "快到 {} 周年了",
    "over_year": "认识超过一年了",
    "over_years": "认识 {} 年多了",
}


def compute_time_depth(first_met_date_str: Optional[str], today: date) -> Optional[Dict]:
    """Calculate the natural-language description of how long since first_met.

    Returns {label, label_key, emotional_weight, is_exact_year, is_milestone,
             years_since, days_since} or None.
    """
    if not first_met_date_str:
        return None

    try:
        first_met = date.fromisoformat(first_met_date_str)
    except (ValueError, TypeError):
        return None

    days_since = (today - first_met).days
    if days_since < 0:
        return None

    diff_years = days_since / 365.2425
    years_since = int(diff_years)
    nearest_year = round(diff_years)
    is_milestone = nearest_year in (1, 2, 3, 5, 10) and nearest_year >= 1
    days_since_last_anniv = days_since - years_since * 365.2425
    distance_to_nearest = days_since - nearest_year * 365.2425

    # Exact anniversary (±15 days)
    if nearest_year >= 1 and abs(distance_to_nearest) <= 15:
        if nearest_year == 1:
            label = "整整一年了"
            label_key = "exact_year"
        else:
            label = f"整整 {nearest_year} 年了"
            label_key = "exact_year"
        emotional_weight = min(0.95, 0.8 + nearest_year * 0.05)
        return {
            "label": label, "label_key": label_key,
            "emotional_weight": emotional_weight, "is_exact_year": True,
            "is_milestone": is_milestone, "years_since": nearest_year,
            "days_since": days_since,
        }

    # Time bands
    if days_since < 30:
        label, label_key, ew = "刚认识不久", "just_met", 0.3
    elif days_since < 90:
        label, label_key, ew = "认识一个多月了", "over_month", 0.4
    elif days_since < 180:
        label, label_key, ew = "认识快半年了", "half_year", 0.5
    elif days_since < 365:
        label, label_key, ew = "认识大半年了", "over_half_year", 0.6
    elif days_since_last_anniv <= 90:
        if years_since == 1:
            label, label_key, ew = "刚过一周年", "just_over_year", 0.78
        else:
            label, label_key, ew = f"刚满 {years_since} 周年", "just_over_years", min(0.95, 0.75 + years_since * 0.03)
    elif days_since_last_anniv > 275:
        label, label_key, ew = f"快到 {years_since + 1} 周年了", "almost_next_year", min(0.95, 0.78 + years_since * 0.04)
    else:
        if years_since <= 1:
            label, label_key, ew = "认识超过一年了", "over_year", 0.73
        else:
            label, label_key, ew = f"认识 {years_since} 年多了", "over_years", min(0.9, 0.7 + years_since * 0.04)

    return {
        "label": label, "label_key": label_key,
        "emotional_weight": ew, "is_exact_year": False,
        "is_milestone": is_milestone, "years_since": years_since,
        "days_since": days_since,
    }


# ═══════════════════════════════════════════════════════════════════
# ③ Special Date Detector — 5-source aggregation
# ═══════════════════════════════════════════════════════════════════

def detect_special_dates(
    today: date,
    first_met_date: Optional[str] = None,
    ackem_birthday: Optional[str] = None,
    birthdays: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Aggregate special dates from 5 sources.

    Sources:
      0: Ackem/Friday's own birthday
      1: First-met anniversary
      2: User birthday
      3: Temporal anchors (from state.db — called separately)
      4: Holiday

    Returns sorted list of {type, title, emotional_intensity, ...}
    """
    today_mmdd = today.strftime("%m-%d")
    results = []

    # Source 0: Friday's own birthday
    if ackem_birthday:
        ackem_mmdd = ackem_birthday[5:10]  # "MM-DD"
        if ackem_mmdd == today_mmdd:
            td = compute_time_depth(ackem_birthday, today)
            years = td["years_since"] if td else 0
            results.append({
                "type": "ackem_birthday",
                "title": "Friday 的生日！" if years == 1 else f"Friday 的 {years} 岁生日！",
                "emotional_intensity": min(1.0, 0.7 + years * 0.05),
            })

    # Source 1: First-met anniversary
    if first_met_date:
        td = compute_time_depth(first_met_date, today)
        if td and td.get("is_exact_year"):
            years = td["years_since"]
            results.append({
                "type": "first_met_anniversary",
                "title": "认识一周年！" if years == 1 else f"认识 {years} 周年！",
                "emotional_intensity": min(0.95, 0.6 + years * 0.1),
                "time_depth": td,
            })

    # Source 2: Birthdays
    if birthdays:
        seen = set()
        for b in birthdays:
            if b.get("birthday_mmdd") != today_mmdd:
                continue
            key = f"{b.get('subject', '')}_{b.get('birthday_mmdd', '')}"
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "type": "birthday",
                "title": f"{b.get('subject', '用户')} 的生日！",
                "emotional_intensity": 1.0,
            })

    # Source 4: Holiday
    holiday = detect_holiday(today)
    if holiday:
        results.append({
            "type": "holiday",
            "title": f"{holiday['key']}快乐！",
            "emotional_intensity": 0.9 if holiday["category"] == "traditional" else 0.7,
        })

    # Sort: anniversary > birthday > holiday
    type_order = {"ackem_birthday": 0, "first_met_anniversary": 0, "birthday": 1, "holiday": 2}
    results.sort(key=lambda r: (type_order.get(r["type"], 3), -r.get("emotional_intensity", 0)))
    return results


# ═══════════════════════════════════════════════════════════════════
# ④ Reunion Boost — affection bonus after long offline
# ═══════════════════════════════════════════════════════════════════

REUNION_OFFLINE_MINUTES = 30
REUNION_OFFLINE_CAP_MINUTES = 1440  # 24 hours cap
REUNION_AFF_BOOST = 5.0
REUNION_SEC_BOOST = 3.0


def compute_reunion_boost(
    last_active_at: Optional[str],
    now: datetime,
) -> Optional[Dict]:
    """Calculate affection/security boost after long offline period.

    Returns {aff_boost, sec_boost, minutes_offline, should_brief} or None.
    """
    if not last_active_at:
        return None

    try:
        last_dt = datetime.fromisoformat(last_active_at)
    except (ValueError, TypeError):
        # Try epoch timestamp
        try:
            last_ts = float(last_active_at)
            last_dt = datetime.fromtimestamp(last_ts)
        except (ValueError, TypeError):
            return None

    minutes_offline = (now - last_dt).total_seconds() / 60
    if minutes_offline < REUNION_OFFLINE_MINUTES:
        return None

    factor = min(minutes_offline / REUNION_OFFLINE_CAP_MINUTES, 1.0) * 0.5 + 0.5
    return {
        "aff_boost": REUNION_AFF_BOOST * factor,
        "sec_boost": REUNION_SEC_BOOST * factor,
        "minutes_offline": minutes_offline,
        "should_brief": minutes_offline >= 720,  # 12h+ = brief the user
    }


# ═══════════════════════════════════════════════════════════════════
# ⑤ Time-of-Day / Season Detection
# ═══════════════════════════════════════════════════════════════════

def get_time_of_day(hour: int) -> str:
    """Classify hour into time-of-day segment."""
    if 5 <= hour < 9:
        return "morning"
    elif 9 <= hour < 12:
        return "forenoon"
    elif 12 <= hour < 14:
        return "noon"
    elif 14 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 22:
        return "evening"
    elif 22 <= hour or hour < 2:
        return "night"
    else:
        return "deep_night"


def get_season(month: int) -> str:
    """Classify month into season."""
    if 3 <= month <= 5:
        return "spring"
    elif 6 <= month <= 8:
        return "summer"
    elif 9 <= month <= 11:
        return "autumn"
    else:
        return "winter"


def is_weekend(d: date) -> bool:
    """Check if date is weekend."""
    return d.weekday() >= 5  # 5=Sat, 6=Sun
