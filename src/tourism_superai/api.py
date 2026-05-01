from __future__ import annotations

import argparse
import csv
import html
import logging
import math
import os
import re
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .storyteller import LocationStoryMonitor


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_FILE = ROOT_DIR / "frontend" / "demo.html"
ATTRACTION_CSV = ROOT_DIR / "src" / "tourism_superai" / "data" / "attraction_clean.csv"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
STORY_AUDIO_DIR = ROOT_DIR / "src" / "tourism_superai" / "outputs" / "story_audio"
logger = logging.getLogger("agentic_api")


THAI_PROVINCES = [
    "กรุงเทพมหานคร",
    "กระบี่",
    "กาญจนบุรี",
    "กาฬสินธุ์",
    "กำแพงเพชร",
    "ขอนแก่น",
    "จันทบุรี",
    "ฉะเชิงเทรา",
    "ชลบุรี",
    "ชัยนาท",
    "ชัยภูมิ",
    "ชุมพร",
    "เชียงราย",
    "เชียงใหม่",
    "ตรัง",
    "ตราด",
    "ตาก",
    "นครนายก",
    "นครปฐม",
    "นครพนม",
    "นครราชสีมา",
    "นครศรีธรรมราช",
    "นครสวรรค์",
    "นนทบุรี",
    "นราธิวาส",
    "น่าน",
    "บึงกาฬ",
    "บุรีรัมย์",
    "ปทุมธานี",
    "ประจวบคีรีขันธ์",
    "ปราจีนบุรี",
    "ปัตตานี",
    "พระนครศรีอยุธยา",
    "พะเยา",
    "พังงา",
    "พัทลุง",
    "พิจิตร",
    "พิษณุโลก",
    "เพชรบุรี",
    "เพชรบูรณ์",
    "แพร่",
    "ภูเก็ต",
    "มหาสารคาม",
    "มุกดาหาร",
    "แม่ฮ่องสอน",
    "ยโสธร",
    "ยะลา",
    "ร้อยเอ็ด",
    "ระนอง",
    "ระยอง",
    "ราชบุรี",
    "ลพบุรี",
    "ลำปาง",
    "ลำพูน",
    "เลย",
    "ศรีสะเกษ",
    "สกลนคร",
    "สงขลา",
    "สตูล",
    "สมุทรปราการ",
    "สมุทรสงคราม",
    "สมุทรสาคร",
    "สระแก้ว",
    "สระบุรี",
    "สิงห์บุรี",
    "สุโขทัย",
    "สุพรรณบุรี",
    "สุราษฎร์ธานี",
    "สุรินทร์",
    "หนองคาย",
    "หนองบัวลำภู",
    "อ่างทอง",
    "อำนาจเจริญ",
    "อุดรธานี",
    "อุตรดิตถ์",
    "อุทัยธานี",
    "อุบลราชธานี",
]


class RouteRequest(BaseModel):
    start_province: str = Field(
        ...,
        examples=["กรุงเทพมหานคร"],
        description="Thai province name where the trip starts.",
    )
    end_province: str = Field(
        ...,
        examples=["เชียงใหม่"],
        description="Thai province name where the trip ends.",
    )
    max_landmarks: int = Field(
        40,
        ge=3,
        le=80,
        description="Maximum number of scenic landmarks to return.",
    )
    max_distance_to_route_km: float = Field(
        10.0,
        gt=0,
        le=100,
        description="Maximum landmark distance from the route in kilometers.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "start_province": "กรุงเทพมหานคร",
                    "end_province": "เชียงใหม่",
                    "max_landmarks": 12,
                    "max_distance_to_route_km": 15.0,
                }
            ]
        }
    }


class StoryRequest(BaseModel):
    keyword: str = Field(
        ...,
        examples=["วัดพระแก้ว"],
        description="Place, landmark, or attraction keyword for story generation.",
    )
    latitude: float = Field(..., examples=[13.7515], description="Current latitude.")
    longitude: float = Field(..., examples=[100.4924], description="Current longitude.")
    user_context: str | None = Field(
        None,
        examples=["family roadtrip from Bangkok"],
        description="Optional travel context to personalize the story.",
    )
    language: str = Field(
        "th",
        examples=["th"],
        description="Narration language code. Thai is the default.",
    )
    generate_audio: bool = Field(
        False,
        description="Set true to generate TTS audio when Cartesia is configured.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "keyword": "วัดพระแก้ว",
                    "latitude": 13.7515,
                    "longitude": 100.4924,
                    "user_context": "family roadtrip from Bangkok",
                    "language": "th",
                    "generate_audio": False,
                }
            ]
        }
    }


OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "Service health and runtime readiness endpoints.",
    },
    {
        "name": "Routes",
        "description": "Build scenic province-to-province routes with nearby attractions.",
    },
    {
        "name": "Stories",
        "description": "Generate location-aware travel narration and optional audio.",
    },
    {
        "name": "Frontend",
        "description": "Serve the browser demo for local testing.",
    },
]


def _clean_text(value: str) -> str:
    """Clean HTML and whitespace from a raw text field.

    Args:
        value (str): The raw text to clean.

    Returns:
        str: The cleaned text.
    """
    stripped = re.sub(r"<[^>]+>", " ", value or "")
    stripped = html.unescape(stripped)
    return " ".join(stripped.split()).strip()


def _parse_location(raw: str) -> tuple[float, float] | None:
    """Parse a comma-separated latitude/longitude string into a coordinate pair.

    Args:
        raw (str): The raw location string.

    Returns:
        tuple[float, float] | None: The parsed coordinates or None if invalid.
    """
    if not raw:
        return None
    parts = [item.strip() for item in raw.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Distance in kilometers.
    """
    radius_km = 6_371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _interpolate_route(start: tuple[float, float], end: tuple[float, float], points: int = 80) -> list[list[float]]:
    """Interpolate a route between two points with subtle curvature.

    Args:
        start (tuple[float, float]): Starting coordinates (lat, lon).
        end (tuple[float, float]): Ending coordinates (lat, lon).
        points (int): Number of points to interpolate. Defaults to 80.

    Returns:
        list[list[float]]: List of [lat, lon] points.
    """
    lat1, lon1 = start
    lat2, lon2 = end
    route: list[list[float]] = []
    for i in range(points + 1):
        t = i / points
        # Add subtle curvature to avoid a completely straight line fallback.
        curve = math.sin(t * math.pi) * 0.35
        route.append(
            [
                lat1 + (lat2 - lat1) * t + curve * 0.08,
                lon1 + (lon2 - lon1) * t + curve * 0.08,
            ]
        )
    return route


def _project_xy_km(lat: float, lon: float, ref_lat: float) -> tuple[float, float]:
    """Project latitude/longitude to approximate x/y coordinates in kilometers.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        ref_lat (float): Reference latitude for projection.

    Returns:
        tuple[float, float]: (x, y) in kilometers.
    """
    x = lon * 111.320 * math.cos(math.radians(ref_lat))
    y = lat * 110.574
    return x, y


def _distance_point_to_segment_km(
    point_lat: float,
    point_lon: float,
    seg_start: list[float],
    seg_end: list[float],
    *,
    ref_lat: float,
) -> float:
    """Calculate the distance from a point to a line segment.

    Args:
        point_lat (float): Point latitude.
        point_lon (float): Point longitude.
        seg_start (list[float]): Segment start [lat, lon].
        seg_end (list[float]): Segment end [lat, lon].
        ref_lat (float): Reference latitude for projection.

    Returns:
        float: Distance in kilometers.
    """
    px, py = _project_xy_km(point_lat, point_lon, ref_lat)
    ax, ay = _project_xy_km(seg_start[0], seg_start[1], ref_lat)
    bx, by = _project_xy_km(seg_end[0], seg_end[1], ref_lat)
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 1e-9:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return math.sqrt(dx * dx + dy * dy)


def _distance_to_route_km(
    lat: float, lon: float, route_points: list[list[float]]
) -> tuple[float, int]:
    """Calculate the distance from a point to the nearest point on the route.

    Args:
        lat (float): Point latitude.
        lon (float): Point longitude.
        route_points (list[list[float]]): List of [lat, lon] route points.

    Returns:
        tuple[float, int]: (distance in km, segment_index)
    """
    nearest = float("inf")
    nearest_segment = 0
    if len(route_points) < 2:
        return nearest, nearest_segment
    ref_lat = sum(point[0] for point in route_points) / len(route_points)
    for idx in range(1, len(route_points)):
        dist = _distance_point_to_segment_km(
            lat,
            lon,
            route_points[idx - 1],
            route_points[idx],
            ref_lat=ref_lat,
        )
        if dist < nearest:
            nearest = dist
            nearest_segment = idx
    return nearest, nearest_segment


def _fetch_osrm_route(start: tuple[float, float], end: tuple[float, float]) -> list[list[float]] | None:
    """Fetch a driving route from OSRM and return waypoints as [lat, lon] pairs.

    Args:
        start (tuple[float, float]): Start coordinates (lat, lon).
        end (tuple[float, float]): End coordinates (lat, lon).

    Returns:
        list[list[float]] | None: Route points as [lat, lon] or None if failed.
    """
    start_lat, start_lon = start
    end_lat, end_lon = end
    query = urllib.parse.urlencode({"overview": "full", "geometries": "geojson"})
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}?{query}"
    )
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AgenticAIRoadtrip/1.0 (route planner)"},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            data = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    try:
        
        payload = json.loads(data)
        routes = payload.get("routes") or []
        if not routes:
            return None
        coords = routes[0].get("geometry", {}).get("coordinates", [])
        points = [[float(coord[1]), float(coord[0])] for coord in coords if isinstance(coord, list) and len(coord) >= 2]
        return points if len(points) >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def _http_get_json(url: str, *, timeout: float = 12.0, headers: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Fetch JSON data from a URL.

    Args:
        url (str): The URL to fetch.
        timeout (float): Request timeout in seconds. Defaults to 12.0.
        headers (dict[str, str] | None): HTTP headers. Defaults to None.

    Returns:
        dict[str, Any] | None: Parsed JSON data or None if failed.
    """
    request = urllib.request.Request(
        url,
        headers=headers
        or {
            "User-Agent": os.environ.get(
                "WIKIMEDIA_USER_AGENT",
                "AgenticAIRoadtrip/1.0 (landmark media fetch)",
            )
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        return None


def _fetch_landmark_media(keyword: str) -> dict[str, str | None]:
    """Fetch media information for a landmark keyword.

    Args:
        keyword (str): The landmark keyword to search for.

    Returns:
        dict[str, str | None]: Dictionary with 'image_url', 'summary', 'source_url'.
    """
    query = keyword.strip()
    if not query:
        return {"image_url": None, "summary": None, "source_url": None}

    searx_base = os.environ.get("SEARXNG_BASE_URL", "").strip()
    if searx_base:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "language": "auto",
                "safesearch": 1,
            }
        )
        searx_url = f"{searx_base.rstrip('/')}/search?{params}"
        payload = _http_get_json(searx_url)
        if payload:
            for item in payload.get("results", []):
                image_url = str(item.get("img_src", "")).strip() or str(item.get("thumbnail", "")).strip()
                if not image_url:
                    continue
                summary = str(item.get("content", "")).strip() or str(item.get("title", "")).strip()
                source_url = str(item.get("url", "")).strip() or None
                return {"image_url": image_url, "summary": summary or None, "source_url": source_url}

    wiki_params = urllib.parse.urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 1,
            "prop": "extracts|pageimages|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "piprop": "original",
            "format": "json",
            "utf8": 1,
        }
    )
    wiki_url = f"https://en.wikipedia.org/w/api.php?{wiki_params}"
    wiki_payload = _http_get_json(
        wiki_url,
        headers={
            "User-Agent": os.environ.get(
                "WIKIMEDIA_USER_AGENT",
                "AgenticAIRoadtrip/1.0 (landmark media fetch)",
            )
        },
    )
    pages = ((wiki_payload or {}).get("query") or {}).get("pages") or {}
    if pages:
        page = next(iter(pages.values()))
        image_url = str(((page.get("original") or {}).get("source")) or "").strip() or None
        summary = str(page.get("extract", "")).strip() or None
        source_url = str(page.get("fullurl", "")).strip() or None
        return {"image_url": image_url, "summary": summary, "source_url": source_url}
    return {"image_url": None, "summary": None, "source_url": None}


def _load_attractions() -> list[dict[str, Any]]:
    """Load attraction records from the CSV dataset and normalize fields.

    Args:
        None

    Returns:
        list[dict[str, Any]]: List of attraction dictionaries.
    """
    if not ATTRACTION_CSV.exists():
        return []
    records: list[dict[str, Any]] = []
    with ATTRACTION_CSV.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            loc = _parse_location(str(row.get("ATT_LOCATION", "")).strip())
            if not loc:
                continue
            lat, lon = loc
            province = str(row.get("PROVINCE_NAME_TH", "")).strip()
            name = _clean_text(str(row.get("ATT_NAME_TH") or row.get("ATT_NAME_EN") or "").strip())
            detail = _clean_text(str(row.get("ATT_DETAIL_TH") or row.get("ATT_DETAIL_EN") or "").strip())
            category = _clean_text(str(row.get("ATT_CATEGORY_LABEL", "")).strip())
            if not province or not name:
                continue
            records.append(
                {
                    "name": name,
                    "province": province,
                    "lat": lat,
                    "lon": lon,
                    "detail": detail[:300],
                    "category": category,
                }
            )
    return records


ATTRACTIONS = _load_attractions()
PROVINCE_POINTS: dict[str, tuple[float, float]] = {}
for row in ATTRACTIONS:
    province = row["province"]
    existing = PROVINCE_POINTS.get(province)
    if existing is None:
        PROVINCE_POINTS[province] = (row["lat"], row["lon"])
    else:
        PROVINCE_POINTS[province] = ((existing[0] + row["lat"]) / 2, (existing[1] + row["lon"]) / 2)

STORY_MONITOR: LocationStoryMonitor | None = None
STORY_INIT_ERROR: str | None = None
try:
    os.environ["GROQ_MODEL"] = DEFAULT_GROQ_MODEL
    STORY_MONITOR = LocationStoryMonitor()
except Exception as exc:  # noqa: BLE001
    STORY_INIT_ERROR = str(exc)

app = FastAPI(
    title="Agentic AI Roadtrip API",
    summary="Scenic route planning and location storytelling for Thailand roadtrips.",
    description=(
        "Tourism SuperAI provides FastAPI endpoints for route building, landmark discovery, "
        "and AI-generated location stories. Use the Swagger UI to try requests directly "
        "from the browser during local development."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=OPENAPI_TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/api/health",
    tags=["System"],
    summary="Check API health",
    description="Returns service status, dataset size, storyteller readiness, and active Groq model.",
)
def health() -> dict[str, str]:
    """Get the health status of the API service.

    Args:
        None

    Returns:
        dict[str, str]: Health status information.
    """
    return {
        "status": "ok",
        "service": "agentic_api",
        "attraction_count": str(len(ATTRACTIONS)),
        "storyteller_ready": str(STORY_MONITOR is not None),
        "groq_model": os.environ.get("GROQ_MODEL", ""),
    }


@app.get(
    "/api/provinces",
    tags=["Routes"],
    summary="List available provinces",
    description="Returns Thai provinces that have attraction coordinates available for route planning.",
)
def provinces() -> dict[str, Any]:
    """Get the list of available Thai provinces with location data.

    Args:
        None

    Returns:
        dict[str, Any]: Dictionary with 'items' and 'count'.
    """
    available = sorted({name for name in THAI_PROVINCES if name in PROVINCE_POINTS})
    return {"items": available, "count": len(available)}


@app.post(
    "/api/route",
    tags=["Routes"],
    summary="Build scenic route",
    description=(
        "Creates a route between two Thai provinces, estimates distance, and returns attractions "
        "near the route. OSRM is used when available, with a local fallback route generator."
    ),
)
def build_route(payload: RouteRequest) -> dict[str, Any]:
    """Build a scenic route between two Thai provinces with optional landmarks.

    Args:
        payload (RouteRequest): Request payload with start/end provinces and options.

    Returns:
        dict[str, Any]: Route data including points, distance, and landmarks.
    """
    start = payload.start_province.strip()
    end = payload.end_province.strip()
    if start == end:
        raise HTTPException(status_code=400, detail="Start and destination provinces must be different.")
    if start not in PROVINCE_POINTS:
        raise HTTPException(status_code=404, detail=f"No location data for start province: {start}")
    if end not in PROVINCE_POINTS:
        raise HTTPException(status_code=404, detail=f"No location data for destination province: {end}")

    osrm_points = _fetch_osrm_route(PROVINCE_POINTS[start], PROVINCE_POINTS[end])
    route_points = osrm_points or _interpolate_route(PROVINCE_POINTS[start], PROVINCE_POINTS[end], points=90)
    total_distance_km = 0.0
    for idx in range(1, len(route_points)):
        total_distance_km += _haversine_km(
            route_points[idx - 1][0],
            route_points[idx - 1][1],
            route_points[idx][0],
            route_points[idx][1],
        )

    scenic_candidates: list[dict[str, Any]] = []
    for row in ATTRACTIONS:
        dist, segment_idx = _distance_to_route_km(row["lat"], row["lon"], route_points)
        if dist <= payload.max_distance_to_route_km:
            scenic_candidates.append(
                {
                    **row,
                    "distance_to_route_km": round(dist, 2),
                    "route_segment_index": int(segment_idx),
                }
            )
    scenic_candidates.sort(key=lambda item: (item["route_segment_index"], item["distance_to_route_km"]))
    province_groups: dict[str, list[dict[str, Any]]] = {}
    for item in scenic_candidates:
        province_groups.setdefault(item["province"], []).append(item)

    # Distribute landmarks across route provinces: first pass 1 each, second pass adds 2nd where available.
    province_order = sorted(
        province_groups.keys(),
        key=lambda p: province_groups[p][0]["route_segment_index"],
    )
    max_landmarks = max(3, min(payload.max_landmarks, 80))
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    for province in province_order:
        candidate = province_groups[province][0]
        key = f"{candidate['name']}::{candidate['province']}"
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(candidate)
        if len(selected) >= max_landmarks:
            break

    if len(selected) < max_landmarks:
        for province in province_order:
            if len(selected) >= max_landmarks:
                break
            candidates = province_groups[province]
            if len(candidates) < 2:
                continue
            candidate = candidates[1]
            key = f"{candidate['name']}::{candidate['province']}"
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(candidate)

    landmarks = sorted(selected, key=lambda item: item["route_segment_index"])

    return {
        "start_province": start,
        "end_province": end,
        "route_points": route_points,
        "distance_km": round(total_distance_km, 2),
        "landmarks": landmarks,
        "routing_mode": "osrm" if osrm_points else "fallback",
    }


@app.post(
    "/api/story",
    tags=["Stories"],
    summary="Generate location story",
    description=(
        "Generates an AI travel story for a location keyword and coordinate. "
        "Requires GROQ_API_KEY. Audio generation also requires CARTESIA_API_KEY."
    ),
)
def create_story(payload: StoryRequest) -> dict[str, Any]:
    """Generate a location-based story and optional audio.

    Args:
        payload (StoryRequest): Request payload with keyword, location, and options.

    Returns:
        dict[str, Any]: Story data including narration and media.
    """
    if STORY_MONITOR is None:
        raise HTTPException(
            status_code=503,
            detail=f"Storyteller is unavailable: {STORY_INIT_ERROR or 'unknown error'}",
        )
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="Missing GROQ_API_KEY for storyteller LLM generation")
    if payload.generate_audio and not os.environ.get("CARTESIA_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="Missing CARTESIA_API_KEY for TTS generation")
    try:
        story = STORY_MONITOR.on_position_update(
            keyword=payload.keyword,
            latitude=payload.latitude,
            longitude=payload.longitude,
            user_context=payload.user_context,
            language=payload.language,
            generate_audio=payload.generate_audio,
        )
    except urllib.error.HTTPError as exc:
        logger.exception(
            "External API HTTP error in /api/story: code=%s url=%s keyword=%s",
            exc.code,
            getattr(exc, "url", "unknown"),
            payload.keyword,
        )
        if exc.code == 403:
            # Keep the trip flow running even when upstream provider denies access.
            media = _fetch_landmark_media(payload.keyword)
            fallback_story = {
                "trigger_keyword": payload.keyword,
                "place_name": payload.keyword,
                "latitude": payload.latitude,
                "longitude": payload.longitude,
                "region_context": payload.user_context or "",
                "headline": f"{payload.keyword} (fallback)",
                "story_hook": "ระบบภายนอกปฏิเสธคำขอชั่วคราว จึงแสดงคำแนะนำแบบสำรอง",
                "narration_script": (
                    f"ตอนนี้ไม่สามารถเรียกบริการภายนอกได้ (HTTP 403) "
                    f"แต่คุณกำลังผ่านจุดสำคัญ: {payload.keyword} โปรดลองอีกครั้งภายหลัง"
                ),
                "fact_highlights": [],
                "sources": [],
                "audio_path": None,
                "audio_error": "HTTP 403 Forbidden from upstream API",
                "audio_url": None,
            }
            return {"triggered": True, "story": fallback_story, "fallback": True, "media": media}
        raise HTTPException(
            status_code=502,
            detail=f"Upstream API failed with HTTP {exc.code}: {getattr(exc, 'reason', 'unknown error')}",
        ) from exc
    except urllib.error.URLError as exc:
        logger.exception("Network error in /api/story for keyword=%s", payload.keyword)
        raise HTTPException(status_code=502, detail=f"Upstream API network error: {exc.reason}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected /api/story failure for keyword=%s", payload.keyword)
        raise HTTPException(status_code=500, detail=f"Story generation failed: {exc}") from exc
    media = _fetch_landmark_media(payload.keyword)
    if story is None:
        return {"triggered": False, "media": media}
    story_payload = story.to_dict()
    audio_url = None
    audio_path = story_payload.get("audio_path")
    if isinstance(audio_path, str) and audio_path:
        audio_file = Path(audio_path)
        if audio_file.exists():
            audio_url = f"/api/audio/{audio_file.name}"
    story_payload["audio_url"] = audio_url
    return {"triggered": True, "story": story_payload, "media": media}


@app.get(
    "/api/audio/{filename}",
    tags=["Stories"],
    summary="Stream generated audio",
    description="Streams a generated story audio file from the local story audio output directory.",
)
def stream_audio(filename: str) -> FileResponse:
    """Stream an audio file for a generated story.

    Args:
        filename (str): The audio filename.

    Returns:
        FileResponse: The audio file response.
    """
    audio_path = STORY_AUDIO_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path)


@app.get(
    "/",
    tags=["Frontend"],
    summary="Open demo frontend",
    description="Serves the local HTML demo that calls the API endpoints.",
    include_in_schema=False,
)
def demo_frontend() -> FileResponse:
    """Serve the demo frontend HTML page.

    Args:
        None

    Returns:
        FileResponse: The HTML file response.
    """
    if not FRONTEND_FILE.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Frontend file not found: {FRONTEND_FILE}"},
        )
    return FileResponse(FRONTEND_FILE)


def _build_parser() -> argparse.ArgumentParser:
    """Create the command line parser for the backend.

    Args:
        None

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run the Agentic FastAPI backend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser


def main() -> None:
    """Load env vars and start the FastAPI backend using uvicorn.

    Args:
        None

    Returns:
        None
    """
    args = _build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'uvicorn'. Install with: pip install uvicorn") from exc
    uvicorn.run("agentic_api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
