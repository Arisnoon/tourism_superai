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
from pydantic import BaseModel

from backend.storyteller import LocationStoryMonitor


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = ROOT_DIR / "frontend" / "demo.html"
ATTRACTION_CSV = ROOT_DIR / "database_demo" / "attraction_clean.csv"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
STORY_AUDIO_DIR = ROOT_DIR / "backend" / "outputs" / "story_audio"
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
    start_province: str
    end_province: str
    max_landmarks: int = 40
    max_distance_to_route_km: float = 10.0


class StoryRequest(BaseModel):
    keyword: str
    latitude: float
    longitude: float
    user_context: str | None = None
    language: str = "th"
    generate_audio: bool = False


def _clean_text(value: str) -> str:
    stripped = re.sub(r"<[^>]+>", " ", value or "")
    stripped = html.unescape(stripped)
    return " ".join(stripped.split()).strip()


def _parse_location(raw: str) -> tuple[float, float] | None:
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

app = FastAPI(title="Agentic AI Roadtrip API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "agentic_api",
        "attraction_count": str(len(ATTRACTIONS)),
        "storyteller_ready": str(STORY_MONITOR is not None),
        "groq_model": os.environ.get("GROQ_MODEL", ""),
    }


@app.get("/api/provinces")
def provinces() -> dict[str, Any]:
    available = sorted({name for name in THAI_PROVINCES if name in PROVINCE_POINTS})
    return {"items": available, "count": len(available)}


@app.post("/api/route")
def build_route(payload: RouteRequest) -> dict[str, Any]:
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


@app.post("/api/story")
def create_story(payload: StoryRequest) -> dict[str, Any]:
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


@app.get("/api/audio/{filename}")
def stream_audio(filename: str) -> FileResponse:
    audio_path = STORY_AUDIO_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path)


@app.get("/")
def demo_frontend() -> FileResponse:
    if not FRONTEND_FILE.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Frontend file not found: {FRONTEND_FILE}"},
        )
    return FileResponse(FRONTEND_FILE)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agentic FastAPI backend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'uvicorn'. Install with: pip install uvicorn") from exc
    uvicorn.run("agentic_api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
