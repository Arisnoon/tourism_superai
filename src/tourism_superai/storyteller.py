from __future__ import annotations

"""Storytelling backend for the Agentic AI Roadtrip service.

This module manages place-triggered story generation, audio synthesis via
Cartesia, and retrigger/ cooldown logic for location-based narration.
"""

import argparse
import json
import math
import os
import re
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from trip_design import (
        AgentConfig,
        GroqChatClient,
        KnowledgeRetriever,
        SearchResult,
        _http_request,
        _as_string_list,
        load_env_file,
    )
except ImportError:
    from .trip_design import (
        AgentConfig,
        GroqChatClient,
        KnowledgeRetriever,
        SearchResult,
        _http_request,
        _as_string_list,
        load_env_file,
    )


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"


def _slugify(value: str) -> str:
    """Convert a string to a slug suitable for filenames.

    Args:
        value (str): The string to slugify.

    Returns:
        str: The slugified string.
    """
    ascii_only = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return ascii_only or "location"


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in meters.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Distance in meters.
    """
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass(slots=True)
class StoryResult:
    """Structured result object for a generated location story."""
    trigger_keyword: str
    place_name: str
    latitude: float
    longitude: float
    region_context: str
    headline: str
    story_hook: str
    narration_script: str
    fact_highlights: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    audio_path: str | None = None
    audio_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the story to a dictionary.

        Args:
            None

        Returns:
            dict[str, Any]: The dictionary representation.
        """
        return asdict(self)

    def pretty_print(self) -> str:
        """Generate a pretty-printed string of the story.

        Args:
            None

        Returns:
            str: The formatted story.
        """
        lines = [
            f"Place: {self.place_name}",
            f"Keyword: {self.trigger_keyword}",
            f"Coordinates: {self.latitude}, {self.longitude}",
            f"Headline: {self.headline}",
            f"Hook: {self.story_hook}",
            "",
            "Narration:",
            self.narration_script,
        ]
        if self.fact_highlights:
            lines.extend(["", "Highlights:"])
            lines.extend(f"- {item}" for item in self.fact_highlights)
        if self.sources:
            lines.extend(["", "Sources:"])
            lines.extend(f"- {item}" for item in self.sources)
        if self.audio_path:
            lines.extend(["", f"Audio file: {self.audio_path}"])
        if self.audio_error:
            lines.extend(["", f"Audio error: {self.audio_error}"])
        return "\n".join(lines)


@dataclass(slots=True)
class StorytellerConfig:
    """Configuration values and defaults for storyteller and TTS services."""
    base: AgentConfig
    cartesia_api_key: str
    cartesia_voice_id: str
    cartesia_model_id: str
    cartesia_version: str
    output_dir: Path
    min_retrigger_distance_m: float = 500.0
    cooldown_seconds: float = 900.0

    @classmethod
    def from_env(cls) -> "StorytellerConfig":
        load_env_file()
        base = AgentConfig.from_env()
        cartesia_api_key = os.environ.get("CARTESIA_API_KEY", "").strip()
        cartesia_voice_id = os.environ.get("CARTESIA_VOICE_ID", "").strip()
        if not cartesia_api_key:
            raise ValueError("Missing CARTESIA_API_KEY in environment or .env")
        if not cartesia_voice_id:
            raise ValueError("Missing CARTESIA_VOICE_ID in environment or .env")
        output_dir = Path(os.environ.get("STORY_AUDIO_DIR", ROOT_DIR / "src" / "tourism_superai" / "outputs" / "story_audio"))
        return cls(
            base=base,
            cartesia_api_key=cartesia_api_key,
            cartesia_voice_id=cartesia_voice_id,
            cartesia_model_id=os.environ.get("CARTESIA_MODEL_ID", "sonic-3").strip(),
            cartesia_version=os.environ.get("CARTESIA_VERSION", "2025-04-16").strip(),
            output_dir=output_dir,
            min_retrigger_distance_m=float(os.environ.get("STORY_MIN_RETRIGGER_DISTANCE_M", 500)),
            cooldown_seconds=float(os.environ.get("STORY_COOLDOWN_SECONDS", 900)),
        )


class CartesiaVoiceClient:
    """Client wrapper for Cartesia text-to-speech audio generation."""

    def __init__(self, config: StorytellerConfig) -> None:
        self.config = config

    def synthesize_to_file(
        self,
        transcript: str,
        output_path: Path,
        *,
        language: str = "th",
    ) -> Path:
        """Synthesize speech from text and save to a file.

        Args:
            transcript (str): The text to synthesize.
            output_path (Path): Path to save the audio file.
            language (str): Language code for synthesis. Defaults to "th".

        Returns:
            Path: The output path.
        """
        payload = {
            "model_id": self.config.cartesia_model_id,
            "transcript": transcript,
            "voice": {"mode": "id", "id": self.config.cartesia_voice_id},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 44100,
            },
            "language": language,
            "generation_config": {
                "speed": 1,
                "volume": 1,
                "emotion": "neutral",
            },
            "save": False,
        }
        raw_audio, _ = _http_request(
            CARTESIA_TTS_URL,
            method="POST",
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.config.cartesia_api_key}",
                "Cartesia-Version": self.config.cartesia_version,
            },
            timeout=self.config.base.request_timeout,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw_audio)
        return output_path


class StorytellerAgent:
    """Agent that generates location-based narration and optional audio."""
    SYSTEM_PROMPT = textwrap.dedent(
        """
        You are storyteller, an in-car narration agent.
        Turn real-world place context into a short spoken story that feels cinematic, factual, and easy to listen to while driving.

        Rules:
        - Output only valid JSON.
        - Use the supplied research notes first; do not invent fake landmarks.
        - Keep the narration concise enough for a 30 to 60 second spoken segment.
        - Mix history, legend, local identity, architecture, or cultural context when available.
        - Be vivid, but do not overclaim uncertain facts.
        - Match the user's language when possible.

        Required JSON shape:
        {
          "place_name": "string",
          "region_context": "string",
          "headline": "string",
          "story_hook": "string",
          "narration_script": "string",
          "fact_highlights": ["string"],
          "sources": ["url"]
        }
        """
    ).strip()

    def __init__(
        self,
        *,
        config: StorytellerConfig | None = None,
        llm: GroqChatClient | None = None,
        retriever: KnowledgeRetriever | None = None,
        voice: CartesiaVoiceClient | None = None,
    ) -> None:
        self.config = config or StorytellerConfig.from_env()
        self.llm = llm or GroqChatClient(self.config.base)
        self.retriever = retriever or KnowledgeRetriever(self.config.base)
        self.voice = voice or CartesiaVoiceClient(self.config)

    def create_story(
        self,
        *,
        keyword: str | None,
        latitude: float,
        longitude: float,
        user_context: str | None = None,
        language: str = "th",
        generate_audio: bool = True,
    ) -> StoryResult:
        """Create a story for a location with optional audio.

        Args:
            keyword (str | None): Location keyword.
            latitude (float): Latitude.
            longitude (float): Longitude.
            user_context (str | None): User context. Defaults to None.
            language (str): Language code. Defaults to "th".
            generate_audio (bool): Whether to generate audio. Defaults to True.

        Returns:
            StoryResult: The generated story result.
        """
        reverse_geocode = self.retriever.reverse_geocode(latitude, longitude) or {}
        inferred_name = self._pick_place_name(keyword, reverse_geocode)
        research_results = self._gather_research(inferred_name, reverse_geocode)
        research_context = self._format_research(research_results)
        region_context = str(reverse_geocode.get("display_name", "")).strip()

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Keyword trigger: {keyword or 'None'}
                    Coordinates: {latitude}, {longitude}
                    Requested narration language: {language}
                    Reverse geocode: {json.dumps(reverse_geocode, ensure_ascii=False)}
                    User travel context: {user_context or 'No extra context'}

                    Research notes:
                    {research_context}
                    """
                ).strip(),
            },
        ]
        payload = self.llm.generate_json(messages, temperature=0.5, max_tokens=1200)
        story = StoryResult(
            trigger_keyword=keyword or inferred_name,
            place_name=str(payload.get("place_name", inferred_name)),
            latitude=latitude,
            longitude=longitude,
            region_context=str(payload.get("region_context", region_context)),
            headline=str(payload.get("headline", "")),
            story_hook=str(payload.get("story_hook", "")),
            narration_script=str(payload.get("narration_script", "")),
            fact_highlights=_as_string_list(payload.get("fact_highlights")),
            sources=_as_string_list(payload.get("sources")),
        )
        if not story.sources:
            story.sources = [item.url for item in research_results[:5]]

        if generate_audio and story.narration_script:
            try:
                output_path = self._make_output_path(story.place_name)
                audio_path = self.voice.synthesize_to_file(
                    story.narration_script,
                    output_path,
                    language=language,
                )
                story.audio_path = str(audio_path)
            except Exception as exc:  # noqa: BLE001
                story.audio_error = str(exc)
        return story

    def _pick_place_name(self, keyword: str | None, reverse_geocode: dict[str, Any]) -> str:
        """Pick the best place name from keyword or geocode data.

        Args:
            keyword (str | None): The keyword.
            reverse_geocode (dict[str, Any]): Reverse geocode data.

        Returns:
            str: The selected place name.
        """
        if keyword and keyword.strip():
            return keyword.strip()
        name = str(reverse_geocode.get("name", "")).strip()
        if name:
            return name
        display_name = str(reverse_geocode.get("display_name", "")).strip()
        if display_name:
            return display_name.split(",")[0].strip()
        return "nearby landmark"

    def _gather_research(
        self,
        place_name: str,
        reverse_geocode: dict[str, Any],
        *,
        limit: int = 8,
    ) -> list[SearchResult]:
        """Gather research results for a place.

        Args:
            place_name (str): The place name.
            reverse_geocode (dict[str, Any]): Reverse geocode data.
            limit (int): Max results. Defaults to 8.

        Returns:
            list[SearchResult]: List of search results.
        """
        region_hint = str(reverse_geocode.get("display_name", "")).strip()
        queries = [
            f"{place_name} history legend landmark",
            f"{place_name} cultural significance",
            f"{place_name} {region_hint}".strip(),
        ]
        seen: set[str] = set()
        results: list[SearchResult] = []
        for query in queries:
            for item in self.retriever.search_web(query, limit=4):
                if item.url in seen:
                    continue
                seen.add(item.url)
                results.append(item)
                if len(results) >= limit:
                    return results

        if len(results) < max(3, limit // 2):
            for item in self.retriever.wikipedia_search(place_name, limit=4):
                if item.url in seen:
                    continue
                seen.add(item.url)
                results.append(item)
                if len(results) >= limit:
                    break
        return results

    def _format_research(self, results: list[SearchResult]) -> str:
        """Format research results into a context string.

        Args:
            results (list[SearchResult]): The search results.

        Returns:
            str: Formatted context string.
        """
        if not results:
            return "No web results available. Use only cautious general knowledge."
        return "\n\n---\n\n".join(item.to_context_block() for item in results)

    def _make_output_path(self, place_name: str) -> Path:
        """Create a unique output path for audio file.

        Args:
            place_name (str): The place name.

        Returns:
            Path: The output file path.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{_slugify(place_name)}_{timestamp}.wav"
        return self.config.output_dir / filename


@dataclass(slots=True)
class TriggerRecord:
    canonical_key: str
    latitude: float
    longitude: float
    triggered_at: float


class LocationStoryMonitor:
    """Monitors location updates and triggers stories with cooldown control."""

    def __init__(self, storyteller: StorytellerAgent | None = None) -> None:
        self.storyteller = storyteller or StorytellerAgent()
        self.trigger_history: list[TriggerRecord] = []

    def on_position_update(
        self,
        *,
        keyword: str | None,
        latitude: float,
        longitude: float,
        user_context: str | None = None,
        language: str = "th",
        generate_audio: bool = True,
    ) -> StoryResult | None:
        """Handle position update and trigger story if appropriate.

        Args:
            keyword (str | None): Location keyword.
            latitude (float): Latitude.
            longitude (float): Longitude.
            user_context (str | None): User context. Defaults to None.
            language (str): Language code. Defaults to "th".
            generate_audio (bool): Whether to generate audio. Defaults to True.

        Returns:
            StoryResult | None: The story result or None if not triggered.
        """
        canonical_key = self._canonical_key(keyword, latitude, longitude)
        now = time.time()
        if not self._should_trigger(canonical_key, latitude, longitude, now):
            return None

        story = self.storyteller.create_story(
            keyword=keyword,
            latitude=latitude,
            longitude=longitude,
            user_context=user_context,
            language=language,
            generate_audio=generate_audio,
        )
        self.trigger_history.append(
            TriggerRecord(
                canonical_key=canonical_key,
                latitude=latitude,
                longitude=longitude,
                triggered_at=now,
            )
        )
        self._trim_history(now)
        return story

    def _canonical_key(self, keyword: str | None, latitude: float, longitude: float) -> str:
        """Generate a canonical key for location deduplication.

        Args:
            keyword (str | None): The keyword.
            latitude (float): Latitude.
            longitude (float): Longitude.

        Returns:
            str: The canonical key.
        """
        if keyword and keyword.strip():
            return keyword.strip().lower()
        lat_bucket = round(latitude, 3)
        lon_bucket = round(longitude, 3)
        return f"{lat_bucket}:{lon_bucket}"

    def _should_trigger(
        self,
        canonical_key: str,
        latitude: float,
        longitude: float,
        now: float,
    ) -> bool:
        """Determine if a story should be triggered based on history and distance.

        Args:
            canonical_key (str): The canonical key.
            latitude (float): Latitude.
            longitude (float): Longitude.
            now (float): Current timestamp.

        Returns:
            bool: Whether to trigger.
        """
        distance_limit = self.storyteller.config.min_retrigger_distance_m
        cooldown = self.storyteller.config.cooldown_seconds
        for record in reversed(self.trigger_history):
            age = now - record.triggered_at
            if age > cooldown:
                continue
            if record.canonical_key == canonical_key:
                return False
            distance = _haversine_meters(record.latitude, record.longitude, latitude, longitude)
            if distance <= distance_limit:
                return False
        return True

    def _trim_history(self, now: float) -> None:
        """Trim old trigger history entries.

        Args:
            now (float): Current timestamp.

        Returns:
            None
        """
        cooldown = self.storyteller.config.cooldown_seconds
        self.trigger_history = [
            item for item in self.trigger_history if now - item.triggered_at <= cooldown * 2
        ]


def _build_parser() -> argparse.ArgumentParser:
    """Create the command line parser for the storyteller.

    Args:
        None

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Landmark storyteller powered by Groq + Cartesia")
    parser.add_argument("--keyword", help="Detected place keyword, e.g. 'Wat Arun'")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--context", help="Optional user travel context")
    parser.add_argument("--language", default="th", help="Narration language code, e.g. th or en")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    parser.add_argument("--skip-audio", action="store_true", help="Skip Cartesia voice synthesis")
    return parser


def main() -> None:
    """Run the storyteller from command line.

    Args:
        None

    Returns:
        None
    """
    parser = _build_parser()
    args = parser.parse_args()
    agent = StorytellerAgent()
    story = agent.create_story(
        keyword=args.keyword,
        latitude=args.lat,
        longitude=args.lon,
        user_context=args.context,
        language=args.language,
        generate_audio=not args.skip_audio,
    )
    if args.json:
        print(json.dumps(story.to_dict(), indent=2, ensure_ascii=False))
        return
    print(story.pretty_print())


if __name__ == "__main__":
    main()
