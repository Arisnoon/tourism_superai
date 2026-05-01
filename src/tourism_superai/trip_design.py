from __future__ import annotations

"""Trip planning and research retrieval backend for the Agentic AI Roadtrip service."""

import argparse
import json
import os
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def load_env_file(env_path: Path = ENV_PATH) -> None:
    """Load environment values from a local .env file if present.

    This supports both a backend-local `.env` and a repository root `.env`.
    """
    search_paths = [env_path]
    if not env_path.exists():
        root_env = env_path.parent.parent / ".env"
        if root_env != env_path:
            search_paths.append(root_env)

    for path in search_paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                # Skip malformed env entries like '=value' instead of 'KEY=value'.
                continue
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)
        return


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from text.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from text, stripping code fences if present.

    Args:
        text (str): The text containing JSON.

    Returns:
        dict[str, Any]: The parsed JSON object.

    Raises:
        ValueError: If the text is not valid JSON.
    """
    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model output is not JSON: {cleaned[:300]}")
    return json.loads(cleaned[start : end + 1])


def _http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 25.0,
) -> tuple[bytes, dict[str, str]]:
    """Make an HTTP request and return response data and headers.

    Args:
        url (str): The URL to request.
        method (str): HTTP method. Defaults to "GET".
        headers (dict[str, str] | None): Request headers. Defaults to None.
        payload (dict[str, Any] | None): JSON payload. Defaults to None.
        timeout (float): Request timeout. Defaults to 25.0.

    Returns:
        tuple[bytes, dict[str, str]]: Response body and headers.
    """
    body: bytes | None = None
    request_headers = dict(headers or {})
    request_headers.setdefault(
        "User-Agent",
        os.environ.get("WIKIMEDIA_USER_AGENT", "AgenticAIRoadtrip/1.0 (backend request)"),
    )
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(url, data=body, method=method, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), dict(response.headers.items())


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    engine: str | None = None

    def to_context_block(self) -> str:
        """Format the search result as a context block.

        Args:
            None

        Returns:
            str: Formatted context string.
        """
        lines = [
            f"Title: {self.title}",
            f"Snippet: {self.snippet or 'No snippet available.'}",
            f"URL: {self.url}",
            f"Source: {self.source}",
        ]
        if self.engine:
            lines.append(f"Engine: {self.engine}")
        return "\n".join(lines)


@dataclass(slots=True)
class TripStop:
    """Structured itinerary stop for a trip plan."""
    time: str
    place_name: str
    area: str
    activity: str
    why_this_place: str
    travel_notes: str
    estimated_latitude: float | None = None
    estimated_longitude: float | None = None
    source_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TripPlan:
    """Data structure representing a complete trip itinerary."""
    trip_title: str
    destination: str
    summary: str
    travel_style: str
    stops: list[TripStop]
    food_recommendations: list[str] = field(default_factory=list)
    practical_tips: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TripPlan":
        """Create a TripPlan from a dictionary.

        Args:
            payload (dict[str, Any]): The dictionary payload.

        Returns:
            TripPlan: The created TripPlan instance.
        """
        stops = [
            TripStop(
                time=str(item.get("time", "")),
                place_name=str(item.get("place_name", "")),
                area=str(item.get("area", "")),
                activity=str(item.get("activity", "")),
                why_this_place=str(item.get("why_this_place", "")),
                travel_notes=str(item.get("travel_notes", "")),
                estimated_latitude=_to_float(item.get("estimated_latitude")),
                estimated_longitude=_to_float(item.get("estimated_longitude")),
                source_refs=_as_string_list(item.get("source_refs")),
            )
            for item in payload.get("stops", [])
            if isinstance(item, dict)
        ]
        return cls(
            trip_title=str(payload.get("trip_title", "Roadtrip Plan")),
            destination=str(payload.get("destination", "Unknown destination")),
            summary=str(payload.get("summary", "")),
            travel_style=str(payload.get("travel_style", "balanced")),
            stops=stops,
            food_recommendations=_as_string_list(payload.get("food_recommendations")),
            practical_tips=_as_string_list(payload.get("practical_tips")),
            sources=_as_string_list(payload.get("sources")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the TripPlan to a dictionary.

        Args:
            None

        Returns:
            dict[str, Any]: The dictionary representation.
        """
        return asdict(self)

    def pretty_print(self) -> str:
        """Generate a pretty-printed string representation of the trip.

        Args:
            None

        Returns:
            str: The formatted trip description.
        """
        lines = [
            f"Trip: {self.trip_title}",
            f"Destination: {self.destination}",
            f"Style: {self.travel_style}",
            f"Summary: {self.summary}",
            "",
            "Timeline:",
        ]
        for stop in self.stops:
            lines.extend(
                [
                    f"- {stop.time} | {stop.place_name} ({stop.area})",
                    f"  Activity: {stop.activity}",
                    f"  Why: {stop.why_this_place}",
                    f"  Notes: {stop.travel_notes}",
                ]
            )
            if stop.source_refs:
                lines.append(f"  Sources: {', '.join(stop.source_refs)}")
        if self.food_recommendations:
            lines.extend(["", "Food recommendations:"])
            lines.extend(f"- {item}" for item in self.food_recommendations)
        if self.practical_tips:
            lines.extend(["", "Practical tips:"])
            lines.extend(f"- {item}" for item in self.practical_tips)
        if self.sources:
            lines.extend(["", "Top sources:"])
            lines.extend(f"- {item}" for item in self.sources)
        return "\n".join(lines)


def _to_float(value: Any) -> float | None:
    """Convert a value to float or None.

    Args:
        value (Any): The value to convert.

    Returns:
        float | None: The float value or None.
    """
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_string_list(value: Any) -> list[str]:
    """Convert a value to a list of strings.

    Args:
        value (Any): The value to convert.

    Returns:
        list[str]: The list of strings.
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


@dataclass(slots=True)
class AgentConfig:
    """Configuration for Groq and search retrieval clients."""
    groq_api_key: str
    searxng_base_url: str | None
    groq_model: str
    wikimedia_user_agent: str
    request_timeout: float = 25.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        load_env_file()
        groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment or .env")
        return cls(
            groq_api_key=groq_api_key,
            searxng_base_url=os.environ.get("SEARXNG_BASE_URL", "").strip() or None,
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            wikimedia_user_agent=os.environ.get(
                "WIKIMEDIA_USER_AGENT",
                "AgenticAIRoadtrip/1.0 (trip planning assistant)",
            ).strip(),
        )


class GroqChatClient:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        """Generate a JSON response from Groq API.

        Args:
            messages (list[dict[str, str]]): The conversation messages.
            temperature (float): Sampling temperature. Defaults to 0.3.
            max_tokens (int): Max tokens. Defaults to 1800.

        Returns:
            dict[str, Any]: The generated JSON.
        """
        payload = {
            "model": self.config.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        raw, _ = _http_request(
            GROQ_API_URL,
            method="POST",
            payload=payload,
            headers={"Authorization": f"Bearer {self.config.groq_api_key}"},
            timeout=self.config.request_timeout,
        )
        data = json.loads(raw.decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Groq response has no choices: {data}")
        content = choices[0].get("message", {}).get("content", "")
        return _extract_json_object(content)


class KnowledgeRetriever:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def search_web(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Search the web using SearxNG.

        Args:
            query (str): The search query.
            limit (int): Max results. Defaults to 5.

        Returns:
            list[SearchResult]: The search results.
        """
        if not self.config.searxng_base_url:
            return []

        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "language": "auto",
                "safesearch": 1,
            }
        )
        base = self.config.searxng_base_url.rstrip("/")
        url = f"{base}/search?{params}"

        try:
            raw, _ = _http_request(url, timeout=self.config.request_timeout)
            payload = json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return []

        results = []
        for item in payload.get("results", [])[:limit]:
            title = str(item.get("title", "")).strip()
            link = str(item.get("url", "")).strip()
            snippet = str(item.get("content", "")).strip()
            if not title or not link:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source="searxng",
                    engine=str(item.get("engine", "")).strip() or None,
                )
            )
        return results

    def wikipedia_search(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        """Search Wikipedia for articles.

        Args:
            query (str): The search query.
            limit (int): Max results. Defaults to 3.

        Returns:
            list[SearchResult]: The search results.
        """
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
                "utf8": 1,
            }
        )
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        headers = {"User-Agent": self.config.wikimedia_user_agent}
        try:
            raw, _ = _http_request(url, headers=headers, timeout=self.config.request_timeout)
            payload = json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return []

        pages = (payload.get("query") or {}).get("pages") or {}
        results = []
        for page in pages.values():
            title = str(page.get("title", "")).strip()
            link = str(page.get("fullurl", "")).strip()
            snippet = str(page.get("extract", "")).strip()
            if not title or not link:
                continue
            results.append(SearchResult(title=title, url=link, snippet=snippet, source="wikipedia"))
        return results

    def reverse_geocode(self, latitude: float, longitude: float) -> dict[str, Any] | None:
        """Perform reverse geocoding using Nominatim.

        Args:
            latitude (float): Latitude.
            longitude (float): Longitude.

        Returns:
            dict[str, Any] | None: The geocode data or None.
        """
        params = urllib.parse.urlencode({"format": "jsonv2", "lat": latitude, "lon": longitude})
        url = f"https://nominatim.openstreetmap.org/reverse?{params}"
        headers = {"User-Agent": self.config.wikimedia_user_agent}
        try:
            raw, _ = _http_request(url, headers=headers, timeout=self.config.request_timeout)
            payload = json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return None
        return payload

    def gather_trip_research(self, prompt: str, *, limit: int = 10) -> list[SearchResult]:
        """Gather research for trip planning.

        Args:
            prompt (str): The research prompt.
            limit (int): Max results. Defaults to 10.

        Returns:
            list[SearchResult]: The search results.
        """
        queries = [
            prompt,
            f"{prompt} attractions landmarks",
            f"{prompt} local food cafes history",
        ]
        seen: set[str] = set()
        collected: list[SearchResult] = []
        for query in queries:
            for result in self.search_web(query, limit=4):
                if result.url in seen:
                    continue
                seen.add(result.url)
                collected.append(result)
                if len(collected) >= limit:
                    return collected

        if len(collected) < max(3, limit // 2):
            for result in self.wikipedia_search(prompt, limit=4):
                if result.url in seen:
                    continue
                seen.add(result.url)
                collected.append(result)
                if len(collected) >= limit:
                    break
        return collected


class TripDesignAgent:
    """Agent that generates one-day trip plans from user prompts."""
    SYSTEM_PROMPT = textwrap.dedent(
        """
        You are trip_design, a travel planner for roadtrip and city-trip chatbots.
        Build a realistic one-day trip using only the user request plus provided research notes.

        Rules:
        - Output only valid JSON.
        - Keep the itinerary grounded in the provided real places.
        - Build a full-day plan in chronological order.
        - Prefer 5 to 8 stops.
        - Include practical travel notes, parking/transit hints, meal timing, and why each stop fits.
        - If a fact is uncertain, keep wording careful instead of inventing certainty.
        - Reply in the same language as the user's prompt when possible.

        Required JSON shape:
        {
          "trip_title": "string",
          "destination": "string",
          "summary": "string",
          "travel_style": "string",
          "stops": [
            {
              "time": "HH:MM-HH:MM",
              "place_name": "string",
              "area": "string",
              "activity": "string",
              "why_this_place": "string",
              "travel_notes": "string",
              "estimated_latitude": 0.0,
              "estimated_longitude": 0.0,
              "source_refs": ["url"]
            }
          ],
          "food_recommendations": ["string"],
          "practical_tips": ["string"],
          "sources": ["url"]
        }
        """
    ).strip()

    def __init__(
        self,
        *,
        config: AgentConfig | None = None,
        llm: GroqChatClient | None = None,
        retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.llm = llm or GroqChatClient(self.config)
        self.retriever = retriever or KnowledgeRetriever(self.config)
        self.chat_history: list[dict[str, str]] = []
        self.last_destination_hint: str | None = None

    def plan_trip(self, user_prompt: str) -> TripPlan:
        """Plan a trip based on user prompt.

        Args:
            user_prompt (str): The user's trip request.

        Returns:
            TripPlan: The planned trip.
        """
        search_prompt = self._build_search_prompt(user_prompt)
        research_results = self.retriever.gather_trip_research(search_prompt)
        research_context = self._format_research(research_results)
        history_context = self._format_history()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    User prompt:
                    {user_prompt}

                    Conversation context:
                    {history_context}

                    Search / research notes:
                    {research_context}
                    """
                ).strip(),
            },
        ]
        payload = self.llm.generate_json(messages)
        trip = TripPlan.from_dict(payload)
        if not trip.sources:
            trip.sources = [item.url for item in research_results[:5]]
        for stop in trip.stops:
            if not stop.source_refs:
                stop.source_refs = trip.sources[:2]
        if trip.destination:
            self.last_destination_hint = trip.destination
        self.chat_history.append({"user": user_prompt, "assistant": trip.summary})
        return trip

    def chat(self, user_prompt: str) -> TripPlan:
        """Chat interface for trip planning.

        Args:
            user_prompt (str): The user's message.

        Returns:
            TripPlan: The trip plan response.
        """
        return self.plan_trip(user_prompt)

    def _build_search_prompt(self, user_prompt: str) -> str:
        """Build a search prompt from user input.

        Args:
            user_prompt (str): The user prompt.

        Returns:
            str: The search prompt.
        """
        compact = user_prompt.strip()
        if self.last_destination_hint and len(compact) < 40:
            return f"{self.last_destination_hint} {compact}".strip()
        return compact

    def _format_history(self) -> str:
        """Format conversation history for context.

        Args:
            None

        Returns:
            str: The formatted history.
        """
        if not self.chat_history:
            return "No previous messages."
        recent = self.chat_history[-3:]
        blocks = []
        for item in recent:
            blocks.append(f"User: {item['user']}\nAssistant summary: {item['assistant']}")
        return "\n\n".join(blocks)

    def _format_research(self, results: Iterable[SearchResult]) -> str:
        """Format research results into context string.

        Args:
            results (Iterable[SearchResult]): The search results.

        Returns:
            str: The formatted context.
        """
        blocks = [result.to_context_block() for result in results]
        if not blocks:
            return "No web results available. Use general knowledge carefully and clearly state uncertainty."
        return "\n\n---\n\n".join(blocks)


def _build_parser() -> argparse.ArgumentParser:
    """Create the command line parser for trip design.

    Args:
        None

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Trip planning agent powered by Groq + SearxNG")
    parser.add_argument("--prompt", help="Prompt describing the trip you want")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted text")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start a simple chatbot loop for follow-up trip edits",
    )
    return parser


def _interactive_chat(agent: TripDesignAgent) -> None:
    print("trip_design chat started. Type 'exit' to stop.")
    while True:
        prompt = input("You> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return
        trip = agent.chat(prompt)
        print(trip.pretty_print())
        print("")


def main() -> None:
    """Run the trip design agent from command line.

    Args:
        None

    Returns:
        None
    """
    parser = _build_parser()
    args = parser.parse_args()
    agent = TripDesignAgent()
    if args.interactive:
        _interactive_chat(agent)
        return
    if not args.prompt:
        parser.error("provide --prompt or use --interactive")
    trip = agent.plan_trip(args.prompt)
    if args.json:
        print(json.dumps(trip.to_dict(), indent=2, ensure_ascii=False))
        return
    print(trip.pretty_print())


if __name__ == "__main__":
    main()
