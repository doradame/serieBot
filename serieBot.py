#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SerieBot — Flexible slot filling → geo (ipinfo) → TMDB by Country → 3 reasoned suggestions.
LangChain 1.0 Innovations:
- LCEL pipeline (ChatPromptTemplate | llm | parser)
- Structured output with Pydantic for slot-filling
- Streaming responses for better UX
- LangGraph for conversation state management
- Tool binding native (bind_tools) for ipinfo / TMDB
- Dynamic provider selection by region (TMDB watch providers)
Author: MojaLab
Improvements: Enhanced validation, error handling, caching, streaming, and UX
"""
import os
import re
import json
import time
import textwrap
import requests
import logging
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableSequence
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")

# =========================
# Logging Configuration
# =========================
def setup_logging():
    """Configure logging with both file and console handlers."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    log_filename = os.path.join(log_dir, f"seriebot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only warnings and errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('⚠️  %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log startup
    logger.info("=" * 80)
    logger.info("SerieBot Started")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info("=" * 80)
    
    return logger

# =========================
# Config & env
# =========================
TMDB_KEY = os.getenv("TMDB_API_KEY")
OVH_BASE = os.getenv("OPENAI_API_BASE", "https://api.ai.ovh.net/v1")
OVH_KEY  = os.getenv("OPENAI_API_KEY")
MODEL    = os.getenv("OVH_MODEL") or None  # None allows the API to use its default model
DEBUG    = os.getenv("DEBUG", "false").lower() == "true"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

# Initialize logging
logger = setup_logging()
logger.info(f"Configuration loaded - TMDB: {'✓' if TMDB_KEY else '✗'}, OVH: {'✓' if OVH_KEY else '✗'}")
logger.info(f"Model: {MODEL or 'gpt-oss-120b'}, Base URL: {OVH_BASE}")

if not OVH_KEY:
    raise SystemExit("Missing OPENAI_API_KEY (OVH AI Endpoints).")
if not TMDB_KEY:
    print("WARNING: Missing TMDB_API_KEY — suggestions might fail.", flush=True)

# =========================
# Feedback System
# =========================
FEEDBACK_DIR = "feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.csv")

def setup_feedback_system():
    """Initialize feedback directory and CSV file."""
    if not os.path.exists(FEEDBACK_DIR):
        os.makedirs(FEEDBACK_DIR)
        logger.info(f"Created feedback directory: {FEEDBACK_DIR}")
    
    # Create CSV with headers if it doesn't exist
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'session_id', 'rating', 'genre', 'mood', 
                'duration', 'providers', 'language', 'country', 
                'suggestions_count', 'comment'
            ])
        logger.info(f"Created feedback file: {FEEDBACK_FILE}")

def save_feedback(session_id: str, rating: str, prefs: 'UserPrefs', 
                 country: str, suggestions_count: int, comment: str = ""):
    """Save user feedback to CSV file."""
    try:
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                session_id,
                rating,
                prefs.genre or "N/A",
                prefs.mood or "N/A",
                prefs.duration or "N/A",
                prefs.providers or "N/A",
                prefs.language or "N/A",
                country,
                suggestions_count,
                comment
            ])
        logger.info(f"Feedback saved: {rating} for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False

# Initialize feedback system
setup_feedback_system()

# =========================
# Helpers
# =========================
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

PROVIDER_SYNONYMS = {
    "netflix": ["netflix"],
    "prime video": ["prime", "primevideo", "amazon", "amazonprime", "amazon prime video", "prime video", "amazon prime"],
    "disney+": ["disney", "disneyplus", "disney+", "disney plus"],
    "apple tv+": ["apple", "appletv", "appletv+", "apple tv+", "apple tv"],
    "paramount+": ["paramount", "paramount+", "paramount plus"],
    "now": ["now", "nowtv"],
    "hbo max": ["hbomax", "max", "hbo max"],
    "rakuten tv": ["rakuten", "rakutentv"],
    "mediaset infinity": ["infinity", "mediasetinfinity", "infinity+"],
}

def normalize_providers(user_text: str) -> List[str]:
    if not user_text:
        return []
    # Accept comma/semicolon/"and"/multi-space separated inputs
    tokens = [t.strip() for t in re.split(r"[,\;]| and |\s{2,}", user_text, flags=re.I) if t.strip()]
    out = set()
    for t in tokens:
        st = _slug(t)
        for canonical, syns in PROVIDER_SYNONYMS.items():
            if st == _slug(canonical) or st in [_slug(x) for x in syns]:
                out.add(canonical)
    return list(out)

# =========================
# Pydantic models (typed I/O)
# =========================
ALLOWED_GENRES = {"sci-fi","crime","drama","comedy","animation","fantasy","mystery","documentary"}
ALLOWED_MOOD = {"light-hearted","intense","mind-blowing","comforting","adrenaline-fueled"}
ALLOWED_DURATION = {"<30m","~45m",">60m"}
ALLOWED_LANG = {"it","en","any"}

class UserPrefs(BaseModel):
    """Holds all user preferences."""
    genre: Optional[str] = None
    mood: Optional[str] = None
    duration: Optional[str] = None
    providers: Optional[str] = None
    language: Optional[str] = None
    
    @field_validator('providers', mode='before')
    @classmethod
    def convert_providers_list_to_string(cls, v):
        """Convert providers list to comma-separated string if it's a list."""
        if isinstance(v, list):
            return ", ".join(str(item) for item in v)
        return v

class CollectResult(BaseModel):
    """Structured output: partial slots + optional brief follow-up question."""
    known: UserPrefs = Field(default_factory=UserPrefs)
    ask_next: Optional[str] = None

class Suggestions(BaseModel):
    country: str
    suggestions: List[Dict] = Field(default_factory=list)
    note: Optional[str] = None

# =========================
# LangGraph State Definition
# =========================
class ConversationState(TypedDict):
    """State for LangGraph conversation flow."""
    # User preferences being collected
    prefs: UserPrefs
    
    # Conversation metadata
    session_id: str
    user_input: str
    next_question: Optional[str]
    
    # Geolocation cache
    geo_cache: Optional[Dict]
    
    # Suggestions tracking
    last_suggestions_count: int
    
    # Flow control
    waiting_for_feedback: bool
    prefs_complete: bool
    
    # Current state
    current_state: Literal["greeting", "collecting", "searching", "feedback", "end"]

# =========================
# Validation
# =========================
def validate_prefs(prefs: UserPrefs) -> UserPrefs:
    """Post-process to ensure only valid values make it through."""
    validated_data = {}
    
    # Validate genre
    if prefs.genre:
        genre_lower = prefs.genre.lower()
        if genre_lower in ALLOWED_GENRES:
            validated_data['genre'] = genre_lower
        else:
            validated_data['genre'] = None
    else:
        validated_data['genre'] = None
    
    # Validate mood
    if prefs.mood:
        mood_lower = prefs.mood.lower()
        if mood_lower in ALLOWED_MOOD:
            validated_data['mood'] = mood_lower
        else:
            validated_data['mood'] = None
    else:
        validated_data['mood'] = None
    
    # Validate duration
    if prefs.duration:
        duration_lower = prefs.duration.lower()
        if duration_lower in ALLOWED_DURATION:
            validated_data['duration'] = duration_lower
        else:
            validated_data['duration'] = None
    else:
        validated_data['duration'] = None
    
    # Validate language
    if prefs.language:
        lang_lower = prefs.language.lower()
        if lang_lower in ALLOWED_LANG:
            validated_data['language'] = lang_lower
        else:
            validated_data['language'] = None
    else:
        validated_data['language'] = None
    
    # Providers - just pass through (will be normalized later)
    validated_data['providers'] = prefs.providers
    
    return UserPrefs(**validated_data)

# =========================
# Tools (ipinfo + TMDB)
# =========================
@tool
def ipinfo_location() -> str:
    """Get user's location (city, country, timezone) using ipinfo.io (no key)."""
    logger.info("Fetching user location from ipinfo.io")
    try:
        r = requests.get("https://ipinfo.io/json", timeout=6)
        r.raise_for_status()
        d = r.json()
        result = {
            "city": d.get("city"),
            "region": d.get("region"),
            "country": d.get("country", "US"),
            "timezone": d.get("timezone"),
        }
        logger.info(f"Location detected: {result.get('city')}, {result.get('country')}")
        return json.dumps(result, ensure_ascii=False)
    except requests.RequestException as e:
        logger.error(f"Failed to get location: {str(e)}")
        return json.dumps({"error": f"Failed to get location: {str(e)}", "country": "US"}, ensure_ascii=False)

# --- TMDB helpers/classes
class TMDB:
    BASE = "https://api.themoviedb.org/3"
    RATE_LIMIT_DELAY = float(os.getenv("TMDB_RATE_LIMIT", "0.25"))  # Rate limit from env
    
    def __init__(self, api_key: str):
        self.key = api_key
        self._providers_cache: Dict[str, Dict[str,int]] = {}
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def region_provider_map(self, region: str) -> Dict[str,int]:
        region = (region or "US").upper()
        if region in self._providers_cache:
            logger.debug(f"Using cached provider map for {region}")
            return self._providers_cache[region]
        
        logger.info(f"Fetching provider map for region: {region}")
        self._rate_limit()
        url = f"{self.BASE}/watch/providers/tv"
        p = {"api_key": self.key, "watch_region": region}
        
        try:
            r = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            items = r.json().get("results", [])
            mp = {}
            for it in items:
                name = (it.get("provider_name") or "").lower()
                pid  = it.get("provider_id")
                if name and pid:
                    mp[name] = pid
            self._providers_cache[region] = mp
            logger.info(f"Found {len(mp)} providers for {region}")
            return mp
        except requests.RequestException as e:
            logger.error(f"Failed to fetch provider map for {region}: {e}")
            print(f"Warning: Failed to fetch provider map for {region}: {e}")
            return {}
    
    def discover_tv(self, params: Dict) -> List[Dict]:
        self._rate_limit()
        url = f"{self.BASE}/discover/tv"
        p = {"api_key": self.key, **{k:v for k,v in params.items() if v is not None}}
        r = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("results", [])
    
    def watch_providers_for(self, tv_id: int, region: str) -> List[str]:
        self._rate_limit()
        url = f"{self.BASE}/tv/{tv_id}/watch/providers"
        p = {"api_key": self.key}
        r = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        res = (r.json().get("results") or {}).get(region.upper(), {})
        flatrate = res.get("flatrate", []) or []
        return [x.get("provider_name") for x in flatrate if x.get("provider_name")]

tmdb = TMDB(TMDB_KEY) if TMDB_KEY else None

GENRE_MAP = {
    "sci-fi": 10765,  # Sci-Fi & Fantasy
    "comedy": 35,
    "drama": 18,
    "crime": 80,
    "animation": 16,
    "mystery": 9648,
    "documentary": 99,
    "fantasy": 10765,  # shares the Sci-Fi & Fantasy bucket
}

def mood_bias(mood: str) -> Tuple[List[int], List[int]]:
    mood = (mood or "").lower()
    prefer, avoid = [], []
    if "light" in mood or "comfort" in mood:
        prefer = [35,16]     # comedy, animation
        avoid  = [80]        # crime
    elif "intens" in mood:
        prefer = [18,80]     # drama, crime
    elif "mind" in mood:
        prefer = [10765,9648,16]  # sci-fi&fantasy, mystery, animation
    elif "adren" in mood:
        prefer = [80]        # crime (placeholder for action)
    return prefer, avoid

def build_discover_params(genre, mood, duration, region, language):
    p = {
        "watch_region": region,
        "sort_by": "popularity.desc",
        "with_watch_monetization_types": "flatrate",
        "with_original_language": None if (language == "any") else (language or None),
        "page": 1
    }
    gid = GENRE_MAP.get((genre or "").lower())
    if gid:
        p["with_genres"] = gid
    
    if duration == "<30m":
        p["with_runtime.lte"] = 30
    elif duration == "~45m":
        p["with_runtime.gte"], p["with_runtime.lte"] = 35, 55
    elif duration == ">60m":
        p["with_runtime.gte"] = 60
    
    prefer, avoid = mood_bias(mood)
    
    # If genre is specified, use mood only for avoid list
    # If no genre, use mood preferences for genre selection
    if not gid and prefer:
        p["with_genres"] = ",".join(str(x) for x in prefer)
    
    # Always apply avoid genres
    if avoid:
        p["without_genres"] = ",".join(str(x) for x in avoid)
    
    return p

def intersect_providers(user_prov: List[str], region_map: Dict[str,int]) -> List[int]:
    """Simplified provider intersection with better normalization."""
    want_ids = []
    for up in user_prov:
        up_normalized = up.lower().replace("+", "").replace(" ", "")
        for name, pid in region_map.items():
            name_normalized = name.lower().replace("+", "").replace(" ", "")
            if up_normalized == name_normalized:
                want_ids.append(pid)
                break
    # Dedupe preserving order
    return list(dict.fromkeys(want_ids))

def _trim_overview(overview: str, limit: int = 280) -> str:
    if not overview:
        return "(Overview not available)"
    overview = overview.strip()
    if len(overview) <= limit:
        return overview
    cut = overview[:limit]
    # Try to not cut mid-word/sentence
    for sep in [". ", "! ", "? ", "; ", ", ", " "]:
        i = cut.rfind(sep)
        if i > limit * 0.6:
            return cut[:i+1] + "…"
    return cut + "…"

def _score_genre_match(tv_genres, requested_genre):
    """
    Score how well a series matches the requested genre.
    For sci-fi requests, prefer series with more 'sci-fi' indicators and fewer fantasy/drama indicators.
    """
    if not requested_genre or requested_genre.lower() not in ["sci-fi", "scifi"]:
        return 0  # Don't apply scoring for non-sci-fi genres
    
    score = 0
    # Sci-Fi & Fantasy is the base (always present if we filtered by it)
    if 10765 in tv_genres:
        score += 1
    
    # Positive indicators for sci-fi (space, action, mystery)
    if 10759 in tv_genres:  # Action & Adventure
        score += 2
    if 9648 in tv_genres:  # Mystery
        score += 1
    if 16 in tv_genres:  # Animation (often sci-fi)
        score += 1
    
    # Negative indicators (more fantasy/drama than sci-fi)
    if 18 in tv_genres:  # Drama (very common in fantasy/vampire shows)
        score -= 1
    if 80 in tv_genres:  # Crime (often supernatural crime, not sci-fi)
        score -= 2
    if 10751 in tv_genres:  # Family
        score -= 1
    
    return score

def _score_freshness(first_air_date):
    """
    Score based on release year to promote newer content and diversify suggestions.
    Returns a freshness bonus: newer series get higher scores.
    """
    if not first_air_date:
        return 0
    
    try:
        from datetime import datetime
        year = int(first_air_date.split("-")[0])
        current_year = datetime.now().year
        age = current_year - year
        
        # Bonus system:
        # Last 2 years: +3 points (brand new)
        # 3-5 years: +2 points (recent)
        # 6-10 years: +1 point (modern)
        # 11-20 years: 0 points (classic)
        # 20+ years: -1 point (very old, but not penalized too much for cult classics)
        
        if age <= 2:
            return 3
        elif age <= 5:
            return 2
        elif age <= 10:
            return 1
        elif age <= 20:
            return 0
        else:
            return -1
    except (ValueError, IndexError):
        return 0

@tool
def suggest_series(genre: str, mood: str, duration: str,
                   providers_text: str, language: str, country: str) -> str:
    """Return up to 3 series tailored to user slots and country (watchable on owned providers)."""
    logger.info(f"Suggesting series - Genre: {genre}, Mood: {mood}, Duration: {duration}, Providers: {providers_text}, Lang: {language}, Country: {country}")
    
    if not tmdb:
        logger.error("TMDB API key is missing")
        return json.dumps({"error":"Missing TMDB_API_KEY"}, ensure_ascii=False)
    
    region = (country or "US").upper()
    user_prov_norm = normalize_providers(providers_text)
    logger.debug(f"Normalized providers: {user_prov_norm}")
    
    try:
        region_map = tmdb.region_provider_map(region)
        owned_ids = intersect_providers(user_prov_norm, region_map)
        logger.debug(f"Provider IDs to filter: {owned_ids}")
        params = build_discover_params(genre, mood, duration, region, language)
        
        if owned_ids:
            params["with_watch_providers"] = "|".join(str(x) for x in owned_ids)
        
        logger.info(f"Querying TMDB with params: {params}")
        candidates = tmdb.discover_tv(params)[:12]
        logger.info(f"Found {len(candidates)} candidate series")
    except requests.RequestException as e:
        logger.error(f"TMDB API error: {str(e)}")
        return json.dumps({"error": f"TMDB API error: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Unexpected error during search: {str(e)}"}, ensure_ascii=False)
    
    # Build candidate list with genre scoring and freshness bonus
    scored_candidates = []
    for tv in candidates:
        tv_id = tv.get("id")
        if not tv_id:
            continue
        
        try:
            where = tmdb.watch_providers_for(tv_id, region)
        except Exception:
            where = []
        
        if owned_ids and not where:
            continue
        
        # Calculate scores
        tv_genres = tv.get("genre_ids", [])
        first_air = tv.get("first_air_date")
        
        genre_score = _score_genre_match(tv_genres, genre)
        freshness_score = _score_freshness(first_air)
        total_score = genre_score + freshness_score
        
        overview = _trim_overview(tv.get("overview") or "")
        scored_candidates.append({
            "title": tv.get("name"),
            "overview": overview,
            "vote": tv.get("vote_average"),
            "first_air_date": first_air,
            "where": where[:5],
            "genre_score": genre_score,
            "freshness_score": freshness_score,
            "total_score": total_score,
            "genre_ids": tv_genres
        })
    
    # Sort by total score (genre + freshness), then by vote rating
    scored_candidates.sort(key=lambda x: (x["total_score"], x.get("vote", 0)), reverse=True)
    logger.debug(f"Sorted candidates by score: {[(c['title'], c['first_air_date'], 'G:' + str(c['genre_score']) + ' F:' + str(c['freshness_score']) + ' T:' + str(c['total_score'])) for c in scored_candidates[:5]]}")
    
    # Take top 3 and remove scoring metadata
    suggestions = []
    for candidate in scored_candidates[:3]:
        suggestions.append({
            "title": candidate["title"],
            "overview": candidate["overview"],
            "vote": candidate["vote"],
            "first_air_date": candidate["first_air_date"],
            "where": candidate["where"]
        })
    
    logger.info(f"Returning {len(suggestions)} suggestions")
    
    if not suggestions:
        msg = "No titles available on the specified providers." if owned_ids else "No suitable titles found."
        logger.warning(f"No suggestions found: {msg}")
        return json.dumps({"country": region, "suggestions": [], "note": msg}, ensure_ascii=False)
    
    return json.dumps({"country": region, "suggestions": suggestions}, ensure_ascii=False)

# =========================
# LLM + LCEL components
# =========================
llm_kwargs = {
    "openai_api_base": OVH_BASE, 
    "openai_api_key": OVH_KEY, 
    "temperature": 0.0,  # Use 0 for more deterministic extraction
    "model": MODEL or "gpt-oss-120b"  # Use the actual model name available on this endpoint
}

llm_base = ChatOpenAI(**llm_kwargs)

# 1) Chain LCEL: slot extraction/validation (structured output)
collect_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly TV concierge helping users find their next favorite series. "
     "Your job is to understand what they're looking for through natural conversation.\n\n"
     
     "WHAT YOU NEED TO KNOW:\n"
     "- Genre: {genres}\n"
     "- Mood/Tone: {moods}\n"
     "- Episode Duration: {dur}\n"
     "- Language: {langs}\n"
     "- Streaming Services: (any service like Netflix, Prime Video, Disney+, etc.)\n\n"
     
     "WHAT YOU ALREADY KNOW:\n"
     "- Genre: {current_genre}\n"
     "- Mood: {current_mood}\n"
     "- Duration: {current_duration}\n"
     "- Providers: {current_providers}\n"
     "- Language: {current_language}\n\n"
     
     "HOW TO INTERACT:\n"
     "1. **Extract everything** from the user's message in one go - read carefully!\n"
     "2. **Keep what you know** - Don't change fields that are already filled unless the user explicitly wants to change them.\n"
     "3. **Be natural** - If something's missing, ask ONE simple question. Be conversational, friendly, and brief (1-2 sentences max).\n"
     "4. **Don't repeat yourself** - NEVER ask about information you already have (check WHAT YOU ALREADY KNOW above).\n"
     "5. **Validate gently** - If the user says something close but not quite right (like 'action' when 'crime' exists), suggest the valid option naturally.\n"
     "6. **Know when you're done** - Once you have all 5 fields, set ask_next to null.\n\n"
     
     "EXAMPLES OF NATURAL QUESTIONS (do NOT list options!):\n"
     "- Missing genre: \"What kind of genre are you in the mood for?\"\n"
     "- Missing mood: \"How are you feeling today - something light or more intense?\"\n"
     "- Missing duration: \"Do you prefer short episodes or longer ones?\"\n"
     "- Missing providers: \"Which streaming service do you have?\"\n"
     "- Missing language: \"Any preference on the language?\"\n\n"
     
     "RESPONSE FORMAT (valid JSON only):\n"
     '{{\n'
     '  "known": {{\n'
     '    "genre": "extracted value or null",\n'
     '    "mood": "extracted value or null",\n'
     '    "duration": "extracted value or null",\n'
     '    "providers": "extracted value or null",\n'
     '    "language": "extracted value or null"\n'
     '  }},\n'
     '  "ask_next": "your natural question" or null\n'
     '}}\n'),
    ("human", "{user_input}")
])

collect_parser = PydanticOutputParser(pydantic_object=CollectResult)
# Note: We'll dynamically add current preferences when invoking the chain

# Tools-enabled model
assistant_with_tools = llm_base.bind_tools([ipinfo_location, suggest_series])

# 3) LCEL Prompt for final response formatting
final_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a TV concierge. Using the user's preferences and the JSON suggestions provided, "
     "present ALL the series from the JSON (up to 3 maximum). For each series: title (in **bold**), "
     "why it fits their mood/slots, and where to watch it in their country.\n\n"
     "CRITICAL RULES:\n"
     "1. ONLY use the series provided in the 'Suggestions (JSON)' data below\n"
     "2. DO NOT suggest any series from your general knowledge or training data\n"
     "3. If the JSON contains no suggestions or has an error, acknowledge it and DO NOT make up alternatives\n"
     "4. Present EXACTLY ALL series in the JSON - if there are 3, show all 3; if there are 2, show both\n"
     "5. DO NOT pick just one favorite - present them all equally\n\n"
     "EXAMPLE FORMAT (when JSON has 3 series):\n"
     "**Series 1** – Description why it fits. Available on Netflix.\n\n"
     "**Series 2** – Description why it fits. Available on Netflix.\n\n"
     "**Series 3** – Description why it fits. Available on Netflix.\n\n"
     "Style: concise, friendly, helpful. Respond in {language}."),
    ("human", "User preferences: {prefs}\nSuggestions (JSON): {suggestions}\nLocation: {geo}")
])
final_chain = final_prompt | llm_base

# =========================
# Runtime CLI
# =========================
def normalize_user_input(user_input: str) -> str:
    """Normalize user input to help the LLM understand duration and other values."""
    text = user_input.strip()
    
    # Handle negative responses for language/preferences ("no", "nope", "don't care", etc.)
    # When user says these in isolation, treat as "any language" (most common missing field)
    if re.match(r'^(no|nope|nah|don\'t care|doesn\'t matter|whatever|surprise me)$', text, re.IGNORECASE):
        text = "any language"
    
    # Handle explicit "any" -> "any language" (before other "any" replacements)
    if text.lower() == "any":
        text = "any language"
    
    # Normalize duration patterns (avoid double replacement)
    # First handle "X minutes" or "X min"
    text = re.sub(r'(?<![~<>])(\d{2,3})\s*min(utes?)?\b', r'~\1m', text, flags=re.IGNORECASE)
    # Then handle standalone "Xm" (but not if already has ~<>)
    text = re.sub(r'(?<![~<>])(\d{2,3})m\b', r'~\1m', text)
    
    # "short" -> "<30m"
    if re.search(r'\bshort\b', text, re.IGNORECASE):
        text = re.sub(r'\bshort\b', '<30m', text, flags=re.IGNORECASE)
    
    # "long" -> ">60m"
    if re.search(r'\blong\b', text, re.IGNORECASE):
        text = re.sub(r'\blong\b', '>60m', text, flags=re.IGNORECASE)
    
    # "standard" or "medium" -> "~45m"  
    if re.search(r'\b(standard|medium)\b', text, re.IGNORECASE):
        text = re.sub(r'\b(standard|medium)\b', '~45m', text, flags=re.IGNORECASE)
    
    # Handle "any duration" -> "~45m" (after handling "any" alone)
    text = re.sub(r'\bany duration\b', '~45m', text, flags=re.IGNORECASE)
    
    return text

def merge_prefs(base: UserPrefs, new: UserPrefs) -> UserPrefs:
    """Merges new preferences into the base, overwriting existing values."""
    data = base.model_dump()
    for k, v in new.model_dump().items():
        if v is not None:
            data[k] = v
    return UserPrefs(**data)

def prefs_complete(p: UserPrefs) -> bool:
    """Checks if all 5 slots are filled."""
    return all([p.genre, p.mood, p.duration, p.providers, p.language])

def prefs_minimal_complete(p: UserPrefs) -> bool:
    """Checks if we have the bare minimum to search."""
    return (p.genre is not None or p.mood is not None) and p.providers is not None

class ParseError(Exception):
    """LLM failed to parse user input."""
    pass

class APIError(Exception):
    """External API (TMDB, ipinfo) failed."""
    pass

# =========================
# LangGraph Node Functions
# =========================
def should_search(state: ConversationState) -> Literal["search", "collect", "end"]:
    """Router: decide if we should search or keep collecting."""
    user_input = state["user_input"].lower().strip()
    
    # Check for exit commands
    if user_input in {"exit", "quit", "bye", "goodbye"}:
        return "end"
    
    # Check if preferences are complete or user wants to search
    if state["prefs_complete"] or user_input == "search":
        return "search"
    
    return "collect"

def should_continue(state: ConversationState) -> Literal["feedback", "collect", "end"]:
    """Router: after search, go to feedback or continue."""
    user_input = state["user_input"].lower().strip()
    
    if user_input in {"exit", "quit", "bye", "goodbye"}:
        return "end"
    
    if state["waiting_for_feedback"]:
        return "feedback"
    
    return "collect"

def greeting_node(state: ConversationState) -> ConversationState:
    """Initial greeting node."""
    state["next_question"] = (
        "Hi! 👋 What kind of TV series are you looking for?\n"
        "(Tell me about genre, mood, duration, platforms, or language)"
    )
    state["current_state"] = "collecting"
    return state

def collecting_node(state: ConversationState) -> ConversationState:
    """Node for collecting user preferences."""
    logger.info("Collecting preferences node activated")
    # This will be handled by BotSession.extract_preferences()
    state["current_state"] = "collecting"
    return state

def searching_node(state: ConversationState) -> ConversationState:
    """Node for performing TMDB search."""
    logger.info("Searching node activated")
    # This will be handled by BotSession.suggest_series()
    state["current_state"] = "searching"
    state["waiting_for_feedback"] = True
    return state

def feedback_node(state: ConversationState) -> ConversationState:
    """Node for handling feedback."""
    logger.info("Feedback node activated")
    # This will be handled by BotSession.handle_feedback()
    state["current_state"] = "feedback"
    return state

# =========================
# Bot Session Class
# =========================
class BotSession:
    """Manages a single conversation session with state and logic."""
    
    def __init__(self):
        """Initialize a new bot session."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prefs = UserPrefs()
        self.geo_cache = None
        self.conversation_turn = 0
        self.last_ask_next = None
        self.repeat_count = 0
        self.last_suggestions_count = 0
        self.waiting_for_feedback = False
        self.is_force_search = False
        
        self.force_search_keywords = {
            "search now", "that's enough", "go", "find", "search", "go search", 
            "just search", "show me", "let's go", "search please", "skip", "whatever"
        }
        
        self.informational_questions = [
            "what are", "what is", "tell me about", "explain", "how do",
            "how does", "why do", "why does", "who is", "who are"
        ]
        
        self.search_keywords = {
            "netflix", "prime", "disney", "apple tv", "hbo", "hulu", 
            "genre", "mood", "duration", "language", "provider", "platform"
        }
        
        logger.info(f"New session created: {self.session_id}")
    
    def reset(self):
        """Reset the session state for a new search."""
        logger.info("Resetting session state")
        self.prefs = UserPrefs()
        self.geo_cache = None
        self.conversation_turn = 0
        self.repeat_count = 0
        self.last_ask_next = None
        self.waiting_for_feedback = False
    
    def handle_feedback(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        Handle user feedback input.
        Returns (handled, next_question) tuple.
        """
        if not self.waiting_for_feedback:
            return False, None
        
        user_lower = user_input.lower()
        
        # Positive feedback
        if user_lower in {"👍", "thumbs up", "good", "like", "yes", "great", "perfect", "love it"}:
            country = (self.geo_cache or {}).get("country", "US")
            if save_feedback(self.session_id, "positive", self.prefs, country, self.last_suggestions_count):
                print("\nai> 😊 Thanks for the feedback! Glad you liked the suggestions!\n")
            else:
                print("\nai> Thanks for the feedback!\n")
            
            self.waiting_for_feedback = False
            next_q = ("Want to search for something else? You can:\n"
                     "• Refine your search (e.g., 'change genre to comedy')\n"
                     "• Type 'new' to start completely over\n"
                     "• Type 'exit' to quit")
            return True, next_q
        
        # Negative feedback
        elif user_lower in {"👎", "thumbs down", "bad", "dislike", "no", "not good", "poor"}:
            country = (self.geo_cache or {}).get("country", "US")
            print("\nai> I'm sorry these didn't work for you. Could you tell me why? (or type 'skip')")
            comment_in = input("> ").strip()
            comment = comment_in if comment_in.lower() != "skip" else ""
            
            if save_feedback(self.session_id, "negative", self.prefs, country, self.last_suggestions_count, comment):
                print("\nai> 😔 Thanks for the feedback. I'll try to do better next time!\n")
            else:
                print("\nai> Thanks for the feedback!\n")
            
            self.waiting_for_feedback = False
            next_q = ("Want to search for something else? You can:\n"
                     "• Refine your search (e.g., 'change genre to comedy')\n"
                     "• Type 'new' to start completely over\n"
                     "• Type 'exit' to quit")
            return True, next_q
        
        # Skip feedback
        elif user_lower == "skip":
            print("\nai> No problem! Skipping feedback.\n")
            self.waiting_for_feedback = False
            next_q = ("Want to search for something else? You can:\n"
                     "• Refine your search (e.g., 'change genre to comedy')\n"
                     "• Type 'new' to start completely over\n"
                     "• Type 'exit' to quit")
            return True, next_q
        
        # Not feedback, turn off feedback mode
        self.waiting_for_feedback = False
        return False, None
    
    def is_informational_question(self, user_input: str) -> bool:
        """Check if input is an informational question (not about search params)."""
        user_lower = user_input.lower().strip()
        
        # Check for explicit list/options requests
        list_request_patterns = [
            r'\b(which|what)\s+(genre|mood|duration|language|provider|platform)s?\s+(can|should|do|are)',
            r'\bgive\s+me\s+(the\s+)?(list|options)',
            r'\bshow\s+me\s+(the\s+)?(list|options|choices)',
            r'\bwhat\s+(are\s+)?(the\s+)?(available|possible|valid)',
            r'\bi\s+need\s+(the\s+)?list',
            r'\blist\s+of\s+(genre|mood|duration|language|provider)s?',
        ]
        
        for pattern in list_request_patterns:
            if re.search(pattern, user_lower):
                return True
        
        # Check for generic greetings/small talk (questions without search params)
        generic_question_patterns = [
            r'\bhow\s+(are|is)\s+(you|it|things)',  # "how are you?", "how is it?"
            r'\bwhat\'?s\s+up\b',                   # "what's up?"
            r'\bhow\'?s\s+(it\s+)?going\b',         # "how's going?", "how's it going?"
        ]
        
        for pattern in generic_question_patterns:
            if re.search(pattern, user_lower):
                return True
        
        # Original logic for other informational questions
        has_question_mark = user_input.strip().endswith("?")
        starts_with_question = any(user_input.lower().startswith(q) for q in self.informational_questions)
        contains_search_params = any(keyword in user_input.lower() for keyword in self.search_keywords)
        
        return has_question_mark and starts_with_question and not contains_search_params
    
    def answer_informational_question(self, user_input: str) -> bool:
        """
        Answer an informational question.
        Returns True if handled, False otherwise.
        """
        logger.info(f"Detected informational question: {user_input}")
        try:
            question_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are a helpful TV concierge assistant. Answer the user's question clearly and concisely. "
                 "Be friendly and informative. Keep your answer to 2-3 sentences.\n\n"
                 "CRITICAL: Only provide information about the system capabilities and supported options listed below. "
                 "DO NOT suggest specific TV series or make recommendations from your general knowledge. "
                 "If asked about specific series availability, tell the user to provide their preferences so you can search.\n\n"
                 "Context about what we're doing:\n"
                 f"- We're helping find TV series recommendations\n"
                 f"- Current preferences collected: genre={self.prefs.genre}, mood={self.prefs.mood}, duration={self.prefs.duration}, providers={self.prefs.providers}, language={self.prefs.language}\n"
                 f"- Supported platforms: Netflix, Prime Video, Disney+, Apple TV+, HBO Max, Hulu, and many others\n"
                 f"- Supported genres: {', '.join(sorted(ALLOWED_GENRES))}\n"
                 f"- Supported moods: {', '.join(sorted(ALLOWED_MOOD))}\n"
                 f"- Duration options: <30m (short episodes), ~45m (standard), >60m (long episodes)\n"
                 f"- Languages: {', '.join(sorted(ALLOWED_LANG))}"
                ),
                ("human", "{question}")
            ])
            
            answer_chain = question_prompt | llm_base
            
            # Stream the response token by token
            print("ai> ", end="", flush=True)
            answer_text = ""
            for chunk in answer_chain.stream({"question": user_input}):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if isinstance(content, str):
                        answer_text += content
                        print(content, end="", flush=True)
            print("\n")
            
            logger.debug(f"Answer to question: {answer_text}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}", exc_info=True)
            print(f"ai> I'm having trouble understanding. Could you rephrase that?\n")
            return True
    
    def extract_preferences(self, user_input: str) -> Optional[str]:
        """
        Extract preferences from user input.
        Returns next question or None if extraction failed.
        """
        try:
            logger.debug("Invoking slot extraction chain")
            normalized_input = normalize_user_input(user_input)
            if normalized_input != user_input:
                logger.debug(f"Normalized input: '{user_input}' -> '{normalized_input}'")
            
            collect_chain = (
                collect_prompt.partial(
                    genres=", ".join(sorted(ALLOWED_GENRES)),
                    moods=", ".join(sorted(ALLOWED_MOOD)),
                    dur=", ".join(sorted(ALLOWED_DURATION)),
                    langs=", ".join(sorted(ALLOWED_LANG)),
                    current_genre=self.prefs.genre or "null",
                    current_mood=self.prefs.mood or "null",
                    current_duration=self.prefs.duration or "null",
                    current_providers=self.prefs.providers or "null",
                    current_language=self.prefs.language or "null"
                )
                | llm_base
                | collect_parser
            )
            
            collect: CollectResult = collect_chain.invoke({"user_input": normalized_input})
            validated_prefs = validate_prefs(collect.known)
            logger.debug(f"Extracted preferences: {validated_prefs}")
            
            self.prefs = merge_prefs(self.prefs, validated_prefs)
            next_q = None if prefs_complete(self.prefs) else (collect.ask_next or "What other preference can you tell me?")
            
            logger.info(f"Current preferences: genre={self.prefs.genre}, mood={self.prefs.mood}, "
                       f"duration={self.prefs.duration}, providers={self.prefs.providers}, language={self.prefs.language}")
            return next_q
            
        except Exception as e:
            logger.error(f"Failed to parse user input: {e}", exc_info=True)
            print(f"ai> I'm having trouble understanding. Could you rephrase that?\n")
            return "ERROR"  # Return special marker for error
    
    def perform_search(self) -> bool:
        """
        Perform the search and display results.
        Returns True if successful, False otherwise.
        """
        logger.info("Initiating search with complete preferences")
        print("ai> Perfect! Let me find the best matches for you...\n")
        
        try:
            # Get location if not cached
            if self.geo_cache is None:
                logger.info("Fetching geolocation")
                geo_res = ipinfo_location.invoke({})
                geo_data = json.loads(geo_res)
                if "error" in geo_data:
                    logger.warning(f"Geolocation error: {geo_data['error']}")
                    print(f"ai> Warning: {geo_data['error']}. Using default location (US).\n")
                self.geo_cache = geo_data
            
            country = (self.geo_cache or {}).get("country", "US")
            logger.info(f"Using country: {country}")
            
            # Get suggestions
            logger.info("Requesting series suggestions")
            sugg_res = suggest_series.invoke({
                "genre": self.prefs.genre,
                "mood": self.prefs.mood,
                "duration": self.prefs.duration,
                "providers_text": self.prefs.providers,
                "language": self.prefs.language or "en",
                "country": country
            })
            suggestions_data = json.loads(sugg_res)
            
            # Check for errors
            if "error" in suggestions_data:
                logger.error(f"Suggestion error: {suggestions_data['error']}")
                print(f"ai> Sorry, I encountered an issue: {suggestions_data['error']}\n")
                print("ai> Let's try again with different preferences.\n")
                self.prefs = UserPrefs()
                return False
            
            # Format and display results
            response_language = "Italian" if self.prefs.language == "it" else "English"
            logger.info(f"Generating final response in {response_language}")
            
            # Stream the final response token by token
            print("\nai> ", end="", flush=True)
            final_text = ""
            for chunk in final_chain.stream({
                "prefs": json.dumps(self.prefs.model_dump(), ensure_ascii=False),
                "suggestions": json.dumps(suggestions_data, ensure_ascii=False),
                "geo": json.dumps(self.geo_cache, ensure_ascii=False),
                "language": response_language
            }):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if isinstance(content, str):
                        final_text += content
                        print(content, end="", flush=True)
            print("\n")
            
            logger.info("Successfully generated response")
            logger.debug(f"Response length: {len(final_text)} chars")
            
            self.last_suggestions_count = len(suggestions_data.get("suggestions", []))
            return True
            
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            print(f"ai> I'm having trouble connecting to the services right now. Please try again in a moment.\n")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            print(f"ai> I received unexpected data from the service. Let's try that again.\n")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"ai> An unexpected error occurred. Let's start fresh.\n")
            self.prefs = UserPrefs()
            self.geo_cache = None
            return False
    
    def run(self):
        """Run the main conversation loop."""
        print("🍿 SerieBot — Geo-located TV Suggestions (OVH + LangChain + ipinfo + TMDB)")
        print("Type 'exit' to quit, 'new' to start over, or 'search' to search with current preferences.")
        print("💡 Tip: For duration, use '<30m' for short, '~45m' for standard, or '>60m' for long episodes.\n")
        
        ask_next = ("Hi! 👋 What kind of TV series are you looking for?\n"
                   "You can tell me about genre, mood, duration, language, or platforms (e.g., Netflix, Prime)...")
        
        while True:
            self.conversation_turn += 1
            logger.info(f"--- Conversation turn {self.conversation_turn} ---")
            
            # Detect infinite loop
            if ask_next == self.last_ask_next and ask_next is not None:
                self.repeat_count += 1
                logger.warning(f"Same question repeated {self.repeat_count} times")
                if self.repeat_count >= 3:
                    logger.error("Infinite loop detected, forcing search")
                    print("\nai> I notice we're stuck in a loop. Let me search with what we have so far...\n")
                    self.is_force_search = True
                    self.repeat_count = 0
                else:
                    self.is_force_search = False
            else:
                self.repeat_count = 0
                self.is_force_search = False
            
            self.last_ask_next = ask_next
            
            # Get user input
            user_in = input(("you> " if ask_next is None else f"{ask_next}\n> ")).strip()
            logger.info(f"User input: {user_in}")
            
            # Handle feedback if waiting for it
            if self.waiting_for_feedback:
                handled, next_q = self.handle_feedback(user_in)
                if handled:
                    ask_next = next_q
                    continue
            
            # Handle empty input
            if not user_in:
                logger.info("Empty input received")
                print("ai> Please tell me something about what you're looking for, or type 'search' to proceed.\n")
                continue
            
            # Handle exit
            if user_in.lower() in {"exit", "quit"}:
                logger.info("User requested exit")
                break
            
            # Handle new conversation
            if user_in.lower() == "new":
                logger.info("User requested new conversation")
                self.reset()
                ask_next = "Starting fresh! What kind of TV series are you looking for?"
                continue
            
            # Check for force search
            if user_in.lower() in self.force_search_keywords:
                self.is_force_search = True
                logger.info("User forced search")
            
            # Handle informational questions
            if self.is_informational_question(user_in) and not self.is_force_search:
                self.answer_informational_question(user_in)
                continue
            
            # Extract preferences
            if not self.is_force_search:
                ask_next = self.extract_preferences(user_in)
                if ask_next == "ERROR":
                    # Extraction failed, skip to next iteration
                    continue
                # If ask_next is None, preferences are complete, continue to search check
            
            # Check if ready to search
            ready_to_search = prefs_complete(self.prefs) or (self.is_force_search and prefs_minimal_complete(self.prefs))
            
            if not ready_to_search:
                if self.is_force_search and not prefs_minimal_complete(self.prefs):
                    print("ai> I need a bit more info to search. Please tell me at least a (genre or mood) AND a provider.\n")
                    print(f"   📝 Available genres: {', '.join(sorted(ALLOWED_GENRES))}")
                    print(f"   🎭 Available moods: {', '.join(sorted(ALLOWED_MOOD))}")
                    print(f"   ⏱️  Duration options: <30m (short), ~45m (standard), >60m (long)")
                    print()
                else:
                    missing = [k for k,v in self.prefs.model_dump().items() if v is None]
                    if missing:
                        logger.debug(f"Still missing: {missing}")
                        if 'duration' in missing and self.repeat_count > 0:
                            print(f"ai> For duration, please choose one of these: <30m, ~45m, or >60m\n")
                        else:
                            print(f"ai> Got it! Still need: {', '.join(missing)}.\n")
                continue
            
            # Perform search
            if self.perform_search():
                self.waiting_for_feedback = True
                ask_next = ("How do these look? Please rate:\n"
                           "• 👍 (or type 'good', 'like', 'yes') if you liked them\n"
                           "• 👎 (or type 'bad', 'dislike', 'no') if you didn't\n"
                           "• 'skip' to skip feedback\n"
                           "Or refine your search (e.g., 'change genre to comedy', 'only Netflix')\n"
                           "Type 'new' to start over or 'exit' to quit")
            else:
                ask_next = "What would you like to search for?"

def build_conversation_graph():
    """
    Build the LangGraph conversation flow.
    
    States:
    - greeting: Initial welcome
    - collecting: Collecting user preferences
    - searching: Performing TMDB search
    - feedback: Collecting user feedback
    - end: Termination
    """
    logger.info("Building LangGraph conversation flow")
    
    # Create the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("collecting", collecting_node)
    workflow.add_node("searching", searching_node)
    workflow.add_node("feedback", feedback_node)
    
    # Add edges from START
    workflow.add_edge(START, "greeting")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "greeting",
        lambda state: "collecting",
        {"collecting": "collecting"}
    )
    
    workflow.add_conditional_edges(
        "collecting",
        should_search,
        {
            "search": "searching",
            "collect": "collecting",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "searching",
        should_continue,
        {
            "feedback": "feedback",
            "collect": "collecting",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "feedback",
        should_continue,
        {
            "feedback": "feedback",
            "collect": "collecting",
            "end": END
        }
    )
    
    # Compile the graph with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info("LangGraph conversation flow built successfully")
    return app

def run_cli():
    """Main entry point for the CLI application."""
    logger.info("Starting CLI interface")
    session = BotSession()
    session.run()

if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        print("\n\nGoodbye! 👋")
    except Exception as e:
        logger.critical(f"Unexpected fatal error: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}")
        print("Please restart the application.")
    finally:
        logger.info("SerieBot session ended")
        logger.info("=" * 80)