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
from urllib.parse import urlencode
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")

# =========================
# Logging Configuration
# =========================
# Load DEBUG setting before setup_logging() is called
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

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
    
    # Console handler - only errors (not warnings)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
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
# API Configuration
TMDB_KEY = os.getenv("TMDB_API_KEY")
OVH_BASE = os.getenv("OPENAI_API_BASE", "https://api.ai.ovh.net/v1")
OVH_KEY  = os.getenv("OPENAI_API_KEY")
MODEL    = os.getenv("OVH_MODEL") or None  # None allows the API to use its default model
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

# Algorithm Constants
MAX_CANDIDATES = 12  # Maximum candidate shows to evaluate
TOP_SUGGESTIONS = 3  # Number of top suggestions to return
MAX_REPEATS = 3      # Max times to ask for same missing info before showing examples
TMDB_RATE_LIMIT = 0.25  # Seconds between TMDB API calls

# Temperature settings for different LLM purposes
TEMP_EXTRACT = 0.0   # Deterministic extraction
TEMP_CREATIVE = 0.7  # Natural response generation
TEMP_CLASSIFY = 0.3  # Balanced classification

# Initialize logging (DEBUG already defined above)
logger = setup_logging()
logger.info(f"Configuration loaded - TMDB: {'✓' if TMDB_KEY else '✗'}, OVH: {'✓' if OVH_KEY else '✗'}")
logger.info(f"Model: {MODEL or 'gpt-oss-120b'}, Base URL: {OVH_BASE}")

if not OVH_KEY:
    logger.error("Missing OPENAI_API_KEY environment variable")
    raise SystemExit("❌ Error: Missing OPENAI_API_KEY (OVH AI Endpoints). Please set it in your .env file.")

if not TMDB_KEY:
    logger.error("Missing TMDB_API_KEY environment variable")
    raise SystemExit("❌ Error: Missing TMDB_API_KEY. Please set it in your .env file or get one from https://www.themoviedb.org/settings/api")

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
                'providers', 'language', 'country', 
                'suggestions_count', 'comment'
            ])
        logger.info(f"Created feedback file: {FEEDBACK_FILE}")

def save_feedback(session_id: str, rating: str, prefs: 'UserPrefs', 
                 country: str, suggestions_count: int, comment: str = ""):
    """Save user feedback to CSV file."""
    try:
        # Convert genre list to comma-separated string for CSV
        genre_str = ", ".join(prefs.genre) if prefs.genre else "N/A"
        
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                session_id,
                rating,
                genre_str,
                prefs.mood or "N/A",
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
    """
    Normalize provider names using LLM for intelligent matching.
    Falls back to regex-based matching if LLM fails.
    """
    if not user_text:
        return []
    
    # Try LLM-based normalization first
    try:
        normalized = normalize_providers_llm(user_text)
        if normalized:
            logger.debug(f"LLM normalized providers: '{user_text}' -> {normalized}")
            return normalized
    except Exception as e:
        logger.warning(f"LLM provider normalization failed: {e}, falling back to regex")
    
    # Fallback to regex-based matching
    tokens = [t.strip() for t in re.split(r"[,\;]| and |\s{2,}", user_text, flags=re.I) if t.strip()]
    out = set()
    for t in tokens:
        st = _slug(t)
        for canonical, syns in PROVIDER_SYNONYMS.items():
            if st == _slug(canonical) or st in [_slug(x) for x in syns]:
                out.add(canonical)
    result = list(out)
    logger.debug(f"Regex normalized providers: '{user_text}' -> {result}")
    return result

def normalize_providers_llm(user_text: str) -> List[str]:
    """
    Use LLM to intelligently normalize provider names.
    Handles typos, abbreviations, and informal names.
    """
    if not user_text:
        return []
    
    # Get list of canonical provider names
    canonical_providers = list(PROVIDER_SYNONYMS.keys())
    
    provider_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a streaming service name normalizer. Your job is to identify streaming service names "
         "from user input and map them to their canonical names.\n\n"
         "CANONICAL PROVIDER NAMES:\n"
         f"{', '.join(canonical_providers)}\n\n"
         "INSTRUCTIONS:\n"
         "1. Extract all streaming service mentions from the user's text\n"
         "2. Handle typos (e.g., 'Netfix' → 'Netflix', 'Primevideo' → 'Prime Video')\n"
         "3. Handle abbreviations (e.g., 'D+' → 'Disney+', 'HBO' → 'HBO Max')\n"
         "4. Handle informal names (e.g., 'Prime' → 'Prime Video', 'Disney' → 'Disney+')\n"
         "5. Return ONLY the canonical names from the list above\n"
         "6. If no valid providers found, return empty list\n\n"
         "RESPONSE FORMAT (valid JSON array only):\n"
         '["Provider1", "Provider2"]\n\n'
         "EXAMPLES:\n"
         "Input: 'netfix and hulu' → [\"Netflix\", \"Hulu\"]\n"
         "Input: 'prime video' → [\"Prime Video\"]\n"
         "Input: 'D+ and apple tv' → [\"Disney+\", \"Apple TV+\"]\n"
         "Input: 'nflx' → [\"Netflix\"]\n"
         "Input: 'no preference' → []\n"
        ),
        ("human", "{user_text}")
    ])
    
    try:
        provider_chain = provider_prompt | llm_classify  # Use classifier LLM
        response = provider_chain.invoke({"user_text": user_text})
        
        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)
        logger.debug(f"LLM provider response: {content}")
        
        # Parse JSON - ensure content is a string
        content_str = content if isinstance(content, str) else str(content)
        providers = json.loads(content_str.strip())
        
        # Validate that all returned providers are in canonical list
        valid_providers = [p for p in providers if p in canonical_providers]
        
        return valid_providers
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.error(f"Failed to parse LLM provider response: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in provider normalization: {e}")
        return []

def validate_mood_llm(user_mood: str) -> Optional[str]:
    """
    Use LLM to intelligently map user's mood description to canonical mood.
    Handles synonyms, similar concepts, and informal descriptions.
    """
    if not user_mood:
        return None
    
    mood_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a mood classifier for TV series recommendations. Your job is to map user's mood "
         "descriptions to one of the canonical mood categories.\n\n"
         "CANONICAL MOODS:\n"
         f"{', '.join(sorted(ALLOWED_MOOD))}\n\n"
         "MOOD DESCRIPTIONS:\n"
         "- light-hearted: fun, funny, cheerful, uplifting, comedy, humorous, lighthearted\n"
         "- intense: serious, dark, heavy, dramatic, gripping, suspenseful\n"
         "- mind-blowing: mind blowing, thought-provoking, complex, philosophical, cerebral, intelligent\n"
         "- comforting: cozy, feel-good, relaxing, wholesome, heartwarming, warm\n"
         "- adrenaline-fueled: action-packed, thrilling, exciting, fast-paced, edge-of-your-seat\n\n"
         "INSTRUCTIONS:\n"
         "1. Analyze the user's mood description\n"
         "2. Map it to the MOST SIMILAR canonical mood from the list above\n"
         "3. If the description doesn't clearly match any mood, return null\n"
         "4. Return ONLY the exact canonical mood name or null\n\n"
         "RESPONSE FORMAT (valid JSON string only):\n"
         '"canonical-mood" or null\n\n'
         "EXAMPLES:\n"
         "Input: 'relaxing' → \"comforting\"\n"
         "Input: 'funny' → \"light-hearted\"\n"
         "Input: 'action packed' → \"adrenaline-fueled\"\n"
         "Input: 'deep and complex' → \"mind-blowing\"\n"
         "Input: 'dark and gritty' → \"intense\"\n"
         "Input: 'pizza' → null\n"
        ),
        ("human", "{user_mood}")
    ])
    
    try:
        mood_chain = mood_prompt | llm_classify  # Use classifier LLM
        response = mood_chain.invoke({"user_mood": user_mood})
        
        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)
        logger.debug(f"LLM mood response: {content}")
        
        # Parse JSON - ensure content is a string
        content_str = content if isinstance(content, str) else str(content)
        mood = json.loads(content_str.strip())
        
        # Validate that returned mood is in canonical list
        if mood and mood in ALLOWED_MOOD:
            logger.info(f"LLM mapped mood: '{user_mood}' -> '{mood}'")
            return mood
        
        return None
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.error(f"Failed to parse LLM mood response: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in mood validation: {e}")
        return None

def validate_language_llm(user_language: str) -> Optional[str]:
    """
    Use LLM to intelligently map user's language preference to canonical language.
    Handles variations like 'any will be fine', 'doesn't matter', 'no preference' → 'any'.
    """
    if not user_language:
        return None
    
    language_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a language preference classifier for TV series recommendations. Your job is to map "
         "user's language descriptions to canonical language codes.\n\n"
         "CANONICAL LANGUAGES:\n"
         f"{', '.join(sorted(ALLOWED_LANG))}\n\n"
         "LANGUAGE MAPPINGS:\n"
         "- 'any', 'any language', 'doesn't matter', 'no preference', 'whatever', 'any will/would be fine/good/ok' → 'any'\n"
         "- 'english', 'eng', 'en' → 'en'\n"
         "- 'italian', 'italiano', 'ita', 'it' → 'it'\n\n"
         "INSTRUCTIONS:\n"
         "1. Analyze the user's language input\n"
         "2. Map it to the canonical language code\n"
         "3. If uncertain, return 'any' as default\n"
         "4. Return ONLY the exact canonical code\n\n"
         "RESPONSE FORMAT (valid JSON string only):\n"
         '"en" or "it" or "any"\n\n'
         "EXAMPLES:\n"
         "Input: 'English' → \"en\"\n"
         "Input: 'italiano' → \"it\"\n"
         "Input: 'any will be fine' → \"any\"\n"
         "Input: 'doesn\\'t matter' → \"any\"\n"
         "Input: 'no preference' → \"any\"\n"
        ),
        ("human", "{user_language}")
    ])
    
    try:
        language_chain = language_prompt | llm_classify  # Use classifier LLM
        response = language_chain.invoke({"user_language": user_language})
        
        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)
        logger.debug(f"LLM language response: {content}")
        
        # Parse JSON - ensure content is a string
        content_str = content if isinstance(content, str) else str(content)
        language = json.loads(content_str.strip())
        
        # Validate that returned language is in canonical list
        if language and language in ALLOWED_LANG:
            logger.info(f"LLM mapped language: '{user_language}' -> '{language}'")
            return language
        
        return None
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.error(f"Failed to parse LLM language response: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in language validation: {e}")
        return None

def validate_genre_llm(user_genres: List[str]) -> List[str]:
    """
    Use LLM to intelligently map user's genre descriptions to canonical genres.
    Handles synonyms, typos, and related concepts.
    """
    if not user_genres:
        return []
    
    # Get allowed genres from TMDB
    allowed_genres = get_allowed_genres()
    
    genre_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a genre classifier for TV series. Your job is to map user's genre descriptions "
         "to canonical TMDB genre categories.\n\n"
         "CANONICAL GENRES:\n"
         f"{', '.join(sorted(allowed_genres))}\n\n"
         "COMMON MAPPINGS:\n"
         "- 'horror' → ['mystery', 'thriller'] (horror is a movie genre, closest TV equivalents)\n"
         "- 'thriller' → ['mystery', 'crime']\n"
         "- 'romantic' → ['drama']\n"
         "- 'sitcom' → ['comedy']\n"
         "- 'space' → ['sci-fi']\n"
         "- 'supernatural' → ['sci-fi & fantasy', 'mystery']\n"
         "- 'true crime' → ['crime', 'documentary']\n\n"
         "INSTRUCTIONS:\n"
         "1. Map each user genre to ONE OR MORE canonical genres\n"
         "2. Handle typos (e.g., 'sifi' → 'sci-fi')\n"
         "3. Handle synonyms (e.g., 'funny' → 'comedy')\n"
         "4. Return ONLY canonical genres from the list above\n"
         "5. If a genre has no clear mapping, exclude it\n\n"
         "RESPONSE FORMAT (valid JSON array only):\n"
         '["genre1", "genre2"]\n\n'
         "EXAMPLES:\n"
         "Input: ['horror', 'suspense'] → [\"mystery\", \"crime\"]\n"
         "Input: ['funny'] → [\"comedy\"]\n"
         "Input: ['space opera'] → [\"sci-fi\", \"action & adventure\"]\n"
         "Input: ['sifi'] → [\"sci-fi\"]\n"
         "Input: ['romantic drama'] → [\"drama\"]\n"
        ),
        ("human", "{user_genres}")
    ])
    
    try:
        genre_chain = genre_prompt | llm_classify  # Use classifier LLM
        response = genre_chain.invoke({"user_genres": json.dumps(user_genres)})
        
        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)
        logger.debug(f"LLM genre response: {content}")
        
        # Parse JSON - ensure content is a string
        content_str = content if isinstance(content, str) else str(content)
        genres = json.loads(content_str.strip())
        
        # Validate that all returned genres are in canonical list
        valid_genres = [g.lower() for g in genres if g.lower() in allowed_genres]
        
        if valid_genres:
            logger.info(f"LLM mapped genres: {user_genres} -> {valid_genres}")
        
        return valid_genres
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.error(f"Failed to parse LLM genre response: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in genre validation: {e}")
        return []

def detect_force_search_intent(user_input: str, current_prefs: dict) -> bool:
    """
    Use LLM to detect if user wants to skip remaining questions and search immediately.
    This is more intelligent than keyword matching.
    
    Args:
        user_input: User's message
        current_prefs: Current preferences collected (dict with genre, mood, providers, language)
        
    Returns:
        bool: True if user wants to force search now
    """
    # Quick checks first
    if not user_input or len(user_input.strip()) < 3:
        return False
    
    # Count how many prefs are filled
    filled_count = sum(1 for v in current_prefs.values() if v is not None)
    
    # If nothing is filled, definitely not a force search
    if filled_count == 0:
        return False
    
    # Create prompt
    force_search_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are analyzing user intent in a TV series recommendation conversation. "
         "The bot is collecting preferences (genre, mood, platform, language) before searching.\n\n"
         "Your job: Determine if the user wants to SKIP remaining questions and search NOW.\n\n"
         "FORCE SEARCH indicators:\n"
         "- Explicit: 'search now', 'just search', 'that's enough', 'go ahead', 'find something'\n"
         "- Impatient: 'I don't care about the rest', 'whatever', 'just show me', 'skip the questions'\n"
         "- Satisfied: 'I'm good', 'that's all I need', 'this is enough'\n\n"
         "NOT force search (normal conversation):\n"
         "- Answering a question: 'I'll go with comedy', 'I prefer Netflix', 'English please'\n"
         "- Casual language: 'let me think', 'hmm, maybe action', 'I like drama'\n"
         "- Questions: 'what about...', 'do you have...', 'can you suggest...'\n\n"
         f"Context: User has provided {filled_count}/4 preferences so far.\n\n"
         "Respond with ONLY 'true' or 'false'."),
        ("user", "{user_input}")
    ])
    
    try:
        force_chain = force_search_prompt | llm_classify
        response = force_chain.invoke({"user_input": user_input})
        
        content = response.content if hasattr(response, 'content') else str(response)
        content_str = content if isinstance(content, str) else str(content)
        result = content_str.strip().lower() == 'true'
        
        if result:
            logger.info(f"Force search intent detected in: '{user_input}'")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to detect force search intent: {e}")
        # Fallback to simple keyword check
        simple_keywords = {"search now", "just search", "that's enough", "skip"}
        return any(kw in user_input.lower() for kw in simple_keywords)

# =========================
# Pydantic models (typed I/O)
# =========================
# Note: ALLOWED_GENRES will be populated dynamically from TMDB at runtime
# This is just a fallback for validation
ALLOWED_GENRES = {"sci-fi","crime","drama","comedy","animation","fantasy","mystery","documentary"}
ALLOWED_MOOD = {"light-hearted","intense","mind-blowing","comforting","adrenaline-fueled"}
ALLOWED_LANG = {"it","en","any"}

def get_allowed_genres() -> set:
    """
    Get the current set of allowed genres from TMDB.
    Falls back to static set if TMDB unavailable.
    """
    try:
        available = get_available_genres()
        return set(available) if available else ALLOWED_GENRES
    except Exception:
        return ALLOWED_GENRES

class UserPrefs(BaseModel):
    """Holds all user preferences."""
    genre: Optional[List[str]] = None  # Changed to list to support multiple genres
    mood: Optional[str] = None
    providers: Optional[str] = None
    language: Optional[str] = None
    
    @field_validator('genre', mode='before')
    @classmethod
    def convert_genre_to_list(cls, v):
        """Convert genre to list if it's a string, or ensure it's a list."""
        if v is None:
            return None
        if isinstance(v, str):
            # Split by comma or "and" to handle multiple genres
            genres = re.split(r'[,&]|\s+and\s+', v, flags=re.IGNORECASE)
            return [g.strip().lower() for g in genres if g.strip()]
        if isinstance(v, list):
            return [str(item).strip().lower() for item in v if str(item).strip()]
        return None
    
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

# =========================
# LangGraph State Definition
# =========================
class ConversationState(TypedDict):
    """State for LangGraph conversation flow."""
    # Bot instance for accessing methods
    bot: 'BotSession'
    
    # User preferences being collected
    prefs: UserPrefs
    
    # Conversation metadata
    session_id: str
    user_input: str
    next_question: Optional[str]
    last_ask_next: Optional[str]
    repeat_count: int
    conversation_turn: int
    
    # Geolocation cache
    geo_cache: Optional[Dict]
    
    # Suggestions tracking
    last_suggestions_count: int
    last_suggestions: Optional[List[Dict]]
    
    # Flow control
    waiting_for_feedback: bool
    prefs_complete: bool
    is_force_search: bool
    
    # Current state
    current_state: Literal["greeting", "collecting", "searching", "feedback", "end"]

# =========================
# Validation
# =========================
def validate_prefs(prefs: UserPrefs) -> Tuple[UserPrefs, Dict]:
    """
    Post-process to ensure only valid values make it through.
    Returns (validated_prefs, mapping_info) where mapping_info contains
    any LLM mappings that were performed.
    """
    validated_data = {}
    mapping_info = {}
    
    # Validate genres - try exact match first, then LLM for fuzzy matching
    if prefs.genre:
        allowed_genres = get_allowed_genres()
        validated_genres = []
        unmatched_genres = []
        
        for genre in prefs.genre:
            genre_lower = genre.lower()
            if genre_lower in allowed_genres:
                validated_genres.append(genre_lower)
            else:
                unmatched_genres.append(genre)
        
        # Try LLM for unmatched genres
        if unmatched_genres:
            try:
                llm_genres = validate_genre_llm(unmatched_genres)
                if llm_genres:
                    validated_genres.extend(llm_genres)
                    # Store mapping info for user feedback
                    mapping_info['genre'] = {
                        'original': unmatched_genres,
                        'mapped': llm_genres
                    }
            except Exception as e:
                logger.warning(f"LLM genre validation failed: {e}")
        
        validated_data['genre'] = validated_genres if validated_genres else None
    else:
        validated_data['genre'] = None
    
    # Validate mood - try LLM first, fallback to exact match
    if prefs.mood:
        mood_lower = prefs.mood.lower()
        if mood_lower in ALLOWED_MOOD:
            validated_data['mood'] = mood_lower
        else:
            # Try LLM to map similar moods
            try:
                llm_mood = validate_mood_llm(prefs.mood)
                if llm_mood:
                    validated_data['mood'] = llm_mood
                    # Store mapping info for user feedback
                    mapping_info['mood'] = {
                        'original': prefs.mood,
                        'mapped': llm_mood
                    }
                else:
                    validated_data['mood'] = None
            except Exception as e:
                logger.warning(f"LLM mood validation failed: {e}")
                validated_data['mood'] = None
    else:
        validated_data['mood'] = None
    
    # Validate language - try LLM first for smart mapping
    if prefs.language:
        lang_lower = prefs.language.lower().strip()
        
        # Direct match
        if lang_lower in ALLOWED_LANG:
            validated_data['language'] = lang_lower
        else:
            # Try LLM to map variations like "any will be fine", "doesn't matter"
            try:
                llm_language = validate_language_llm(prefs.language)
                if llm_language:
                    validated_data['language'] = llm_language
                    # Store mapping info for user feedback
                    mapping_info['language'] = {
                        'original': prefs.language,
                        'mapped': llm_language
                    }
                else:
                    validated_data['language'] = None
            except Exception as e:
                logger.warning(f"LLM language validation failed: {e}")
                validated_data['language'] = None
    else:
        validated_data['language'] = None
    
    # Providers - just pass through (will be normalized later)
    validated_data['providers'] = prefs.providers
    
    return (UserPrefs(**validated_data), mapping_info)

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
    RATE_LIMIT_DELAY = TMDB_RATE_LIMIT  # Use constant from config
    
    def __init__(self, api_key: str):
        self.key = api_key
        self._providers_cache: Dict[str, Dict[str,int]] = {}
        self._genres_cache: Optional[Dict[str, int]] = None  # Cache for genre name -> ID mapping
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def get_tv_genres(self) -> Dict[str, int]:
        """
        Fetch TV genre list from TMDB API.
        Returns a dict mapping genre name (lowercase) to genre ID.
        Caches the result for the session.
        """
        if self._genres_cache is not None:
            logger.debug("Using cached TV genres")
            return self._genres_cache
        
        logger.info("Fetching TV genres from TMDB")
        self._rate_limit()
        url = f"{self.BASE}/genre/tv/list"
        params = {"api_key": self.key, "language": "en-US"}
        
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            genres = r.json().get("genres", [])
            
            # Build mapping: lowercase name -> ID
            genre_map = {}
            for genre in genres:
                name = genre.get("name", "").lower()
                genre_id = genre.get("id")
                if name and genre_id:
                    genre_map[name] = genre_id
            
            # Add common aliases for better UX
            sci_fi_fantasy_id = None
            action_adventure_id = None
            
            # Find IDs for aliasing
            for name, gid in genre_map.items():
                if "sci-fi" in name and "fantasy" in name:
                    sci_fi_fantasy_id = gid
                if "action" in name and "adventure" in name:
                    action_adventure_id = gid
            
            # Add sci-fi aliases (all point to "Sci-Fi & Fantasy" genre)
            if sci_fi_fantasy_id:
                genre_map["sci-fi"] = sci_fi_fantasy_id
                genre_map["scifi"] = sci_fi_fantasy_id
                genre_map["science fiction"] = sci_fi_fantasy_id
                genre_map["fantasy"] = sci_fi_fantasy_id
            
            # Add action/adventure aliases
            if action_adventure_id:
                genre_map["action"] = action_adventure_id
                genre_map["adventure"] = action_adventure_id
            
            self._genres_cache = genre_map
            logger.info(f"Loaded {len(genre_map)} TV genres (including aliases)")
            logger.debug(f"Available genres: {', '.join(sorted(genre_map.keys()))}")
            
            return genre_map
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch TV genres: {e}")
            # Fallback to basic genre map if API fails
            logger.warning("Using fallback genre map")
            fallback = {
                "sci-fi": 10765,
                "comedy": 35,
                "drama": 18,
                "crime": 80,
                "animation": 16,
                "mystery": 9648,
                "documentary": 99,
                "fantasy": 10765,
            }
            self._genres_cache = fallback
            return fallback
    
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
        
        # Log the full request URL (without API key for security)
        debug_params = {k:v for k,v in p.items() if k != 'api_key'}
        logger.debug(f"TMDB request URL: {url}?{urlencode(debug_params)}")
        
        r = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        
        # Log the response for debugging
        response_data = r.json()
        results = response_data.get("results", [])
        total_results = response_data.get("total_results", 0)
        total_pages = response_data.get("total_pages", 0)
        
        logger.info(f"TMDB discover_tv response: {len(results)} results on page, total_results={total_results}, total_pages={total_pages}")
        if DEBUG:
            logger.debug(f"TMDB full response: {response_data}")
        
        return results
    
    def watch_providers_for(self, tv_id: int, region: str) -> List[str]:
        self._rate_limit()
        url = f"{self.BASE}/tv/{tv_id}/watch/providers"
        p = {"api_key": self.key}
        r = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        res = (r.json().get("results") or {}).get(region.upper(), {})
        flatrate = res.get("flatrate", []) or []
        return [x.get("provider_name") for x in flatrate if x.get("provider_name")]
    
    def search_tv_show(self, query: str) -> List[Dict]:
        """
        Search for TV shows by name.
        Returns a list of matching shows with their details.
        """
        self._rate_limit()
        url = f"{self.BASE}/search/tv"
        p = {"api_key": self.key, "query": query, "language": "en-US"}
        
        try:
            r = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.RequestException as e:
            logger.error(f"Failed to search for TV show '{query}': {e}")
            return []
    
    def get_tv_details(self, tv_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific TV show.
        """
        self._rate_limit()
        url = f"{self.BASE}/tv/{tv_id}"
        p = {"api_key": self.key, "language": "en-US"}
        
        try:
            r = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get TV show details for ID {tv_id}: {e}")
            return None

tmdb = TMDB(TMDB_KEY) if TMDB_KEY else None

def get_genre_id(genre_name: str) -> Optional[int]:
    """
    Get TMDB genre ID from genre name.
    Returns None if genre not found or TMDB not available.
    """
    if not tmdb:
        return None
    
    genres = tmdb.get_tv_genres()
    return genres.get((genre_name or "").lower())

def get_genre_ids(genre_names: Optional[List[str]]) -> Optional[str]:
    """
    Get comma-separated TMDB genre IDs from list of genre names.
    Returns None if no genres found or TMDB not available.
    """
    if not tmdb or not genre_names:
        return None
    
    genre_map = tmdb.get_tv_genres()
    genre_ids = []
    
    for genre_name in genre_names:
        genre_id = genre_map.get(genre_name.lower())
        if genre_id:
            genre_ids.append(str(genre_id))
    
    return ",".join(genre_ids) if genre_ids else None

def get_available_genres() -> List[str]:
    """
    Get list of available genre names from TMDB.
    Returns ALL genre names including aliases so users can use any variant.
    Returns fallback list if TMDB not available.
    """
    if not tmdb:
        return ["sci-fi", "scifi", "crime", "drama", "comedy", "animation", "fantasy", "mystery", "documentary"]
    
    genres = tmdb.get_tv_genres()
    # Return ALL genre names (including aliases) for better UX
    # Users can say "sci-fi", "scifi", "fantasy" - all are valid
    return sorted(genres.keys())

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

def get_mood_keywords(mood: str) -> Tuple[List[str], List[str]]:
    """
    Return positive and negative keywords for semantic matching in series overviews.
    
    Args:
        mood: User's mood preference
        
    Returns:
        Tuple of (positive_keywords, negative_keywords)
    """
    mood = (mood or "").lower()
    positive_keywords = []
    negative_keywords = []
    
    if "light" in mood or "comfort" in mood:
        positive_keywords = ["fun", "heartwarming", "comedy", "lighthearted", "family", 
                           "feel-good", "charming", "delightful", "wholesome", "uplifting"]
        negative_keywords = ["dark", "violent", "disturbing", "intense", "gritty", "brutal"]
    elif "intens" in mood:
        positive_keywords = ["intense", "dark", "suspense", "thriller", "dramatic", "gritty",
                           "complex", "psychological", "tension", "gripping", "powerful"]
        negative_keywords = ["comedy", "lighthearted", "fun", "family"]
    elif "mind" in mood:
        positive_keywords = ["mind-bending", "mystery", "complex", "intricate", "puzzle",
                           "philosophical", "thought-provoking", "cerebral", "enigmatic", "twist"]
        negative_keywords = ["straightforward", "simple", "predictable"]
    elif "adren" in mood:
        positive_keywords = ["action", "thrilling", "fast-paced", "explosive", "adrenaline",
                           "chase", "intense", "exciting", "dynamic", "edge"]
        negative_keywords = ["slow", "quiet", "contemplative", "peaceful"]
    
    return positive_keywords, negative_keywords

def score_mood_match_semantic(overview: str, mood: str) -> float:
    """
    Score how well a series' overview matches the mood using semantic keyword matching.
    
    Args:
        overview: The series overview/description
        mood: User's mood preference
        
    Returns:
        float: Score between 0.0 and 1.0 indicating mood match quality
    """
    if not overview or not mood:
        return 0.5  # Neutral score if no overview or mood
    
    overview_lower = overview.lower()
    positive_keywords, negative_keywords = get_mood_keywords(mood)
    
    # Count keyword matches
    positive_matches = sum(1 for keyword in positive_keywords if keyword in overview_lower)
    negative_matches = sum(1 for keyword in negative_keywords if keyword in overview_lower)
    
    # Calculate score: positive matches boost, negative matches reduce
    # Normalize by the number of keywords we're checking
    if len(positive_keywords) > 0:
        positive_score = min(positive_matches / len(positive_keywords), 1.0)
    else:
        positive_score = 0.5
    
    if len(negative_keywords) > 0:
        negative_penalty = min(negative_matches / len(negative_keywords), 0.5)
    else:
        negative_penalty = 0.0
    
    # Final score: start at 0.5 (neutral), add positive, subtract negative
    final_score = 0.5 + (positive_score * 0.3) - (negative_penalty * 0.2)
    
    # Clamp between 0.0 and 1.0
    return max(0.0, min(1.0, final_score))

def apply_mood_to_score(base_score: float, tv_genres: List[int], overview: str, mood: Optional[str]) -> float:
    """
    Apply mood-based adjustments to the base score (Opzione A + C combined).
    
    Args:
        base_score: The base score before mood adjustment
        tv_genres: List of genre IDs for the series
        overview: Series overview/description for semantic matching
        mood: User's mood preference (optional)
        
    Returns:
        float: Adjusted score after mood considerations
    """
    if not mood:
        return base_score
    
    mood_lower = mood.lower()
    genre_multiplier = 1.0
    
    # Opzione A: Genre-based mood matching
    if "intens" in mood_lower:
        # Boost drama (18) and crime (80)
        if 18 in tv_genres or 80 in tv_genres:
            genre_multiplier = 1.15
    elif "light" in mood_lower or "comfort" in mood_lower:
        # Boost comedy (35) and animation (16)
        if 35 in tv_genres or 16 in tv_genres:
            genre_multiplier = 1.15
        # Reduce crime (80)
        if 80 in tv_genres:
            genre_multiplier = 0.85
    elif "mind" in mood_lower:
        # Boost sci-fi&fantasy (10765), mystery (9648)
        if 10765 in tv_genres or 9648 in tv_genres:
            genre_multiplier = 1.15
    elif "adren" in mood_lower:
        # Boost action & adventure (10759), crime (80)
        if 10759 in tv_genres or 80 in tv_genres:
            genre_multiplier = 1.15
    
    # Opzione C: Semantic matching in overview
    semantic_score = score_mood_match_semantic(overview, mood)
    
    # Combine genre multiplier and semantic score
    # Genre multiplier: up to ±15%
    # Semantic score: up to ±20% (0.8 to 1.2 range)
    semantic_multiplier = 0.8 + (semantic_score * 0.4)
    
    # Apply both adjustments
    adjusted_score = base_score * genre_multiplier * semantic_multiplier
    
    logger.debug(f"Mood adjustment - Genre mult: {genre_multiplier:.2f}, Semantic: {semantic_score:.2f} (mult: {semantic_multiplier:.2f})")
    
    return adjusted_score

def build_discover_params(genres, mood, region, language):
    p = {
        "watch_region": region,
        "sort_by": "popularity.desc",
        "with_watch_monetization_types": "flatrate",
        "with_original_language": None if (language == "any") else (language or None),
        "page": 1
    }
    
    # Get genre IDs from TMDB (supports multiple genres now)
    genre_ids_str = get_genre_ids(genres) if genres else None
    if genre_ids_str:
        p["with_genres"] = genre_ids_str
    
    prefer, avoid = mood_bias(mood)
    
    # If genres are specified, use mood only for avoid list
    # If no genres, use mood preferences for genre selection
    if not genre_ids_str and prefer:
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

def _score_genre_match(tv_genres, requested_genres):
    """
    Score how well a series matches the requested genres.
    For sci-fi requests, prefer series with more 'sci-fi' indicators and fewer fantasy/drama indicators.
    
    Args:
        tv_genres: List of genre IDs for the TV show
        requested_genres: List of genre names requested by user (e.g., ["sci-fi", "mystery"])
    
    Returns:
        int: Score indicating genre match quality
    """
    if not requested_genres:
        return 0
    
    # Map genre names to TMDB IDs
    genre_map = {
        "action": 10759,
        "action & adventure": 10759,
        "adventure": 10759,
        "animation": 16,
        "comedy": 35,
        "crime": 80,
        "documentary": 99,
        "drama": 18,
        "family": 10751,
        "fantasy": 10765,
        "kids": 10762,
        "mystery": 9648,
        "news": 10763,
        "reality": 10764,
        "sci-fi": 10765,
        "sci-fi & fantasy": 10765,
        "science fiction": 10765,
        "scifi": 10765,
        "soap": 10766,
        "talk": 10767,
        "war & politics": 10768,
        "western": 37
    }
    
    # Convert requested genres to IDs
    requested_ids = set()
    has_scifi_request = False
    has_fantasy_request = False
    
    for g in requested_genres:
        g_lower = g.lower().strip()
        if g_lower in ["sci-fi", "scifi", "science fiction"]:
            has_scifi_request = True
            requested_ids.add(10765)
        elif g_lower == "fantasy":
            has_fantasy_request = True
            requested_ids.add(10765)
        elif g_lower in genre_map:
            requested_ids.add(genre_map[g_lower])
    
    # Base score: count how many requested genres match
    base_score = len(requested_ids.intersection(set(tv_genres)))
    
    # Special logic for sci-fi vs fantasy distinction
    # Only apply if user specifically requested sci-fi (not fantasy)
    if has_scifi_request and not has_fantasy_request and 10765 in tv_genres:
        # User wants sci-fi, series has "Sci-Fi & Fantasy" genre
        # Now check for sci-fi indicators vs fantasy indicators
        
        # Positive indicators for sci-fi (space, action, mystery, tech)
        scifi_indicators = 0
        if 10759 in tv_genres:  # Action & Adventure (often sci-fi)
            scifi_indicators += 2
        if 9648 in tv_genres:  # Mystery (often sci-fi thrillers)
            scifi_indicators += 1
        if 16 in tv_genres:  # Animation (often sci-fi anime)
            scifi_indicators += 1
        
        # Negative indicators (more fantasy/supernatural than sci-fi)
        fantasy_indicators = 0
        if 18 in tv_genres:  # Drama (common in fantasy/supernatural)
            fantasy_indicators += 1
        if 80 in tv_genres:  # Crime (supernatural crime, not sci-fi)
            fantasy_indicators += 2
        if 10751 in tv_genres:  # Family (fairy tales, not sci-fi)
            fantasy_indicators += 1
        
        # Adjust score based on indicators
        indicator_score = scifi_indicators - fantasy_indicators
        base_score += indicator_score
    
    return max(0, base_score)  # Don't go negative

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

def _score_quality(vote_average, vote_count, popularity):
    """
    Score based on quality metrics: rating, number of votes, and popularity.
    Returns a quality score that balances all three factors.
    
    Args:
        vote_average: TMDB vote average (0-10)
        vote_count: Number of votes
        popularity: TMDB popularity score
    
    Returns:
        float: Quality score (0-10 scale)
    """
    # Normalize vote average (already 0-10)
    rating_score = vote_average if vote_average else 0
    
    # Vote count score: logarithmic scale to avoid over-weighting popular shows
    # 0-100 votes: 0-2 points
    # 100-1000 votes: 2-4 points
    # 1000-10000 votes: 4-6 points
    # 10000+ votes: 6-8 points
    import math
    if vote_count:
        vote_score = min(8, math.log10(max(1, vote_count)) * 2)
    else:
        vote_score = 0
    
    # Popularity score: normalize to 0-2 scale
    # TMDB popularity typically ranges 0-100+ but most shows are 0-50
    if popularity:
        popularity_score = min(2, popularity / 25)
    else:
        popularity_score = 0
    
    # Weighted combination:
    # - Rating: 60% (most important)
    # - Vote count: 30% (credibility)
    # - Popularity: 10% (current relevance)
    total_score = (
        rating_score * 0.6 +
        vote_score * 0.3 +
        popularity_score * 0.1
    )
    
    return round(total_score, 2)

def _add_randomness(base_score, randomness_factor=0.15):
    """
    Add controlled randomness to avoid always showing the same results.
    
    Args:
        base_score: The base score to randomize
        randomness_factor: How much randomness to add (0.0-1.0)
            0.15 = ±15% variation
    
    Returns:
        float: Score with randomness added
    """
    import random
    # Add random variation: ±randomness_factor of the base score
    variation = base_score * randomness_factor * (2 * random.random() - 1)
    return base_score + variation

@tool
def suggest_series(genre: Optional[str], mood: Optional[str],
                   providers_text: str, language: str, country: str) -> str:
    """
    Return up to 3 series tailored to user slots and country (watchable on owned providers).
    
    Args:
        genre: Comma-separated genre names or None (e.g., "sci-fi,mystery" or "comedy")
        mood: User's mood preference (optional)
        providers_text: Streaming providers
        language: Language preference
        country: User's country code
    """
    # Parse genres from comma-separated string
    genres = None
    if genre:
        genres = [g.strip() for g in genre.split(',') if g.strip()]
    
    logger.info(f"Suggesting series - Genres: {genres}, Mood: {mood}, Providers: {providers_text}, Lang: {language}, Country: {country}")
    
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
        params = build_discover_params(genres, mood, region, language)
        
        if owned_ids:
            params["with_watch_providers"] = "|".join(str(x) for x in owned_ids)
        
        logger.info(f"Querying TMDB with params: {params}")
        candidates = tmdb.discover_tv(params)[:MAX_CANDIDATES]
        logger.info(f"Found {len(candidates)} candidate series")
    except requests.RequestException as e:
        logger.error(f"TMDB API error: {str(e)}")
        return json.dumps({"error": f"TMDB API error: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Unexpected error during search: {str(e)}"}, ensure_ascii=False)
    
    # Build candidate list with multi-factor scoring
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
        
        # Extract metrics
        tv_genres = tv.get("genre_ids", [])
        first_air = tv.get("first_air_date")
        vote_average = tv.get("vote_average", 0)
        vote_count = tv.get("vote_count", 0)
        popularity = tv.get("popularity", 0)
        overview = tv.get("overview") or ""
        
        # Calculate individual scores
        genre_score = _score_genre_match(tv_genres, genres)
        freshness_score = _score_freshness(first_air)
        quality_score = _score_quality(vote_average, vote_count, popularity)
        
        # Combined score with weights:
        # - Quality: 50% (rating, votes, popularity)
        # - Genre match: 30% (relevance to request)
        # - Freshness: 20% (variety and recency)
        base_score = (
            quality_score * 0.5 +
            genre_score * 0.3 +
            freshness_score * 0.2
        )
        
        # Apply mood adjustments (genre-based + semantic matching in overview)
        mood_adjusted_score = apply_mood_to_score(base_score, tv_genres, overview, mood)
        
        # Add controlled randomness for variety (±15%)
        final_score = _add_randomness(mood_adjusted_score, randomness_factor=0.15)
        
        overview_trimmed = _trim_overview(overview)
        scored_candidates.append({
            "title": tv.get("name"),
            "overview": overview_trimmed,
            "vote": vote_average,
            "vote_count": vote_count,
            "popularity": popularity,
            "first_air_date": first_air,
            "where": where[:5],
            "genre_score": genre_score,
            "freshness_score": freshness_score,
            "quality_score": quality_score,
            "base_score": base_score,
            "mood_adjusted_score": mood_adjusted_score,
            "final_score": final_score,
            "genre_ids": tv_genres
        })
    
    # Sort by final score (includes randomness for variety)
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Log top candidates with scoring breakdown
    logger.debug("Top candidates with scoring breakdown:")
    for i, c in enumerate(scored_candidates[:5], 1):
        logger.debug(
            f"  {i}. {c['title']} ({c['first_air_date']}) - "
            f"Final: {c['final_score']:.2f} "
            f"[Base: {c['base_score']:.2f}, "
            f"Mood-adj: {c['mood_adjusted_score']:.2f}, "
            f"Quality: {c['quality_score']:.2f}, "
            f"Genre: {c['genre_score']}, "
            f"Fresh: {c['freshness_score']}, "
            f"Votes: {c['vote_count']}, "
            f"Rating: {c['vote']:.1f}]"
        )
    
    # Take top N and remove scoring metadata
    suggestions = []
    for candidate in scored_candidates[:TOP_SUGGESTIONS]:
        suggestions.append({
            "title": candidate["title"],
            "overview": candidate["overview"],
            "vote": candidate["vote"],
            "first_air_date": candidate["first_air_date"],
            "where": candidate["where"]
        })
    
    logger.info(f"Returning {len(suggestions)} suggestions")
    
    if not suggestions:
        if owned_ids:
            msg = (f"No titles available on the specified providers. "
                   f"💡 Try removing the provider filter or selecting different genres/moods.")
        else:
            msg = (f"No suitable titles found. "
                   f"💡 Try different genres, moods, or broadening your language preference.")
        logger.warning(f"No suggestions found: {msg}")
        return json.dumps({"country": region, "suggestions": [], "note": msg}, ensure_ascii=False)
    
    return json.dumps({"country": region, "suggestions": suggestions}, ensure_ascii=False)

@tool
def get_show_info(show_name: str, country: str, providers_text: Optional[str] = None) -> str:
    """
    Get detailed information about a specific TV show by name.
    Returns info about the show including where it's available to watch.
    
    Args:
        show_name: The name of the TV show to search for
        country: The user's country code (e.g., 'US', 'IT')
        providers_text: Optional. User's preferred providers to check availability
    
    Returns:
        JSON string with show details and availability info
    """
    logger.info(f"Getting info for show: {show_name} in country: {country}")
    
    if not tmdb:
        logger.error("TMDB API key is missing")
        return json.dumps({"error": "Missing TMDB_API_KEY"}, ensure_ascii=False)
    
    region = (country or "US").upper()
    
    try:
        # Search for the show
        search_results = tmdb.search_tv_show(show_name)
        
        if not search_results:
            logger.info(f"No results found for '{show_name}'")
            return json.dumps({
                "error": f"No TV show found matching '{show_name}'",
                "suggestion": "Try checking the spelling or being more specific"
            }, ensure_ascii=False)
        
        # Get the first (most relevant) result
        show = search_results[0]
        show_id = show.get("id")
        
        if not show_id:
            return json.dumps({
                "error": f"Invalid show data for '{show_name}'"
            }, ensure_ascii=False)
        
        # Get detailed info
        details = tmdb.get_tv_details(show_id)
        
        if not details:
            return json.dumps({
                "error": f"Could not retrieve details for '{show_name}'"
            }, ensure_ascii=False)
        
        # Get watch providers
        try:
            providers = tmdb.watch_providers_for(show_id, region)
        except Exception:
            providers = []
        
        # Check if it's available on user's preferred providers
        user_has_access = False
        if providers_text:
            user_prov_norm = normalize_providers(providers_text)
            user_providers_lower = [p.lower() for p in user_prov_norm]
            providers_lower = [p.lower() for p in providers]
            user_has_access = any(up in providers_lower for up in user_providers_lower)
        
        # Prepare response
        response = {
            "title": details.get("name"),
            "original_title": details.get("original_name"),
            "overview": details.get("overview"),
            "first_air_date": details.get("first_air_date"),
            "last_air_date": details.get("last_air_date"),
            "status": details.get("status"),
            "number_of_seasons": details.get("number_of_seasons"),
            "number_of_episodes": details.get("number_of_episodes"),
            "genres": [g.get("name") for g in details.get("genres", [])],
            "vote_average": details.get("vote_average"),
            "vote_count": details.get("vote_count"),
            "popularity": details.get("popularity"),
            "available_on": providers,
            "user_has_access": user_has_access,
            "country": region,
            "episode_runtime": details.get("episode_run_time", []),
            "homepage": details.get("homepage"),
            "in_production": details.get("in_production"),
            "languages": details.get("languages", []),
            "networks": [n.get("name") for n in details.get("networks", [])]
        }
        
        logger.info(f"Found show: {response['title']}, Available on: {providers}, User has access: {user_has_access}")
        
        return json.dumps(response, ensure_ascii=False)
        
    except requests.RequestException as e:
        logger.error(f"TMDB API error while getting show info: {str(e)}")
        return json.dumps({"error": f"Failed to retrieve show information: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Unexpected error getting show info: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Unexpected error: {str(e)}"}, ensure_ascii=False)


# =========================
# LLM + LCEL components
# =========================
# Create separate LLM instances for different purposes with appropriate temperatures

# Deterministic LLM for slot extraction
llm_extract = ChatOpenAI(
    base_url=OVH_BASE,
    api_key=OVH_KEY,
    temperature=TEMP_EXTRACT,  # Deterministic for structured extraction
    model=MODEL or "gpt-oss-120b"
)

# Creative LLM for response generation
llm_creative = ChatOpenAI(
    base_url=OVH_BASE,
    api_key=OVH_KEY,
    temperature=TEMP_CREATIVE,  # More creative for natural responses
    model=MODEL or "gpt-oss-120b"
)

# Balanced LLM for classification/validation
llm_classify = ChatOpenAI(
    base_url=OVH_BASE,
    api_key=OVH_KEY,
    temperature=TEMP_CLASSIFY,  # Balanced for classification tasks
    model=MODEL or "gpt-oss-120b"
)

# Legacy alias for backward compatibility (use llm_extract for slot-filling)
llm_base = llm_extract

# 1) Chain LCEL: slot extraction/validation (structured output)
collect_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly TV concierge helping users find their next favorite series. "
     "Your job is to understand what they're looking for through natural conversation.\n\n"
     
     "WHAT YOU NEED TO KNOW:\n"
     "- Genre(s): {genres} (users can choose ONE or MULTIPLE genres, like 'sci-fi and mystery' or just 'comedy')\n"
     "- Mood/Tone: {moods}\n"
     "- Language: {langs} (OR 'any' if they don't have a preference)\n"
     "- Streaming Services: (any service like Netflix, Prime Video, Disney+, etc.)\n\n"
     
     "WHAT YOU ALREADY KNOW:\n"
     "- Genre(s): {current_genre}\n"
     "- Mood: {current_mood}\n"
     "- Providers: {current_providers}\n"
     "- Language: {current_language}\n\n"
     
     "HOW TO INTERACT:\n"
     "1. **Extract everything** from the user's message in one go - read carefully!\n"
     "2. **Multiple genres are OK** - If user says 'sci-fi and mystery' or 'comedy, drama', extract as a comma-separated list\n"
     "3. **Be permissive with genres** - Extract ANY genre-related words (even if not in the list above). The system will map them intelligently (e.g., 'horror' → 'mystery, thriller')\n"
     "4. **Be permissive with language** - If user says 'any language', 'any will be fine', 'doesn't matter', or 'no preference', extract as 'any'\n"
     "5. **Keep what you know** - Don't change fields that are already filled unless the user explicitly wants to change them.\n"
     "6. **Be natural** - If something's missing, ask ONE simple question. Be conversational, friendly, and brief (1-2 sentences max).\n"
     "7. **Don't repeat yourself** - NEVER ask about information you already have (check WHAT YOU ALREADY KNOW above).\n"
     "8. **Know when you're done** - Once you have all 4 required fields (genre, mood, providers, language), set ask_next to null.\n\n"
     
     "EXAMPLES OF NATURAL QUESTIONS (do NOT list options!):\n"
     "- Missing genre: \"What kind of genre are you in the mood for? You can choose one or more!\"\n"
     "- Missing mood: \"How are you feeling today? Choose from: light-hearted, intense, mind-blowing, comforting, or adrenaline-fueled.\"\n"
     "- Missing providers: \"Which streaming service do you have?\"\n"
     "- Missing language: \"Any preference on the language?\"\n\n"
     
     "RESPONSE FORMAT (valid JSON only):\n"
     '{{\n'
     '  "known": {{\n'
     '    "genre": "comma-separated genres or null (e.g., \'sci-fi,mystery\' or \'comedy\')",\n'
     '    "mood": "extracted value or null",\n'
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
assistant_with_tools = llm_base.bind_tools([ipinfo_location, suggest_series, get_show_info])

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
final_chain = final_prompt | llm_creative  # Use creative LLM for responses

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
    
    # Handle "any will/would be fine/good/ok" variations
    if re.search(r'\bany\s+(will|would)\s+be\s+(fine|good|ok|okay|great)\b', text, re.IGNORECASE):
        text = re.sub(r'\bany\s+(will|would)\s+be\s+(fine|good|ok|okay|great)\b', 'any language', text, flags=re.IGNORECASE)
    
    # Handle explicit "any" -> "any language" (before other "any" replacements)
    if text.lower() == "any":
        text = "any language"
    
    # Handle "any language" variations
    if re.search(r'\b(any language|all languages|no language preference)\b', text, re.IGNORECASE):
        text = re.sub(r'\b(any language|all languages|no language preference)\b', 'any', text, flags=re.IGNORECASE)
    
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
            # For genres, merge lists instead of replacing
            if k == 'genre' and isinstance(v, list):
                existing = data.get('genre')
                if existing and isinstance(existing, list):
                    # Combine and deduplicate genres
                    combined = list(set(existing + v))
                    data[k] = combined
                else:
                    data[k] = v
            else:
                data[k] = v
    return UserPrefs(**data)

def prefs_complete(p: UserPrefs) -> bool:
    """Checks if all required slots are filled."""
    return all([p.genre, p.mood, p.providers, p.language])

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
    user_input = state.get("user_input", "").lower().strip()
    
    # If no input yet (initial greeting), end and wait for user
    if not user_input:
        return "end"
    
    # Check for exit commands
    if user_input in {"exit", "quit", "bye", "goodbye"}:
        return "end"
    
    # Check for new search request
    if user_input == "new":
        return "end"
    
    # Check if preferences are complete or user wants to search
    if state["prefs_complete"] or user_input in {"search", "go", "find"}:
        return "search"
    
    # After processing input once, always end to wait for next user input
    # Don't loop back to collect - let main loop handle that
    return "end"

def should_continue(state: ConversationState) -> Literal["feedback", "collect", "end"]:
    """Router: after search/feedback, decide next step."""
    user_input = state.get("user_input", "").lower().strip()
    
    # Check for exit
    if user_input in {"exit", "quit", "bye", "goodbye"}:
        return "end"
    
    # Check for new search
    if user_input == "new":
        return "end"
    
    # Always end after processing to wait for next user input
    return "end"

def router_entry(state: ConversationState) -> ConversationState:
    """Entry router to direct to the right node based on state."""
    # Just pass through - routing is done by should_route_entry
    return state

def should_route_entry(state: ConversationState) -> Literal["feedback", "collect"]:
    """Router from entry point."""
    # If waiting for feedback, go to feedback node
    if state.get("waiting_for_feedback"):
        return "feedback"
    # Otherwise go to collecting
    return "collect"

def greeting_node(state: ConversationState) -> ConversationState:
    """Initial greeting node - display welcome message only once."""
    # Only show greeting if this is the first turn
    if state["conversation_turn"] == 0:
        print("🍿 SerieBot — Geo-located TV Suggestions (OVH + LangChain + ipinfo + TMDB)")
        print("Type 'exit' to quit, 'new' to start over, or 'search' to search with current preferences.\n")
        
        print("Hi! 👋 What kind of TV series are you looking for?")
        print("You can tell me about genre, mood, language, or platforms (e.g., Netflix, Prime)...")
    
    state["current_state"] = "collecting"
    return state

def collecting_node(state: ConversationState) -> ConversationState:
    """Node for collecting user preferences using slot-filling extraction."""
    logger.info(f"Collecting preferences - Turn {state['conversation_turn']}")
    
    user_input = state.get("user_input", "")
    
    # Skip if no input (first greeting)
    if not user_input:
        state["current_state"] = "collecting"
        return state
    
    bot = state["bot"]
    
    # Check for show-specific questions BEFORE extraction
    if bot._is_show_question(user_input):
        logger.info("Detected show-specific question")
        bot.answer_show_question(user_input)
        state["current_state"] = "collecting"
        return state
    
    # Check for informational questions BEFORE extraction
    if bot.is_informational_question(user_input):
        logger.info("Detected informational question")
        bot.answer_informational_question(user_input)
        state["current_state"] = "collecting"
        return state
    
    # Check if input is off-topic BEFORE extraction
    # The is_off_topic method now handles its own response
    if bot.is_off_topic(user_input, state.get("last_ask_next") or ""):
        logger.warning(f"Off-topic input detected: {user_input}")
        # Response already generated by is_off_topic method
        state["current_state"] = "collecting"
        return state
    
    # Check for force search intent using LLM-based detection
    current_prefs = {
        "genre": state["prefs"].genre,
        "mood": state["prefs"].mood,
        "providers": state["prefs"].providers,
        "language": state["prefs"].language
    }
    
    if detect_force_search_intent(user_input, current_prefs):
        state["is_force_search"] = True
        logger.info("LLM detected force search intent")
    
    # Extract preferences and get mapping info
    # Store prefs BEFORE extraction to detect what changed
    prefs_before = bot.prefs.model_dump()
    
    ask_next, mapping_info = bot.extract_preferences(user_input)
    
    # Update bot's prefs to state
    state["prefs"] = bot.prefs
    prefs_after = bot.prefs.model_dump()
    
    # Build "Got it!" message showing what was JUST extracted (changed)
    extracted_items = []
    
    for field, value_after in prefs_after.items():
        value_before = prefs_before.get(field)
        # If value changed and is not None, it was just extracted
        if value_after != value_before and value_after is not None:
            # Format the value nicely
            if isinstance(value_after, list):
                formatted_value = ', '.join(value_after)
            else:
                formatted_value = str(value_after)
            
            # Map to user-friendly names
            field_names = {
                'genre': 'Genre',
                'mood': 'Mood',
                'providers': 'Platforms',
                'language': 'Language'
            }
            friendly_name = field_names.get(field, field)
            extracted_items.append(f"{friendly_name}: {formatted_value}")
    
    # Show "Got it!" if something was extracted
    if extracted_items:
        print(f"\nai> Got it! {', '.join(extracted_items)}")
    
    # Display any LLM mapping feedback to user
    if mapping_info:
        for field, info in mapping_info.items():
            if 'mapped' in info:
                orig = info['original']
                mapped = info['mapped']
                if isinstance(orig, list):
                    orig_str = ', '.join(orig)
                else:
                    orig_str = orig
                if isinstance(mapped, list):
                    mapped_str = ', '.join(mapped)
                else:
                    mapped_str = mapped
                print(f"💡 I mapped '{orig_str}' to: {mapped_str}")
    
    # Update state
    state["next_question"] = ask_next
    state["last_ask_next"] = ask_next
    state["conversation_turn"] += 1
    
    # Check if preferences are complete
    state["prefs_complete"] = (
        state["prefs"].genre is not None and
        state["prefs"].mood is not None and
        state["prefs"].providers is not None and
        state["prefs"].language is not None
    ) or state["is_force_search"]
    
    # Detect repeat loops
    if ask_next == state.get("last_ask_next") and ask_next is not None:
        state["repeat_count"] += 1
        if state["repeat_count"] >= MAX_REPEATS:
            # Show contextual help
            print(f"\nai> I notice we're going in circles. Let me help! 🔄\n")
            # Provide examples based on what's missing
            if not state["prefs"].genre:
                print("📺 For genre, try: 'sci-fi', 'comedy', 'drama', 'mystery', 'action'")
            if not state["prefs"].mood:
                print("🎭 For mood, try: 'light-hearted', 'intense', 'comforting', 'adrenaline-fueled'")
            if not state["prefs"].providers:
                print("📺 For platforms, try: 'Netflix', 'Prime Video', 'Disney+', 'Apple TV+'")
            if not state["prefs"].language:
                print("🌍 For language, try: 'English', 'Italian', 'Spanish', 'any language'")
            print()
            state["repeat_count"] = 0
    else:
        state["repeat_count"] = 0
    
    # Print next question if not complete
    if not state["prefs_complete"] and ask_next:
        print(f"\nai> {ask_next}")
    
    state["current_state"] = "collecting"
    return state

def searching_node(state: ConversationState) -> ConversationState:
    """Node for performing TMDB search and displaying results."""
    logger.info("Searching node activated")
    
    bot = state["bot"]
    prefs = state["prefs"]
    
    # Get geo location if not cached
    if state["geo_cache"] is None:
        try:
            geo_res = ipinfo_location.invoke({})
            geo_data = json.loads(geo_res)
            state["geo_cache"] = geo_data
        except Exception as e:
            logger.error(f"Failed to get location: {e}")
            state["geo_cache"] = {"country": "US"}
    
    country = state["geo_cache"].get("country", "US")
    
    # Prepare parameters for suggest_series tool
    genre_str = ', '.join(prefs.genre) if prefs.genre else None
    providers_str = ', '.join(prefs.providers) if prefs.providers else None
    
    print("\n📍 Searching for recommendations...")
    
    try:
        # Call the suggest_series tool
        result_json = suggest_series.invoke({
            "genre": genre_str,
            "mood": prefs.mood,
            "providers_text": providers_str,
            "language": prefs.language,
            "country": country
        })
        
        result = json.loads(result_json)
        
        if "error" in result:
            print(f"\nai> {result['error']}")
            state["last_suggestions_count"] = 0
        else:
            suggestions = result.get("suggestions", [])
            state["last_suggestions_count"] = len(suggestions)
            state["last_suggestions"] = suggestions
            
            # Display suggestions
            print("\n🎬 Here are my top suggestions:\n")
            for i, show in enumerate(suggestions, 1):
                print(f"{i}. {show['title']} ({show.get('first_air_date', 'N/A')})")
                print(f"   ⭐ {show['vote']}/10")
                print(f"   📝 {show['overview']}")
                if show.get('where'):
                    print(f"   📺 Available on: {show['where']}")
                print()
            
            # Show feedback prompt
            print("How do these look? Please rate:")
            print("• 👍 (or type 'good', 'like', 'yes') if you liked them")
            print("• 👎 (or type 'bad', 'dislike', 'no') if you didn't")
            print("• 'skip' to skip feedback")
            print("Or refine your search (e.g., 'change genre to comedy', 'only Netflix')")
            print("Type 'new' to start over or 'exit' to quit")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"\nai> Sorry, search failed: {str(e)}")
        state["last_suggestions_count"] = 0
    
    state["waiting_for_feedback"] = True
    state["current_state"] = "searching"
    
    return state

def feedback_node(state: ConversationState) -> ConversationState:
    """Node for handling user feedback on suggestions."""
    logger.info("Feedback node activated")
    
    bot = state["bot"]
    user_input = state["user_input"]
    
    # Use BotSession's feedback handler
    handled, next_q = bot.handle_feedback(user_input)
    
    if handled:
        state["waiting_for_feedback"] = bot.waiting_for_feedback
        if next_q:
            print(f"\nai> {next_q}")
            state["next_question"] = next_q
    
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
        self.repeat_count = 0
        self.last_suggestions_count = 0
        self.waiting_for_feedback = False
        
        self.informational_questions = [
            "what are", "what is", "tell me about", "explain", "how do",
            "how does", "why do", "why does", "who is", "who are"
        ]
        
        self.search_keywords = {
            "netflix", "prime", "disney", "apple tv", "hbo", "hulu", 
            "genre", "mood", "language", "provider", "platform"
        }
        
        logger.info(f"New session created: {self.session_id}")
    
    def reset(self):
        """Reset the session state for a new search."""
        logger.info("Resetting session state")
        self.prefs = UserPrefs()
        self.geo_cache = None
        self.repeat_count = 0
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
        
        # Check if it's a question about a specific show or informational question
        # Keep feedback mode active and answer the question
        if self._is_show_question(user_input):
            # Answer the show question but stay in feedback mode
            self.answer_show_question(user_input)
            # Return the same feedback prompt
            next_q = ("How do these look? Please rate:\n"
                     "• 👍 (or type 'good', 'like', 'yes') if you liked them\n"
                     "• 👎 (or type 'bad', 'dislike', 'no') if you didn't\n"
                     "• 'skip' to skip feedback\n"
                     "Or refine your search (e.g., 'change genre to comedy', 'only Netflix')\n"
                     "Type 'new' to start over or 'exit' to quit")
            return True, next_q
        
        if self.is_informational_question(user_input):
            # Answer the informational question but stay in feedback mode
            self.answer_informational_question(user_input)
            # Return the same feedback prompt
            next_q = ("How do these look? Please rate:\n"
                     "• 👍 (or type 'good', 'like', 'yes') if you liked them\n"
                     "• 👎 (or type 'bad', 'dislike', 'no') if you didn't\n"
                     "• 'skip' to skip feedback\n"
                     "Or refine your search (e.g., 'change genre to comedy', 'only Netflix')\n"
                     "Type 'new' to start over or 'exit' to quit")
            return True, next_q
        
        # Not feedback and not a question, turn off feedback mode
        self.waiting_for_feedback = False
        return False, None
    
    def is_informational_question(self, user_input: str) -> bool:
        """Check if input is an informational question (not about search params)."""
        user_lower = user_input.lower().strip()
        
        # Check for explicit list/options requests
        list_request_patterns = [
            r'\b(which|what)\s+(genre|mood|language|provider|platform)s?\s+(can|should|do|are)',
            r'\bgive\s+me\s+(the\s+)?(list|options)',
            r'\bshow\s+me\s+(the\s+)?(list|options|choices)',
            r'\bwhat\s+(are\s+)?(the\s+)?(available|possible|valid)',
            r'\bi\s+need\s+(the\s+)?list',
            r'\blist\s+of\s+(genre|mood|language|provider)s?',
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
    
    def _is_show_question(self, user_input: str) -> bool:
        """
        Check if the user is asking about a specific TV show.
        e.g., "what about the expanse?", "is breaking bad available?", "tell me about stranger things"
        """
        user_lower = user_input.lower().strip()
        
        # Patterns for asking about specific shows
        show_question_patterns = [
            r'\bwhat about\b',                    # "what about the expanse?"
            r'\bhow about\b',                     # "how about breaking bad?"
            r'\btell me about\b',                 # "tell me about stranger things"
            r'\btalk to me about\b',              # "talk to me about lord of mysteries"
            r'\bis\s+.+\bavailable\b',            # "is the expanse available?"
            r'\bdo you have\s+["\']?.+["\']?\b',  # "do you have game of thrones?"
            r'\bcan (i|you) find\s+.+',           # "can you find dark?"
            r'\bwhy (not|isn\'t)\s+.+',           # "why not the expanse?"
            r'\bwhere (is|can i watch)\s+.+',     # "where can i watch the expanse?"
            r'\bhave you heard of\s+["\']?.+["\']?\b',  # "have you heard of The Expanse?"
            r'\bdo you know\s+["\']?.+["\']?\s*\??\s*$', # "do you know Breaking Bad?" or "Do you know "reincarnated in a sword?""
            r'\bwhat do you think (of|about)\s+["\']?.+["\']?', # "what do you think about/of Stranger Things?"
            r'\binfo (on|about)\b',               # "info on The Expanse"
            r'\bdetails (on|about)\b',            # "details about Breaking Bad"
            r'\bever heard of\s+["\']?.+["\']?\b',# "ever heard of X?"
            r'\bknow anything about\s+.+',        # "know anything about X?"
            r'\byour (opinion|thoughts?) (on|about)\s+.+', # "your opinion on X"
        ]
        
        for pattern in show_question_patterns:
            if re.search(pattern, user_lower):
                return True
        
        return False
    
    def _extract_show_name(self, user_input: str) -> Optional[str]:
        """
        Extract the show name from a question about a specific show.
        e.g., "what about The Expanse?" -> "The Expanse"
        """
        user_input = user_input.strip()
        
        # Remove trailing punctuation
        user_input = re.sub(r'[?!.]+$', '', user_input).strip()
        
        # Patterns to extract show name
        patterns = [
            r'what about\s+["\']?(.+?)["\']?$',
            r'how about\s+["\']?(.+?)["\']?$',
            r'tell me about\s+["\']?(.+?)["\']?$',
            r'talk to me about\s+["\']?(.+?)["\']?$',
            r'is\s+["\']?(.+?)["\']?\s+available',
            r'do you have\s+["\']?(.+?)["\']?$',
            r'can (?:i|you) find\s+["\']?(.+?)["\']?$',
            r'why (?:not|isn\'t)\s+["\']?(.+?)["\']?$',
            r'where (?:is|can i watch)\s+["\']?(.+?)["\']?$',
            r'have you heard of\s+["\']?(.+?)["\']?$',
            r'do you know\s+["\']?(.+?)["\']?$',
            r'what do you think (?:of|about)\s+["\']?(.+?)["\']?$',
            r'your (?:opinion|thoughts?) (?:on|about)\s+["\']?(.+?)["\']?$',
            r'info (?:on|about)\s+["\']?(.+?)["\']?$',
            r'details (?:on|about)\s+["\']?(.+?)["\']?$',
            r'ever heard of\s+["\']?(.+?)["\']?$',
            r'know anything about\s+["\']?(.+?)["\']?$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                show_name = match.group(1).strip()
                # Remove surrounding quotes if present
                show_name = re.sub(r'^["\']|["\']$', '', show_name)
                # Clean up common trailing words
                show_name = re.sub(r'\s+(show|series|tv show)$', '', show_name, flags=re.IGNORECASE)
                return show_name
        
        return None
    
    def answer_show_question(self, user_input: str) -> bool:
        """
        Answer a question about a specific TV show using TMDB.
        Returns True if handled, False otherwise.
        """
        show_name = self._extract_show_name(user_input)
        
        if not show_name:
            logger.warning(f"Could not extract show name from: {user_input}")
            return False
        
        logger.info(f"Answering question about show: {show_name}")
        
        # Get user's country (use cached or fetch)
        if self.geo_cache is None:
            try:
                geo_res = ipinfo_location.invoke({})
                geo_data = json.loads(geo_res)
                self.geo_cache = geo_data
            except Exception as e:
                logger.error(f"Failed to get location: {e}")
                self.geo_cache = {"country": "US"}
        
        country = (self.geo_cache or {}).get("country", "US")
        
        try:
            # Call the tool
            show_info_json = get_show_info.invoke({
                "show_name": show_name,
                "country": country,
                "providers_text": self.prefs.providers
            })
            
            show_info = json.loads(show_info_json)
            
            # Check for errors
            if "error" in show_info:
                print(f"ai> {show_info['error']}")
                if "suggestion" in show_info:
                    print(f"    {show_info['suggestion']}")
                print()
                return True
            
            # Format and display the information
            print(f"\nai> Here's what I found about **{show_info.get('title', show_name)}**:\n")
            
            # Overview
            overview = show_info.get('overview')
            if overview:
                wrapped = textwrap.fill(overview, width=80, initial_indent='    ', subsequent_indent='    ')
                print(wrapped)
                print()
            
            # Basic info
            print(f"    📅 First aired: {show_info.get('first_air_date', 'N/A')}")
            print(f"    📺 Status: {show_info.get('status', 'N/A')}")
            
            seasons = show_info.get('number_of_seasons')
            episodes = show_info.get('number_of_episodes')
            if seasons and episodes:
                print(f"    🎬 {seasons} season{'s' if seasons > 1 else ''}, {episodes} episodes")
            
            # Genres
            genres = show_info.get('genres', [])
            if genres:
                print(f"    🎭 Genres: {', '.join(genres)}")
            
            # Runtime
            runtime = show_info.get('episode_runtime', [])
            if runtime:
                avg_runtime = sum(runtime) / len(runtime)
                print(f"    ⏱️  Episode length: ~{int(avg_runtime)} min")
            
            # Rating
            vote = show_info.get('vote_average')
            if vote:
                print(f"    ⭐ Rating: {vote:.1f}/10")
            
            # Availability
            providers = show_info.get('available_on', [])
            if providers:
                print(f"    📡 Available on: {', '.join(providers[:5])}")
                
                # Check if user has access
                if show_info.get('user_has_access'):
                    print(f"    ✅ Good news! It's available on your platform{'s' if self.prefs.providers and ',' in self.prefs.providers else ''}!")
                elif self.prefs.providers:
                    print(f"    ⚠️  Note: Not currently available on {self.prefs.providers}")
            else:
                print(f"    ⚠️  Not available for streaming in {country}")
            
            print()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to get show info: {e}", exc_info=True)
            print(f"ai> I'm having trouble getting information about that show right now.\n")
            return True
    
    def answer_informational_question(self, user_input: str) -> bool:
        """
        Answer an informational question.
        Returns True if handled, False otherwise.
        """
        logger.info(f"Detected informational question: {user_input}")
        
        # Check if it's a question about a specific show
        if self._is_show_question(user_input):
            return self.answer_show_question(user_input)
        
        # Otherwise, answer as a general informational question
        try:
            question_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are a helpful TV concierge assistant. Answer the user's question clearly and concisely. "
                 "Be friendly and informative. Keep your answer to 2-3 sentences.\n\n"
                 "CONTEXT:\n"
                 f"- Current preferences: genre={self.prefs.genre}, mood={self.prefs.mood}, providers={self.prefs.providers}, language={self.prefs.language}\n"
                 f"- Supported platforms: Netflix, Prime Video, Disney+, Apple TV+, HBO Max, Hulu, and many others\n"
                 f"- Supported genres: {', '.join(sorted(get_allowed_genres()))}\n"
                 f"- Supported moods: {', '.join(sorted(ALLOWED_MOOD))}\n"
                 f"- Languages: {', '.join(sorted(ALLOWED_LANG))}\n\n"
                 "INSTRUCTIONS:\n"
                 "1. If asked about system capabilities:\n"
                 "   - Provide information about supported options\n"
                 "2. DO NOT make new recommendations unless the user explicitly asks to search again\n"
                 "3. Stay helpful and conversational"
                ),
                ("human", "{question}")
            ])
            
            answer_chain = question_prompt | llm_creative  # Use creative LLM
            
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
    
    def is_off_topic(self, user_input: str, last_bot_question: str = "") -> bool:
        """
        Check if the user input is completely off-topic (not related to TV series at all).
        Uses LLM to intelligently detect off-topic conversations.
        If off-topic, generates and displays a natural response to redirect user.
        
        Args:
            user_input: The user's message
            last_bot_question: The last question asked by the bot (for context)
        
        Returns True if the input seems unrelated to TV series recommendations.
        """
        user_lower = user_input.lower().strip()
        
        # Very short inputs are usually not off-topic
        if len(user_input.strip()) < 5:
            return False
        
        # Quick check: obvious on-topic keywords
        on_topic_keywords = [
            'show', 'series', 'tv', 'watch', 'episode', 'season', 'stream',
            'netflix', 'prime', 'disney', 'hbo', 'genre', 'recommend', 'suggestion',
            'sci-fi', 'comedy', 'drama', 'thriller', 'mystery', 'animation',
            'intense', 'light', 'light-hearted', 'mood', 'language',
            'english', 'italian', 'japanese', 'korean'
        ]
        
        # If it contains obvious on-topic keywords, it's not off-topic
        if any(keyword in user_lower for keyword in on_topic_keywords):
            return False
        
        # Quick regex check for obvious cases (faster than LLM call)
        obvious_off_topic = [
            r'\b(weather|wheater|temperature|forecast|rain|sunny|cloud)\b',
            r'\b(recipe|cook|bake|pizza|pasta|burger|meal|eat|hungry|food)\b',
            r'\b(math|calculate|equation|formula|algebra)\b',
            r'\b(stock|invest|trading|crypto|bitcoin)\b',
        ]
        
        for pattern in obvious_off_topic:
            if re.search(pattern, user_lower):
                # Check if they're talking about a show
                show_context = any([
                    'show' in user_lower,
                    'series' in user_lower,
                    'watch' in user_lower,
                    'tv' in user_lower,
                    'episode' in user_lower,
                ])
                if not show_context:
                    logger.info(f"Off-topic detected via regex: {pattern}")
                    self._respond_to_off_topic(user_input)
                    return True
        
        # Use LLM for nuanced detection with conversation context
        try:
            context_info = ""
            if last_bot_question:
                context_info = f"\n\nPrevious bot question: {last_bot_question}"
            
            off_topic_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a classifier that determines if a user's message is related to TV series recommendations.\n\n"
                 "TV Series topics include:\n"
                 "- Asking for series recommendations (genre, mood, platform, etc.)\n"
                 "- Answering questions about preferences (e.g., 'intense', 'light-hearted', '~45m')\n"
                 "- Asking about specific TV shows\n"
                 "- Questions about streaming platforms or availability\n"
                 "- Preferences for watching series (duration, language, etc.)\n"
                 "- Feedback on recommendations\n\n"
                 "Off-topic includes:\n"
                 "- Weather, food, sports, politics, health, finance, etc.\n"
                 "- Math problems, translations, jokes\n"
                 "- Anything not related to TV series or entertainment\n\n"
                 "If the user is answering a question from the bot, consider it on-topic if it's about preferences."
                 "{context}"
                 "\n\nRespond ONLY with 'on-topic' or 'off-topic' (no explanation)."),
                ("human", "{user_input}")
            ])
            
            classifier_chain = off_topic_prompt | llm_classify  # Use classifier LLM
            result = classifier_chain.invoke({
                "user_input": user_input,
                "context": context_info
            })
            
            # Extract content from the result
            content = result.content if hasattr(result, 'content') else str(result)
            
            # Handle if content is a list (sometimes LangChain returns list of content blocks)
            if isinstance(content, list):
                # Join list elements if it's a list
                classification = " ".join(str(item) for item in content).strip().lower()
            else:
                classification = str(content).strip().lower()
            
            is_off = "off-topic" in classification
            
            if is_off:
                logger.info(f"Off-topic detected via LLM: {user_input}")
                self._respond_to_off_topic(user_input)
            else:
                logger.debug(f"On-topic confirmed via LLM: {user_input}")
            
            return is_off
            
        except Exception as e:
            logger.error(f"LLM off-topic detection failed: {e}")
            # Fallback to False (don't block if LLM fails)
            return False
    
    def _respond_to_off_topic(self, user_input: str) -> None:
        """
        Generate a natural response to off-topic input using LLM.
        The response should gently redirect the user to TV series recommendations.
        """
        try:
            # Build context about what we need
            missing = []
            if not self.prefs.genre:
                missing.append("genre")
            if not self.prefs.mood:
                missing.append("mood")
            if not self.prefs.providers:
                missing.append("streaming platform")
            if not self.prefs.language:
                missing.append("language")
            
            context_prefs = f"Current preferences: genre={self.prefs.genre}, mood={self.prefs.mood}, providers={self.prefs.providers}, language={self.prefs.language}"
            missing_info = f"Still missing: {', '.join(missing)}" if missing else "All preferences collected"
            
            redirect_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a friendly TV series recommendation bot. The user just said something off-topic.\n\n"
                 "Your task:\n"
                 "1. Politely acknowledge their message (1 brief sentence)\n"
                 "2. Gently redirect them back to TV series recommendations\n"
                 "3. If preferences are missing, ask about ONE missing preference naturally\n"
                 "4. Keep it warm, conversational, and brief (2-3 sentences total)\n\n"
                 "Context:\n"
                 f"{context_prefs}\n"
                 f"{missing_info}\n\n"
                 "Examples:\n"
                 "- User: 'what's the weather?' → 'I wish I could help with that! 😊 But I'm here to help you find great TV series. What kind of genre are you in the mood for?'\n"
                 "- User: 'tell me a joke' → 'Haha, I'm better at recommending shows than telling jokes! 😄 Let's find you something entertaining to watch. Do you have a streaming platform preference?'\n"
                 "\n"
                 "Be creative but always redirect to TV series. Don't list options unless asked."),
                ("human", "{user_input}")
            ])
            
            redirect_chain = redirect_prompt | llm_creative
            
            # Stream the response
            print("\nai> ", end="", flush=True)
            response_text = ""
            for chunk in redirect_chain.stream({"user_input": user_input}):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if isinstance(content, str):
                        response_text += content
                        print(content, end="", flush=True)
            print("\n")
            
            logger.debug(f"Off-topic redirect response: {response_text}")
            
        except Exception as e:
            logger.error(f"Failed to generate off-topic response: {e}", exc_info=True)
            # Fallback to simple message
            help_msg = self.get_contextual_help_message()
            print(f"\nai> I appreciate you talking to me, but I specialize in TV series recommendations! 😊\n")
            print(f"{help_msg}\n")
    
    def _looks_like_question(self, user_input: str) -> bool:
        """
        Check if the input looks like a question.
        """
        user_input = user_input.strip()
        
        # Ends with question mark
        if user_input.endswith('?'):
            return True
        
        # Starts with question words
        question_starters = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'do', 'does', 'is', 'are']
        first_word = user_input.lower().split()[0] if user_input else ""
        if first_word in question_starters:
            return True
        
        return False
    
    def get_contextual_help_message(self) -> str:
        """
        Generate a contextual help message based on the current state of the conversation.
        This helps users understand what the bot needs or can do.
        """
        # Check if we're waiting for feedback
        if self.waiting_for_feedback:
            return (
                "I can help you find TV series to watch! 📺\n\n"
                "Right now, I'm waiting for your feedback on my suggestions:\n"
                "• Rate them with 👍 (good) or 👎 (bad)\n"
                "• Type 'skip' if you don't want to give feedback\n"
                "• Or refine your search (e.g., 'change genre to comedy')\n"
                "• Type 'new' to start a fresh search"
            )
        
        # Check what we still need
        missing = []
        if not self.prefs.genre:
            missing.append("genre (e.g., sci-fi, comedy, drama)")
        if not self.prefs.mood:
            missing.append("mood (e.g., intense, light-hearted)")
        if not self.prefs.providers:
            missing.append("streaming platform (e.g., Netflix, Prime Video)")
        if not self.prefs.language:
            missing.append("language (e.g., English, Italian, any)")
        
        # Check what we have
        have = []
        if self.prefs.genre:
            have.append(f"genre: {', '.join(self.prefs.genre)}")
        if self.prefs.mood:
            have.append(f"mood: {self.prefs.mood}")
        if self.prefs.providers:
            have.append(f"platform: {self.prefs.providers}")
        if self.prefs.language:
            have.append(f"language: {self.prefs.language}")
        
        # Build the message
        msg = "I can help you find TV series to watch! 📺\n\n"
        
        if have:
            msg += f"✓ What I know: {', '.join(have)}\n\n"
        
        if missing:
            if len(missing) == 1:
                msg += f"I still need: {missing[0]}\n\n"
            elif len(missing) <= 3:
                msg += f"I still need: {' and '.join([', '.join(missing[:-1]), missing[-1]])}\n\n"
            else:
                msg += f"I still need: {', '.join(missing[:-1])}, and {missing[-1]}\n\n"
            
            msg += "💡 Tip: Just tell me naturally what you're looking for!\n"
            msg += "   Example: 'sci-fi and mystery on Netflix, intense, English'"
        else:
            msg += "I have all the info I need! Type 'search' to find suggestions."
        
        return msg
    
    def extract_preferences(self, user_input: str) -> Tuple[Optional[str], Dict]:
        """
        Extract preferences from user input.
        Returns (next_question, mapping_info) tuple.
        mapping_info contains any LLM mappings done (genre/mood/provider).
        """
        try:
            logger.debug("Invoking slot extraction chain")
            normalized_input = normalize_user_input(user_input)
            if normalized_input != user_input:
                logger.debug(f"Normalized input: '{user_input}' -> '{normalized_input}'")
            
            collect_chain = (
                collect_prompt.partial(
                    genres=", ".join(sorted(get_allowed_genres())),
                    moods=", ".join(sorted(ALLOWED_MOOD)),
                    langs=", ".join(sorted(ALLOWED_LANG)),
                    current_genre=", ".join(self.prefs.genre) if self.prefs.genre else "null",
                    current_mood=self.prefs.mood or "null",
                    current_providers=self.prefs.providers or "null",
                    current_language=self.prefs.language or "null"
                )
                | llm_extract  # Use deterministic extraction LLM
                | collect_parser
            )
            
            collect: CollectResult = collect_chain.invoke({"user_input": normalized_input})
            validated_prefs, mapping_info = validate_prefs(collect.known)
            logger.debug(f"Extracted preferences: {validated_prefs}")
            
            # Store old prefs to check if extraction made progress
            old_prefs = self.prefs.model_dump()
            self.prefs = merge_prefs(self.prefs, validated_prefs)
            new_prefs = self.prefs.model_dump()
            
            # Check if extraction made no progress and input wasn't empty
            no_progress = (
                old_prefs == new_prefs and  # No change in preferences
                not any(validated_prefs.model_dump().values()) and  # Nothing was extracted
                len(user_input.strip()) > 3 and  # Input wasn't trivial
                not self._looks_like_question(user_input) and  # And it wasn't a question
                self.repeat_count >= 2  # After 2 repeated questions, show help
            )
            
            if no_progress:
                logger.warning(f"No progress after {self.repeat_count} attempts with input: {user_input}")
                # Return contextual help
                return ("CONTEXTUAL_HELP", {})
            
            next_q = None if prefs_complete(self.prefs) else (collect.ask_next or "What other preference can you tell me?")
            
            logger.info(f"Current preferences: genre={self.prefs.genre}, mood={self.prefs.mood}, "
                       f"providers={self.prefs.providers}, language={self.prefs.language}")
            return (next_q, mapping_info)
            
        except Exception as e:
            logger.error(f"Failed to parse user input: {e}", exc_info=True)
            # Return contextual help instead of generic error
            return ("CONTEXTUAL_HELP", {})  # Return special marker for contextual help

# =========================
# LangGraph Graph Builder
# =========================
def build_conversation_graph():
    """
    Build the LangGraph conversation flow.
    
    States:
    - greeting: Initial welcome
    - router_entry: Routes to collecting or feedback based on state
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
    workflow.add_node("router_entry", router_entry)
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
    
    # Route from router_entry based on waiting_for_feedback
    workflow.add_conditional_edges(
        "router_entry",
        should_route_entry,
        {
            "feedback": "feedback",
            "collect": "collecting"
        }
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
    
    # Compile the graph WITHOUT memory (BotSession is not serializable)
    # Memory is managed through the BotSession instance passed in state
    app = workflow.compile()
    
    logger.info("LangGraph conversation flow built successfully")
    return app

def run_graph_cli():
    """Main entry point using LangGraph for conversation flow."""
    logger.info("Starting LangGraph CLI interface")
    
    # Build the graph
    app = build_conversation_graph()
    
    # Build a separate entry app that starts from router_entry
    entry_workflow = StateGraph(ConversationState)
    entry_workflow.add_node("router_entry", router_entry)
    entry_workflow.add_node("collecting", collecting_node)
    entry_workflow.add_node("searching", searching_node)
    entry_workflow.add_node("feedback", feedback_node)
    
    entry_workflow.add_edge(START, "router_entry")
    
    entry_workflow.add_conditional_edges(
        "router_entry",
        should_route_entry,
        {
            "feedback": "feedback",
            "collect": "collecting"
        }
    )
    
    entry_workflow.add_conditional_edges(
        "collecting",
        should_search,
        {
            "search": "searching",
            "collect": "collecting",
            "end": END
        }
    )
    
    entry_workflow.add_conditional_edges(
        "searching",
        should_continue,
        {
            "feedback": "feedback",
            "collect": "collecting",
            "end": END
        }
    )
    
    entry_workflow.add_conditional_edges(
        "feedback",
        should_continue,
        {
            "feedback": "feedback",
            "collect": "collecting",
            "end": END
        }
    )
    
    entry_app = entry_workflow.compile()
    
    # Initialize bot session and state
    bot = BotSession()
    
    # Initial state
    state: ConversationState = {
        "bot": bot,
        "prefs": bot.prefs,
        "session_id": bot.session_id,
        "user_input": "",
        "next_question": None,
        "last_ask_next": None,
        "repeat_count": 0,
        "conversation_turn": 0,
        "geo_cache": None,
        "last_suggestions_count": 0,
        "last_suggestions": None,
        "waiting_for_feedback": False,
        "prefs_complete": False,
        "is_force_search": False,
        "current_state": "greeting"
    }
    
    # Show greeting ONCE using full app
    state = app.invoke(state)
    
    # Main conversation loop using entry_app
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Update state with user input
            state["user_input"] = user_input
            
            # Check for exit
            if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
                print("\nGoodbye! 👋\n")
                break
            
            # Check for reset
            if user_input.lower() == "new":
                bot.reset()
                state["prefs"] = bot.prefs
                state["repeat_count"] = 0
                state["conversation_turn"] = 0
                state["waiting_for_feedback"] = False
                state["prefs_complete"] = False
                state["is_force_search"] = False
                print("\nai> Starting fresh! What are you looking for?")
                continue
            
            # Invoke entry app (routes to feedback or collecting based on waiting_for_feedback)
            state = entry_app.invoke(state)
            
            # Sync bot state back to state dict AND update bot's internal state
            state["prefs"] = bot.prefs
            bot.waiting_for_feedback = state.get("waiting_for_feedback", False)
            state["waiting_for_feedback"] = bot.waiting_for_feedback
            state["geo_cache"] = bot.geo_cache if bot.geo_cache else state.get("geo_cache")
            state["last_suggestions_count"] = bot.last_suggestions_count
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error in conversation loop: {e}", exc_info=True)
            print(f"\nai> Sorry, something went wrong: {e}\n")

if __name__ == "__main__":
    try:
        # Use LangGraph version
        run_graph_cli()
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