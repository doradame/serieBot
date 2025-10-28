# 🍿 SerieBot — Intelligent TV Series Recommendation System

A sophisticated TV series recommendation chatbot powered by **LangChain 1.0**, **LangGraph**, **OVH AI Endpoints**, **TMDB API**, and **geolocation** to provide personalized, context-aware suggestions with streaming responses and intelligent state management.

## 🌟 Key Features

### Core Functionality
- **Natural Conversation Flow**: Engages users in a friendly dialogue to understand their preferences
- **Intelligent Genre Matching**: Advanced scoring system that prioritizes content most relevant to user requests
- **Freshness Bonus**: Promotes newer series to diversify suggestions and avoid repetitive recommendations
- **Geographic Awareness**: Automatically detects your location and filters by available streaming providers
- **Multi-platform Support**: Works with Netflix, Prime Video, Disney+, Apple TV+, HBO Max, and many others
- **Feedback System**: Collects user ratings to improve future recommendations

### LangChain 1.0 Innovations ✨
- **Streaming Responses**: Real-time token-by-token response generation for better UX
- **LangGraph State Management**: Structured conversation flow with automatic state transitions
- **Memory Checkpointing**: Persistent conversation state with rollback capability
- **Modular Architecture**: Clean separation of concerns with testable components
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

---

## 🏗️ Architecture Overview

SerieBot leverages **LangChain 1.0** and **LangGraph** to build a robust, maintainable, and scalable AI application. Here's how modern LangChain features transform the conversation experience:

### 1. LangGraph Conversation Flow 🗺️

The bot uses **LangGraph** to manage conversation states with clear transitions:

```
START → greeting → collecting ⇄ searching → feedback → END
                      ↓
                    exit
```

**States:**
- `greeting`: Welcome message and initial setup
- `collecting`: Gathering user preferences (genre, mood, duration, platforms, language)
- `searching`: Executing TMDB search with geo-filtering
- `feedback`: Collecting user feedback (👍/👎)
- `end`: Graceful termination

**Benefits:**
- 🧩 Modular design - each state is isolated and testable
- 🔄 Easy to extend - add new conversation paths without breaking existing logic
- 💾 Automatic checkpointing - conversation state persists across sessions
- 📊 Visual debugging - graph structure is self-documenting

### 2. Streaming Responses 🌊

All LLM responses now stream token-by-token for better user experience:

```python
# Streaming implementation
for chunk in chain.stream(input):
    print(chunk.content, end="", flush=True)
```

Users see the bot "thinking" in real-time, creating a more interactive experience.

### 3. Orchestrating the Logic (LCEL) 🔗

The bot uses **LangChain Expression Language (LCEL)** to create powerful chains using the `|` (pipe) symbol.

#### **Preference Extraction Chain**
This pipeline takes the user's input, processes it through a ChatPromptTemplate, sends it to the LLM, and then forces the model's text answer into a structured `CollectResult` object using the `PydanticOutputParser`.

```python
# This is LCEL in action: Prompt | Model | Parser
collect_chain = (
    collect_prompt.partial(
        genres=", ".join(sorted(ALLOWED_GENRES)),
        moods=", ".join(sorted(ALLOWED_MOOD)),
        current_genre=self.prefs.genre or "null",
        current_mood=self.prefs.mood or "null",
        # ... other context
    )
    | llm_base
    | collect_parser
)
```

**What's happening here:**
1. `collect_prompt.partial(...)` injects dynamic context (available options, current state)
2. `| llm_base` sends the prompt to the AI model
3. `| collect_parser` transforms the LLM's text output into a structured Python object

#### **Final Response Chain**
A simpler chain that takes the search results and user preferences, formats them into a prompt, and generates a friendly, natural response.

```python
final_chain = final_prompt | llm_base
```

This orchestration makes the code **declarative and easy to read**. You're defining the **flow of data** rather than writing complex procedural code with nested if-statements and error handling.

---

### 2. Structuring Communication with the LLM 🤖

This is perhaps the **most critical role** of LangChain in SerieBot. LLMs naturally produce unstructured text, which is hard for a program to work with. LangChain solves this in two powerful ways:

#### **ChatPromptTemplate: Dynamic Context Injection**

Instead of sending static strings to the LLM, `ChatPromptTemplate` allows you to create complex and dynamic prompts. You programmatically inject:
- The list of allowed genres, moods, durations, languages
- The **current state of the conversation** (what you already know)
- Context about what's missing and what to ask next

```python
collect_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly TV concierge...\n"
     "WHAT YOU ALREADY KNOW:\n"
     "- Genre: {current_genre}\n"
     "- Mood: {current_mood}\n"
     "- Duration: {current_duration}\n"
     # ... the LLM gets perfect context every time
    ),
    ("human", "{user_input}")
])
```

This gives the LLM the **context it needs** to act intelligently without repeating questions or losing track of the conversation.

#### **PydanticOutputParser: Turning Text into Structure**

This is the **magic ingredient**. It takes the text output from the LLM and forces it to fit the structure of your Pydantic class (`CollectResult`). 

```python
class CollectResult(BaseModel):
    known: UserPrefs = Field(default_factory=UserPrefs)
    ask_next: Optional[str] = None

collect_parser = PydanticOutputParser(pydantic_object=CollectResult)
```

**What this means:**
- If the LLM produces a response that doesn't match the expected JSON structure, the parser will raise an error
- This turns the LLM's **unpredictable output** into a **predictable and structured Python object**
- Your code can safely access `result.known.genre` without worrying about KeyErrors or missing fields

This is what transforms an LLM from a "chatbot" into a **reliable component** of your application.

---

### 3. Defining Tools for the AI 🛠️

The `@tool` decorator labels Python functions as tools that can be used by LangChain agents:

```python
@tool
def ipinfo_location() -> str:
    """Get user's location (city, country, timezone) using ipinfo.io."""
    # Implementation...

@tool
def suggest_series(genre: str, mood: str, duration: str, ...) -> str:
    """Return up to 3 series tailored to user preferences."""
    # Implementation...
```

**Why this matters:**
- While currently these functions are called manually from Python code, the `@tool` decorator prepares them for use by **LangChain agents**
- An agent is a more advanced type of chain that can **decide for itself** which tool to call and when
- This decorator provides **automatic documentation** and **type validation** for the LLM
- It's part of a standard LangChain pattern that makes your code future-proof and extensible

**Potential Evolution:**
Instead of manually calling `ipinfo_location()` and `suggest_series()`, you could upgrade to an agent that automatically decides:
> "The user wants recommendations → I need their location → call `ipinfo_location` → now I need series data → call `suggest_series` with the extracted preferences"

---

### 4. Abstracting the Model Connection 🔌

`ChatOpenAI` provides a standardized, high-level interface for connecting to AI models:

```python
llm_base = ChatOpenAI(
    openai_api_base=OVH_BASE,  # OVH AI Endpoints
    openai_api_key=OVH_KEY,
    temperature=0.0,            # Deterministic extraction
    model="gpt-oss-120b"
)
```

**Benefits:**
- You don't have to write custom `requests` code to handle API calls, headers, and authentication
- If you wanted to switch from OVH to OpenAI, Azure, or Anthropic, you would only need to change a few lines of configuration
- The rest of your code (`collect_chain`, `final_chain`, etc.) would work **exactly the same**
- LangChain handles rate limiting, retries, and error handling automatically

**In short, LangChain provides the essential building blocks that allow you to move from simply "chatting" with an AI to building a robust, stateful, and reliable application on top of it.**

---

## 🎯 Intelligent Recommendation Algorithm

### Genre Matching Score

For sci-fi requests, the system applies a sophisticated scoring algorithm to prioritize authentic sci-fi content over fantasy/supernatural series:

```python
def _score_genre_match(tv_genres, requested_genre):
    """Score series by genre relevance."""
    score = 0
    
    # Positive indicators (true sci-fi)
    if 10765 in tv_genres:  # Sci-Fi & Fantasy base
        score += 1
    if 10759 in tv_genres:  # Action & Adventure
        score += 2
    if 9648 in tv_genres:   # Mystery
        score += 1
    
    # Negative indicators (fantasy/supernatural)
    if 18 in tv_genres:     # Drama (common in vampire shows)
        score -= 1
    if 80 in tv_genres:     # Crime (supernatural crime ≠ sci-fi)
        score -= 2
    
    return score
```

**Problem Solved**: TMDB categorizes many fantasy/vampire series (like "The Vampire Diaries") under the same "Sci-Fi & Fantasy" genre ID (10765) as true sci-fi shows. This scoring system ensures that when you ask for sci-fi, you get Star Trek, not Twilight.

### Freshness Bonus

To avoid repetitive suggestions and promote variety, newer series receive a bonus:

```python
def _score_freshness(first_air_date):
    """Bonus for newer content to diversify recommendations."""
    age = current_year - release_year
    
    if age <= 2:    return 3  # Brand new (2023-2025)
    elif age <= 5:  return 2  # Recent (2020-2022)
    elif age <= 10: return 1  # Modern (2015-2019)
    elif age <= 20: return 0  # Classic (2005-2014)
    else:           return -1 # Vintage (pre-2005)
```

**Final Score**: `total_score = genre_score + freshness_score + vote_rating`

This ensures a balance between:
- **Relevance** (genre matching)
- **Variety** (promoting newer content)
- **Quality** (TMDB ratings as tiebreaker)

---

## � Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or higher** installed on your system
- **pip** (Python package installer)
- **Git** (optional, for cloning the repository)
- API Keys (we'll get these in step 4):
  - **OVH AI Endpoints** API key
  - **TMDB** (The Movie Database) API key

### Quick Start Guide

Follow these steps to get SerieBot up and running:

#### 1. Clone or Download the Repository

```bash
# Using Git
git clone <repository-url>
cd serieBot

# Or download and extract the ZIP file, then navigate to the directory
cd serieBot
```

#### 2. Create and Activate Virtual Environment

**On macOS/Linux:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

#### 3. Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- LangChain (0.2.16) - For AI orchestration
- Pydantic (2.8.2) - For data validation
- Requests - For HTTP calls
- python-dotenv - For environment variables

#### 4. Get Your API Keys

##### TMDB API Key (Required)

1. Go to [TMDB](https://www.themoviedb.org/) and create a free account
2. Navigate to Settings → API
3. Click "Create" → Choose "Developer"
4. Fill out the form (select "Personal" or "Educational" use)
5. Accept the terms
6. Copy your **API Key (v3 auth)** (NOT the API Read Access Token)

##### OVH AI Endpoints API Key (Required)

1. Visit [OVH AI Endpoints](https://endpoints.ai.cloud.ovh.net/)
2. Create an OVH account if you don't have one
3. Go to the AI Endpoints dashboard
4. Generate a new API token/key
5. Copy your API key

#### 5. Configure Environment Variables

Copy the template file and add your API keys:

```bash
cp .env.template .env
```

Now edit the `.env` file with your favorite text editor:

```bash
# On macOS/Linux
nano .env

# Or use any text editor
code .env  # VS Code
vim .env   # Vim
```

Fill in your API keys:

```env
# TMDB API Key - Get from https://www.themoviedb.org/settings/api
TMDB_API_KEY=your_actual_tmdb_api_key_here

# OVH AI Endpoints Key - Get from https://endpoints.ai.cloud.ovh.net/
OPENAI_API_KEY=your_actual_ovh_api_key_here

# OVH Configuration (usually no need to change)
OPENAI_API_BASE=https://gpt-oss-120b.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1
OVH_MODEL=gpt-oss-120b

# Optional Settings
DEBUG=false
DEFAULT_LANGUAGE=en
REQUEST_TIMEOUT=10
TMDB_RATE_LIMIT=0.25
```

**Important:** Replace `your_actual_tmdb_api_key_here` and `your_actual_ovh_api_key_here` with your real API keys!

#### 6. Run SerieBot

Start the application:

```bash
python serieBot.py
```

You should see:

```text
🍿 SerieBot — Geo-located TV Suggestions (OVH + LangChain + ipinfo + TMDB)
Type 'exit' to quit, 'new' to start over, or 'search' to search with current preferences.
💡 Tip: For duration, use '<30m' for short, '~45m' for standard, or '>60m' for long episodes.

Hi! 👋 What kind of TV series are you looking for?
You can tell me about genre, mood, duration, language, or platforms (e.g., Netflix, Prime)...
>
```

#### 7. Start Chatting!

Try your first query:

```text
> mind blowing sci-fi on netflix, any language, any duration

ai> Got it! Still need: mood.
How are you feeling today - something light or more intense?

> intense

ai> Perfect! Let me find the best matches for you...
```

### Troubleshooting

**Problem:** `ModuleNotFoundError` when running the script

**Solution:** Make sure your virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

**Problem:** `Missing OPENAI_API_KEY` error

**Solution:** Check that your `.env` file exists and contains valid API keys. The file should be in the same directory as `serieBot.py`.

---

**Problem:** `TMDB API error` messages

**Solution:** Verify your TMDB API key is correct and you're using the API Key (v3), not the Read Access Token (v4).

---

**Problem:** No suggestions found

**Solution:** The streaming service might not be available in your region, or there might be no content matching all your filters. Try:
- Broadening your search (e.g., remove duration filter)
- Trying different providers
- Checking if the provider name is spelled correctly

### Deactivating the Virtual Environment

When you're done using SerieBot:

```bash
deactivate
```

This returns you to your system's default Python environment.

---

## 📚 Additional Configuration

## 🎮 Usage

### Basic Conversation

```text
🍿 SerieBot — Geo-located TV Suggestions (OVH + LangChain + ipinfo + TMDB)

Hi! 👋 What kind of TV series are you looking for?
> mind blowing sci-fi on netflix, any duration, any language

ai> Got it! Still need: mood.
How are you feeling today - something light or more intense?
> intense

ai> Perfect! Let me find the best matches for you...

ai> **Van Helsing** – A post-apocalyptic sci-fi thriller where humanity battles
vampire overlords, delivering mind-blowing twists and action-packed 45-minute
episodes. Available on Netflix in Italy.

**Star Trek: The Next Generation** – Classic sci-fi exploration with
philosophical dilemmas and spectacular space adventures that constantly surprise
and expand the mind, each episode runs about 45 minutes. Available on Netflix
in Italy.

**Star Trek: Deep Space Nine** – A gritty sci-fi series set on a space station
near a wormhole, offering complex storylines and mind-bending plot twists, with
episodes around 45 minutes long. Available on Netflix in Italy.
```

### Commands

- **`exit`** or **`quit`**: Exit the application
- **`new`**: Start a fresh search (reset all preferences)
- **`search`**: Force search with current preferences
- **`skip`**: Skip feedback after suggestions

### Preference Format

- **Genre**: `sci-fi`, `comedy`, `drama`, `crime`, `animation`, `fantasy`, `mystery`, `documentary`
- **Mood**: `light-hearted`, `intense`, `mind-blowing`, `comforting`, `adrenaline-fueled`
- **Duration**:
  - `<30m` (short episodes)
  - `~45m` (standard)
  - `>60m` (long episodes)
  - Or use natural language: `45 minutes`, `short`, `long`
- **Language**: `en`, `it`, `any`
- **Providers**: `Netflix`, `Prime Video`, `Disney+`, `Apple TV+`, `HBO Max`, etc.

---

## 🗂️ Project Structure

```text
serieBot/
├── serieBot.py           # Main application
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create from template)
├── .env.template         # Template for environment variables
├── README.md             # This file
├── REFACTORING.md        # Documentation of BotSession refactoring
├── logs/                 # Application logs (auto-created)
│   └── seriebot_*.log
└── feedback/             # User feedback data (auto-created)
    └── feedback.csv
```

### Key Components

- **`BotSession` class**: Manages conversation state and logic
  - `extract_preferences()`: Extracts user preferences using LCEL chain
  - `perform_search()`: Executes TMDB search with intelligent scoring
  - `handle_feedback()`: Processes user ratings (👍/👎)
  - `is_informational_question()`: Distinguishes questions from search queries

- **LCEL Chains**:
  - `collect_chain`: Prompt → LLM → PydanticParser → Structured preferences
  - `final_chain`: Prompt → LLM → Natural language response

- **Tools**:
  - `ipinfo_location`: Geolocation detection
  - `suggest_series`: TMDB search with scoring algorithm

- **Scoring System**:
  - `_score_genre_match()`: Genre relevance scoring
  - `_score_freshness()`: Recency bonus
  - Final ranking: `(genre_score + freshness_score, rating)`

---

## 📊 Feedback System

User feedback is automatically saved to `feedback/feedback.csv` with:

- Timestamp
- Session ID
- Rating (positive/negative)
- All preferences (genre, mood, duration, providers, language)
- Country
- Number of suggestions shown
- Optional user comment

This data can be used to:

- Analyze which combinations work best
- Identify gaps in the recommendation algorithm
- Track user satisfaction over time

---

## 🔍 Logging

SerieBot uses a dual-handler logging system:

- **File logs** (`logs/seriebot_*.log`): DEBUG level, full context
  - Detailed preference extraction
  - API calls and responses
  - Scoring calculations
  - Error stack traces

- **Console logs**: WARNING level only
  - Critical errors
  - API failures
  - Infinite loop detection

Enable debug console output by setting `DEBUG=true` in `.env`.

## 📄 License

This project is provided as-is for educational purposes.

---

## 🙏 Acknowledgments

- **LangChain**: For providing the orchestration framework
- **OVH AI Endpoints**: For LLM inference
- **TMDB**: For comprehensive TV series data
- **ipinfo.io**: For geolocation services

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using LangChain, Python, and AI**

