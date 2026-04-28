"""
Groq-powered agentic layer for VibeFinder.

Agentic loop (observable intermediate steps):
  1. CLASSIFY  — determine what type of request this is
  2. RETRIEVE  — fetch genre knowledge from the RAG knowledge base
  3. PLAN      — extract structured preferences (few-shot + RAG-grounded)
  4. ACT       — score all songs with the recommender engine
  5. REFLECT   — evaluate result quality + confidence score
  6. RETRY     — adjust and loop back if quality is poor (max 2x)
"""

import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv
from rag import retrieve_for_query

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"

VALID_GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave",
    "indie pop", "r&b", "hip-hop", "classical", "country", "electronic", "metal"
]
VALID_MOODS = [
    "happy", "chill", "intense", "relaxed", "focused",
    "moody", "sad", "romantic", "energetic", "angry"
]
VALID_DETAILED_MOODS = [
    "euphoric", "nostalgic", "aggressive", "dreamy", "calm", "melancholic", "romantic"
]
VALID_DECADES = ["2000s", "2010s", "2020s"]

# ---------------------------------------------------------------------------
# Few-shot examples for the PLAN step (Fine-Tuning / Specialization)
# These ground the model's output format and reasoning style.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """
Examples of good preference extraction:

Input:  "I want something chill and nostalgic for late night studying"
Output: {"genre": "lofi", "mood": "focused", "energy": 0.35, "likes_acoustic": true, "preferred_decade": "", "preferred_detailed_mood": "nostalgic", "scoring_mode": "mood-first"}

Input:  "give me aggressive hip hop to hype me up at the gym"
Output: {"genre": "hip-hop", "mood": "energetic", "energy": 0.88, "likes_acoustic": false, "preferred_decade": "", "preferred_detailed_mood": "aggressive", "scoring_mode": "genre-first"}

Input:  "something romantic and jazzy from the 2000s, acoustic vibes"
Output: {"genre": "jazz", "mood": "romantic", "energy": 0.38, "likes_acoustic": true, "preferred_decade": "2000s", "preferred_detailed_mood": "romantic", "scoring_mode": "mood-first"}

Input:  "I need intense electronic music to focus while coding"
Output: {"genre": "electronic", "mood": "focused", "energy": 0.82, "likes_acoustic": false, "preferred_decade": "", "preferred_detailed_mood": "aggressive", "scoring_mode": "energy-focused"}
"""


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Step 1 — CLASSIFY
# ---------------------------------------------------------------------------

def classify(query: str) -> dict:
    """
    Classify the user's request before planning.
    Determines the primary intent: genre-driven, mood-driven, activity-driven, or era-driven.
    This is the first observable reasoning step in the agentic chain.
    """
    client = _get_client()

    prompt = f"""You are classifying a music request to determine what the user cares about most.

User request: "{query}"

Classify the request by choosing ONE primary intent:
- "genre-driven"    : user explicitly names a genre (lofi, jazz, metal, etc.)
- "mood-driven"     : user describes a feeling or vibe (chill, sad, hype, romantic, etc.)
- "activity-driven" : user describes an activity (studying, gym, sleeping, driving, etc.)
- "era-driven"      : user mentions a decade or time period

Also identify any genre or mood keywords you detect.

Return a JSON object:
{{
  "primary_intent": "genre-driven",
  "detected_genre": "lofi",
  "detected_mood": "chill",
  "detected_activity": "studying",
  "detected_era": "",
  "reasoning": "one sentence explaining your classification"
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Step 2 + 3 — RETRIEVE + PLAN (RAG-grounded, few-shot)
# ---------------------------------------------------------------------------

def plan(query: str, classification: dict) -> dict:
    """
    PLAN step — translate a natural language query into structured preferences.
    Augmented with:
      - RAG: retrieved genre knowledge injected into the prompt
      - Few-shot examples: 4 labeled input/output pairs for consistency
    """
    client = _get_client()

    detected_genre = classification.get("detected_genre", "")
    rag_context = retrieve_for_query(query)
    if not rag_context and detected_genre:
        from rag import retrieve_as_text
        rag_context = retrieve_as_text(detected_genre)

    rag_section = f"\nRetrieved genre knowledge (use this to set accurate energy and mood values):\n{rag_context}\n" if rag_context else ""

    prompt = f"""You are a music preference extractor. Given a user's music request, extract structured preferences.
{rag_section}
{FEW_SHOT_EXAMPLES}
Now extract preferences for this new request:

User request: "{query}"
Classification result: {classification}

Available genres (pick the single closest one): {VALID_GENRES}
Available moods (pick the single closest one): {VALID_MOODS}
Available detailed moods (optional): {VALID_DETAILED_MOODS}
Available decades (optional, only if user mentions an era): {VALID_DECADES}
Scoring modes:
  - "genre-first"    -> user strongly wants a specific genre
  - "mood-first"     -> user describes a vibe or emotion
  - "energy-focused" -> user describes an intensity level

Use the retrieved genre knowledge above to set an accurate energy value.
Return a JSON object with exactly these fields:
{{
  "genre": "one value from the valid genres list",
  "mood": "one value from the valid moods list",
  "energy": 0.00,
  "likes_acoustic": true,
  "preferred_decade": "",
  "preferred_detailed_mood": "",
  "scoring_mode": "genre-first"
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Step 5 — REFLECT (with confidence scoring)
# ---------------------------------------------------------------------------

def reflect(query: str, preferences: dict, results: list) -> tuple:
    """
    REFLECT step — check whether the top results actually match the user's request.
    Returns (is_good_match: bool, confidence: float, reason: str, suggested_adjustment: str).
    """
    client = _get_client()

    results_summary = "\n".join([
        f"  #{i+1} {r[0]['title']} | genre={r[0]['genre']} | mood={r[0]['mood']} | energy={r[0]['energy']}"
        for i, r in enumerate(results)
    ])

    prompt = f"""You are a music recommendation quality checker.

Original user request: "{query}"
Preferences extracted: {preferences}

Top 5 results:
{results_summary}

Check for obvious mismatches such as:
- Wrong genre (user asked for lofi, got metal)
- Energy way off (user asked for chill but results have energy above 0.80)
- Wrong mood (user asked for happy, got sad songs)

Return a JSON object:
{{
  "is_good_match": true,
  "confidence": 0.95,
  "reason": "brief explanation",
  "suggested_adjustment": ""
}}

confidence is a float from 0.0 to 1.0 — how well the results match the request.
  1.0 = perfect match, 0.5 = mediocre, 0.0 = completely wrong.
If is_good_match is false, fill suggested_adjustment with a short instruction like
"lower energy to 0.35, prefer mood=relaxed" so the planner can correct itself."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    result = json.loads(response.choices[0].message.content)
    return (
        result.get("is_good_match", True),
        result.get("confidence", 1.0),
        result.get("reason", ""),
        result.get("suggested_adjustment", ""),
    )


# ---------------------------------------------------------------------------
# Full agentic loop
# ---------------------------------------------------------------------------

def run_agent(query: str, songs: list, max_retries: int = 2) -> tuple:
    """
    Full agentic loop with observable intermediate steps:
      CLASSIFY -> RETRIEVE -> PLAN -> ACT -> REFLECT -> (RETRY if needed)

    Returns:
        results: list of (song_dict, score, explanation) tuples
        log:     list of (step_label, message) tuples for display in the UI
    """
    from recommender import recommend_songs

    log = []
    results = []
    current_query = query

    for attempt in range(max_retries + 1):

        # --- CLASSIFY ---
        try:
            classification = classify(current_query)
        except Exception as e:
            logger.error(f"[CLASSIFY] Failed: {e}")
            classification = {"primary_intent": "genre-driven", "detected_genre": "", "detected_mood": ""}
            log.append(("CLASSIFY", f"Classification failed, proceeding with defaults: {e}"))
        else:
            classify_msg = (
                f"intent={classification.get('primary_intent')} · "
                f"genre={classification.get('detected_genre') or 'none'} · "
                f"mood={classification.get('detected_mood') or 'none'} · "
                f"activity={classification.get('detected_activity') or 'none'}"
            )
            log.append(("CLASSIFY", classify_msg))
            logger.info(f"[CLASSIFY] {classify_msg}")

        # --- RETRIEVE (RAG) ---
        detected_genre = classification.get("detected_genre", "")
        rag_doc = retrieve_for_query(current_query)
        if rag_doc:
            log.append(("RETRIEVE", f"Found knowledge base entry for '{detected_genre}'"))
            logger.info(f"[RETRIEVE] Loaded genre knowledge for '{detected_genre}'")
        else:
            log.append(("RETRIEVE", "No exact genre match in knowledge base — using model knowledge"))
            logger.info("[RETRIEVE] No RAG document found, relying on model training")

        # --- PLAN ---
        try:
            prefs = plan(current_query, classification)
        except Exception as e:
            logger.error(f"[PLAN] Failed: {e}")
            log.append(("ERROR", f"Could not extract preferences: {e}"))
            break

        plan_msg = (
            f"genre={prefs.get('genre')} · mood={prefs.get('mood')} · "
            f"energy={prefs.get('energy')} · acoustic={prefs.get('likes_acoustic')} · "
            f"decade={prefs.get('preferred_decade') or 'any'} · "
            f"vibe={prefs.get('preferred_detailed_mood') or 'any'} · "
            f"mode={prefs.get('scoring_mode', 'genre-first')}"
        )
        log.append(("PLAN", plan_msg))
        logger.info(f"[PLAN]     {plan_msg}")

        # --- ACT ---
        try:
            results = recommend_songs(
                prefs,
                songs,
                k=5,
                mode=prefs.get("scoring_mode", "genre-first"),
                diversity=True,
            )
        except Exception as e:
            logger.error(f"[ACT] Failed: {e}")
            log.append(("ERROR", f"Recommender engine error: {e}"))
            break

        act_msg = f"Scored {len(songs)} songs, retrieved top 5"
        log.append(("ACT", act_msg))
        logger.info(f"[ACT]      {act_msg}")

        # --- REFLECT ---
        try:
            is_good, confidence, reason, adjustment = reflect(query, prefs, results)
        except Exception as e:
            logger.error(f"[REFLECT] Failed: {e}")
            log.append(("REFLECT", "Could not verify quality — returning best available results"))
            confidence = None
            break

        reflect_msg = f"{reason} (confidence: {confidence:.0%})"
        log.append(("REFLECT", reflect_msg))
        log.append(("CONFIDENCE", confidence))
        logger.info(f"[REFLECT]  {'Good' if is_good else 'Poor'} match — {reflect_msg}")

        if is_good:
            log.append(("DONE", "Results approved."))
            return results, log

        # --- RETRY ---
        if attempt < max_retries:
            retry_msg = f"Adjusting and retrying ({attempt + 1}/{max_retries})..."
            log.append(("RETRY", retry_msg))
            logger.info(f"[RETRY]    {retry_msg}")
            current_query = f"{query}. Correction needed: {adjustment}"

    log.append(("DONE", "Returning best available results."))
    return results, log
