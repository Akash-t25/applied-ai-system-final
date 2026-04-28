"""
Few-shot specialization comparison — stretch feature evidence.

Demonstrates that injecting few-shot examples into the PLAN prompt
produces measurably more consistent and accurate output than the
baseline (no examples). Runs three queries through both approaches
and compares energy accuracy and format consistency.

Run with: python3 tests/test_fewshot_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

VALID_GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave",
    "indie pop", "r&b", "hip-hop", "classical", "country", "electronic", "metal"
]
VALID_MOODS = [
    "happy", "chill", "intense", "relaxed", "focused",
    "moody", "sad", "romantic", "energetic", "angry"
]

# The four few-shot examples used in production
FEW_SHOT = """
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

# Test queries with known correct energy ranges
QUERIES = [
    {
        "query":           "soft ambient background music for sleeping",
        "expected_genre":  "ambient",
        "energy_floor":    0.0,
        "energy_ceiling":  0.45,
        "expected_mode":   "mood-first",
    },
    {
        "query":           "heavy aggressive metal for working out",
        "expected_genre":  "metal",
        "energy_floor":    0.85,
        "energy_ceiling":  1.0,
        "expected_mode":   "genre-first",
    },
    {
        "query":           "upbeat happy pop from the 2020s",
        "expected_genre":  "pop",
        "energy_floor":    0.60,
        "energy_ceiling":  1.0,
        "expected_mode":   "genre-first",
    },
]

SCHEMA = '{"genre":"...","mood":"...","energy":0.00,"likes_acoustic":true,"preferred_decade":"","preferred_detailed_mood":"","scoring_mode":"genre-first"}'


def run_plan(query: str, use_fewshot: bool) -> dict:
    """Run the PLAN step with or without few-shot examples."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    examples_section = FEW_SHOT if use_fewshot else ""

    prompt = f"""You are a music preference extractor.
{examples_section}
Extract preferences for: "{query}"

Available genres: {VALID_GENRES}
Available moods: {VALID_MOODS}

Return JSON matching this schema exactly: {SCHEMA}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)


def check(prefs: dict, case: dict) -> dict:
    """Check whether extracted preferences meet the expected constraints."""
    genre_ok  = prefs.get("genre") == case["expected_genre"]
    energy_ok = case["energy_floor"] <= prefs.get("energy", 0) <= case["energy_ceiling"]
    mode_ok   = prefs.get("scoring_mode") == case["expected_mode"]
    has_all   = all(k in prefs for k in ["genre", "mood", "energy", "likes_acoustic",
                                          "preferred_decade", "preferred_detailed_mood",
                                          "scoring_mode"])
    return {
        "genre_ok":  genre_ok,
        "energy_ok": energy_ok,
        "mode_ok":   mode_ok,
        "has_all":   has_all,
        "score":     sum([genre_ok, energy_ok, mode_ok, has_all]),
    }


def run_comparison():
    print("\n" + "=" * 65)
    print("  FEW-SHOT SPECIALIZATION — BASELINE vs. PRODUCTION COMPARISON")
    print("=" * 65)

    baseline_total = 0
    fewshot_total  = 0
    max_total      = len(QUERIES) * 4  # 4 checks per query

    for case in QUERIES:
        print(f"\n  Query: \"{case['query']}\"")
        print(f"  Expected: genre={case['expected_genre']} · energy {case['energy_floor']}–{case['energy_ceiling']} · mode={case['expected_mode']}")

        baseline = run_plan(case["query"], use_fewshot=False)
        fewshot  = run_plan(case["query"], use_fewshot=True)

        b = check(baseline, case)
        f = check(fewshot,  case)

        baseline_total += b["score"]
        fewshot_total  += f["score"]

        print(f"\n  WITHOUT few-shot:  genre={baseline.get('genre')} · energy={baseline.get('energy')} · mode={baseline.get('scoring_mode')}")
        print(f"    genre ✓/✗: {'✓' if b['genre_ok'] else '✗'}  energy ✓/✗: {'✓' if b['energy_ok'] else '✗'}  mode ✓/✗: {'✓' if b['mode_ok'] else '✗'}  all fields: {'✓' if b['has_all'] else '✗'}  [{b['score']}/4]")

        print(f"\n  WITH few-shot:     genre={fewshot.get('genre')} · energy={fewshot.get('energy')} · mode={fewshot.get('scoring_mode')}")
        print(f"    genre ✓/✗: {'✓' if f['genre_ok'] else '✗'}  energy ✓/✗: {'✓' if f['energy_ok'] else '✗'}  mode ✓/✗: {'✓' if f['mode_ok'] else '✗'}  all fields: {'✓' if f['has_all'] else '✗'}  [{f['score']}/4]")

    print("\n" + "-" * 65)
    print(f"  Baseline total:      {baseline_total}/{max_total} checks passed  ({baseline_total/max_total:.0%})")
    print(f"  Few-shot total:      {fewshot_total}/{max_total} checks passed  ({fewshot_total/max_total:.0%})")
    improvement = fewshot_total - baseline_total
    print(f"  Improvement:        +{improvement} checks  ({improvement/max_total:.0%} gain)")
    print("=" * 65 + "\n")

    return baseline_total, fewshot_total


def test_fewshot_outperforms_baseline():
    """pytest entry point — few-shot must score at least as well as baseline."""
    baseline, fewshot = run_comparison()
    assert fewshot >= baseline, (
        f"Few-shot ({fewshot}) should score >= baseline ({baseline})"
    )


if __name__ == "__main__":
    run_comparison()
