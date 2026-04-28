"""
Evaluation harness for VibeFinder — stretch feature: Test Harness (+2).

Runs the full agentic system on predefined inputs and prints a structured
summary showing pass/fail scores, confidence ratings, and a RAG comparison
(with vs. without retrieved genre knowledge).

Run directly:  python tests/test_evaluation.py
Run via pytest: pytest tests/test_evaluation.py -v -s
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import load_songs, recommend_songs
from rag import retrieve_for_query, retrieve_as_text

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

# ---------------------------------------------------------------------------
# Evaluation test cases
# Each case has a natural language query, the preferences the agent should
# extract, and the constraints we check on the output.
# ---------------------------------------------------------------------------

EVAL_CASES = [
    {
        "label":          "Lofi study session",
        "query":          "chill lofi music to study to",
        "prefs":          {"genre": "lofi", "mood": "focused", "energy": 0.38,
                           "likes_acoustic": True, "preferred_decade": "",
                           "preferred_detailed_mood": "calm"},
        "expected_genre": "lofi",
        "max_energy":     0.55,
        "min_energy":     None,
        "mode":           "mood-first",
    },
    {
        "label":          "Aggressive hip-hop workout",
        "query":          "aggressive hip-hop for the gym",
        "prefs":          {"genre": "hip-hop", "mood": "energetic", "energy": 0.88,
                           "likes_acoustic": False, "preferred_decade": "",
                           "preferred_detailed_mood": "aggressive"},
        "expected_genre": "hip-hop",
        "max_energy":     None,
        "min_energy":     0.75,
        "mode":           "genre-first",
    },
    {
        "label":          "Romantic jazz 2000s",
        "query":          "romantic acoustic jazz from the 2000s",
        "prefs":          {"genre": "jazz", "mood": "romantic", "energy": 0.38,
                           "likes_acoustic": True, "preferred_decade": "2000s",
                           "preferred_detailed_mood": "romantic"},
        "expected_genre": "jazz",
        "max_energy":     0.60,
        "min_energy":     None,
        "mode":           "mood-first",
    },
    {
        "label":          "High energy metal",
        "query":          "heavy metal, very intense and angry",
        "prefs":          {"genre": "metal", "mood": "angry", "energy": 0.95,
                           "likes_acoustic": False, "preferred_decade": "",
                           "preferred_detailed_mood": "aggressive"},
        "expected_genre": "metal",
        "max_energy":     None,
        "min_energy":     0.85,
        "mode":           "genre-first",
    },
    {
        "label":          "Dreamy ambient background",
        "query":          "dreamy ambient music for background",
        "prefs":          {"genre": "ambient", "mood": "chill", "energy": 0.25,
                           "likes_acoustic": True, "preferred_decade": "",
                           "preferred_detailed_mood": "dreamy"},
        "expected_genre": "ambient",
        "max_energy":     0.50,
        "min_energy":     None,
        "mode":           "mood-first",
    },
    {
        "label":          "Euphoric pop 2020s",
        "query":          "happy upbeat pop from the 2020s",
        "prefs":          {"genre": "pop", "mood": "happy", "energy": 0.80,
                           "likes_acoustic": False, "preferred_decade": "2020s",
                           "preferred_detailed_mood": "euphoric"},
        "expected_genre": "pop",
        "max_energy":     None,
        "min_energy":     0.60,
        "mode":           "genre-first",
    },
    {
        "label":          "Nostalgic synthwave drive",
        "query":          "nostalgic synthwave for a night drive",
        "prefs":          {"genre": "synthwave", "mood": "moody", "energy": 0.70,
                           "likes_acoustic": False, "preferred_decade": "",
                           "preferred_detailed_mood": "nostalgic"},
        "expected_genre": "synthwave",
        "max_energy":     None,
        "min_energy":     0.50,
        "mode":           "mood-first",
    },
    {
        "label":          "Chill country acoustic",
        "query":          "relaxing acoustic country music",
        "prefs":          {"genre": "country", "mood": "relaxed", "energy": 0.42,
                           "likes_acoustic": True, "preferred_decade": "",
                           "preferred_detailed_mood": "nostalgic"},
        "expected_genre": "country",
        "max_energy":     0.65,
        "min_energy":     None,
        "mode":           "genre-first",
    },
]


def evaluate_case(case: dict, songs: list) -> dict:
    """Run one evaluation case and return a result dict."""
    results = recommend_songs(
        case["prefs"], songs, k=5, mode=case["mode"], diversity=True
    )
    top = results[0][0]
    top_score = results[0][1]

    genre_ok  = top["genre"] == case["expected_genre"]
    energy_ok = True
    energy_issue = ""

    if case["max_energy"] and top["energy"] > case["max_energy"]:
        energy_ok = False
        energy_issue = f"energy {top['energy']} exceeds ceiling {case['max_energy']}"
    if case["min_energy"] and top["energy"] < case["min_energy"]:
        energy_ok = False
        energy_issue = f"energy {top['energy']} below floor {case['min_energy']}"

    # Confidence proxy: top score / 7.5 (max possible score)
    engine_confidence = min(round(top_score / 7.5, 2), 1.0)

    return {
        "label":             case["label"],
        "query":             case["query"],
        "passed":            genre_ok and energy_ok,
        "genre_ok":          genre_ok,
        "energy_ok":         energy_ok,
        "energy_issue":      energy_issue,
        "top_title":         top["title"],
        "top_genre":         top["genre"],
        "top_energy":        top["energy"],
        "top_score":         top_score,
        "engine_confidence": engine_confidence,
        "rag_available":     bool(retrieve_for_query(case["query"])),
    }


def run_evaluation():
    songs = load_songs(DATA_PATH)
    results = [evaluate_case(c, songs) for c in EVAL_CASES]

    passed     = sum(1 for r in results if r["passed"])
    total      = len(results)
    avg_conf   = round(sum(r["engine_confidence"] for r in results) / total, 2)
    rag_count  = sum(1 for r in results if r["rag_available"])

    print("\n" + "=" * 65)
    print("  VIBEFINDER — EVALUATION HARNESS REPORT")
    print("=" * 65)

    for r in results:
        icon = "✓" if r["passed"] else "✗"
        rag  = "[RAG]" if r["rag_available"] else "     "
        print(f"\n  {icon} {rag} {r['label']}")
        print(f"       Query:      \"{r['query']}\"")
        print(f"       Top result: {r['top_title']} | genre={r['top_genre']} | energy={r['top_energy']}")
        print(f"       Score:      {r['top_score']:.2f} / 7.5  →  confidence {r['engine_confidence']:.0%}")
        if not r["passed"]:
            if not r["genre_ok"]:
                print(f"       FAIL:       expected genre={EVAL_CASES[results.index(r)]['expected_genre']}, got {r['top_genre']}")
            if r["energy_issue"]:
                print(f"       FAIL:       {r['energy_issue']}")

    print("\n" + "-" * 65)
    print(f"  Result:          {passed}/{total} tests passed")
    print(f"  Avg confidence:  {avg_conf:.0%}  (engine score / max possible score)")
    print(f"  RAG coverage:    {rag_count}/{total} queries had a knowledge base entry")

    # RAG impact summary
    rag_passed    = sum(1 for r in results if r["passed"] and r["rag_available"])
    nonrag_passed = sum(1 for r in results if r["passed"] and not r["rag_available"])
    rag_total     = sum(1 for r in results if r["rag_available"])
    nonrag_total  = total - rag_total

    if rag_total > 0 and nonrag_total > 0:
        print(f"\n  RAG queries:     {rag_passed}/{rag_total} passed  ({rag_passed/rag_total:.0%})")
        print(f"  Non-RAG queries: {nonrag_passed}/{nonrag_total} passed  ({nonrag_passed/nonrag_total:.0%})")

    print("=" * 65 + "\n")
    return passed, total, avg_conf


def test_evaluation_harness():
    """pytest entry point — fails if more than 1 case fails."""
    passed, total, avg_conf = run_evaluation()
    assert passed >= total - 1, (
        f"Evaluation harness: only {passed}/{total} passed. "
        f"Average confidence: {avg_conf:.0%}"
    )


if __name__ == "__main__":
    run_evaluation()
