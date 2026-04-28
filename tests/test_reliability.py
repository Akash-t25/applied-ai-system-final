"""
Reliability tests for the VibeFinder agentic system.
Tests predefined queries against expected output constraints — no manual review needed.

Each test case defines:
  - query:          the natural language input
  - expected_genre: genre we expect to dominate the top results
  - max_energy:     energy ceiling for the results (None = no check)
  - min_energy:     energy floor for the results (None = no check)

Run with: pytest tests/test_reliability.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import load_songs, recommend_songs

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

# ---------------------------------------------------------------------------
# Test cases — simulate what the agent would extract, check engine output
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label":          "Lofi study music",
        "prefs":          {"genre": "lofi", "mood": "chill", "energy": 0.38,
                           "likes_acoustic": True, "preferred_decade": "",
                           "preferred_detailed_mood": "calm"},
        "expected_genre": "lofi",
        "max_energy":     0.55,
        "min_energy":     None,
    },
    {
        "label":          "Aggressive hip-hop",
        "prefs":          {"genre": "hip-hop", "mood": "energetic", "energy": 0.87,
                           "likes_acoustic": False, "preferred_decade": "",
                           "preferred_detailed_mood": "aggressive"},
        "expected_genre": "hip-hop",
        "max_energy":     None,
        "min_energy":     0.75,
    },
    {
        "label":          "Romantic jazz",
        "prefs":          {"genre": "jazz", "mood": "romantic", "energy": 0.40,
                           "likes_acoustic": True, "preferred_decade": "",
                           "preferred_detailed_mood": "romantic"},
        "expected_genre": "jazz",
        "max_energy":     0.60,
        "min_energy":     None,
    },
    {
        "label":          "High energy metal",
        "prefs":          {"genre": "metal", "mood": "angry", "energy": 0.95,
                           "likes_acoustic": False, "preferred_decade": "",
                           "preferred_detailed_mood": "aggressive"},
        "expected_genre": "metal",
        "max_energy":     None,
        "min_energy":     0.85,
    },
    {
        "label":          "Chill ambient",
        "prefs":          {"genre": "ambient", "mood": "chill", "energy": 0.25,
                           "likes_acoustic": True, "preferred_decade": "",
                           "preferred_detailed_mood": "dreamy"},
        "expected_genre": "ambient",
        "max_energy":     0.50,
        "min_energy":     None,
    },
    {
        "label":          "Happy pop 2020s",
        "prefs":          {"genre": "pop", "mood": "happy", "energy": 0.80,
                           "likes_acoustic": False, "preferred_decade": "2020s",
                           "preferred_detailed_mood": "euphoric"},
        "expected_genre": "pop",
        "max_energy":     None,
        "min_energy":     0.60,
    },
]


def run_reliability_tests():
    songs = load_songs(DATA_PATH)
    passed = 0
    failed = 0
    results_log = []

    print("\n" + "=" * 60)
    print("  VIBEFINDER RELIABILITY TEST SUITE")
    print("=" * 60)

    for case in TEST_CASES:
        results = recommend_songs(case["prefs"], songs, k=5, mode="genre-first", diversity=True)
        top = results[0][0]  # best scoring song

        genre_ok    = top["genre"] == case["expected_genre"]
        energy_ok   = True
        energy_note = ""

        if case["max_energy"] is not None and top["energy"] > case["max_energy"]:
            energy_ok   = False
            energy_note = f"energy {top['energy']} exceeds ceiling {case['max_energy']}"

        if case["min_energy"] is not None and top["energy"] < case["min_energy"]:
            energy_ok   = False
            energy_note = f"energy {top['energy']} below floor {case['min_energy']}"

        ok = genre_ok and energy_ok

        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            if not genre_ok:
                energy_note = f"expected genre={case['expected_genre']}, got {top['genre']}. " + energy_note

        results_log.append((case["label"], status, top["title"], top["genre"], top["energy"], energy_note))

    # Print results table
    for label, status, title, genre, energy, note in results_log:
        icon = "✓" if status == "PASS" else "✗"
        print(f"\n  {icon} [{status}] {label}")
        print(f"       Top result: {title} | genre={genre} | energy={energy}")
        if note:
            print(f"       Issue: {note}")

    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  {passed}/{total} tests passed")
    if failed == 0:
        print("  All checks passed — engine is behaving as expected.")
    else:
        print(f"  {failed} test(s) failed — review edge cases above.")
    print("=" * 60 + "\n")

    return passed, failed


# Run directly or via pytest
def test_all_reliability_cases_pass():
    passed, failed = run_reliability_tests()
    assert failed == 0, f"{failed} reliability test(s) failed — see output above."


if __name__ == "__main__":
    run_reliability_tests()
