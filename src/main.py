"""
Command line runner for the Music Recommender Simulation.
Runs all scoring modes and diversity options for evaluation.
"""

from recommender import load_songs, recommend_songs, SCORING_MODES

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def print_recommendations(
    label: str,
    user_prefs: dict,
    songs: list,
    k: int = 5,
    mode: str = "genre-first",
    diversity: bool = False,
) -> None:
    """Prints a formatted recommendation table for a given profile and mode."""
    recommendations = recommend_songs(user_prefs, songs, k=k, mode=mode, diversity=diversity)

    header = f"  Profile: {label}  |  Mode: {mode}  |  Diversity: {diversity}"
    prefs_line = (
        f"  genre={user_prefs['genre']} | mood={user_prefs['mood']} | "
        f"energy={user_prefs['energy']} | acoustic={user_prefs['likes_acoustic']}"
    )
    if user_prefs.get("preferred_decade"):
        prefs_line += f" | decade={user_prefs['preferred_decade']}"
    if user_prefs.get("preferred_detailed_mood"):
        prefs_line += f" | vibe={user_prefs['preferred_detailed_mood']}"

    print("\n" + "=" * 70)
    print(header)
    print(prefs_line)
    print("=" * 70)

    # Challenge 4 — tabulate output
    if HAS_TABULATE:
        rows = []
        for i, (song, score, explanation) in enumerate(recommendations, start=1):
            rows.append([
                f"#{i}",
                song["title"],
                song["artist"],
                song["genre"],
                song["mood"],
                song["energy"],
                song.get("popularity", "—"),
                song.get("release_decade", "—"),
                f"{score:.2f}",
                explanation[:60] + "..." if len(explanation) > 60 else explanation,
            ])
        headers = ["#", "Title", "Artist", "Genre", "Mood", "Energy", "Pop", "Era", "Score", "Why"]
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        # ASCII fallback if tabulate not installed
        for i, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"\n  #{i}  {song['title']} — {song['artist']}")
            print(f"       {song['genre']} | {song['mood']} | energy {song['energy']} | pop {song.get('popularity','?')} | {song.get('release_decade','')}")
            print(f"       Score: {score:.2f}")
            print(f"       Why:   {explanation}")
    print()


def main() -> None:
    songs = load_songs("data/songs.csv")

    # -----------------------------------------------------------------------
    # Base profiles
    # -----------------------------------------------------------------------
    profiles = [
        ("High-Energy Hip-Hop", {
            "genre": "hip-hop", "mood": "energetic", "energy": 0.85,
            "likes_acoustic": False, "preferred_decade": "2010s",
            "preferred_detailed_mood": "aggressive",
        }),
        ("Chill Lofi", {
            "genre": "lofi", "mood": "chill", "energy": 0.38,
            "likes_acoustic": True, "preferred_decade": "2020s",
            "preferred_detailed_mood": "nostalgic",
        }),
        ("Intense Rock", {
            "genre": "rock", "mood": "intense", "energy": 0.90,
            "likes_acoustic": False, "preferred_decade": "2010s",
            "preferred_detailed_mood": "aggressive",
        }),
    ]

    # -----------------------------------------------------------------------
    # Challenge 2 — run each profile in all 3 scoring modes
    # -----------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("  CHALLENGE 2: Scoring Modes Comparison")
    print("#" * 70)

    for label, prefs in profiles:
        for mode in SCORING_MODES:
            print_recommendations(label, prefs, songs, k=5, mode=mode)

    # -----------------------------------------------------------------------
    # Challenge 3 — diversity penalty ON vs OFF
    # -----------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("  CHALLENGE 3: Diversity Penalty ON vs OFF")
    print("#" * 70)

    high_energy_prefs = {
        "genre": "hip-hop", "mood": "energetic", "energy": 0.85,
        "likes_acoustic": False, "preferred_decade": "2010s",
        "preferred_detailed_mood": "aggressive",
    }
    print_recommendations("High-Energy Hip-Hop [no diversity]", high_energy_prefs, songs, mode="genre-first", diversity=False)
    print_recommendations("High-Energy Hip-Hop [diversity ON]", high_energy_prefs, songs, mode="genre-first", diversity=True)

    # -----------------------------------------------------------------------
    # Challenge 1 — new features in action: decade + detailed_mood
    # -----------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("  CHALLENGE 1: New Features — Decade + Detailed Mood")
    print("#" * 70)

    nostalgic_2000s = {
        "genre": "jazz", "mood": "relaxed", "energy": 0.35,
        "likes_acoustic": True, "preferred_decade": "2000s",
        "preferred_detailed_mood": "nostalgic",
    }
    print_recommendations("Nostalgic 2000s Jazz", nostalgic_2000s, songs, mode="mood-first")

    euphoric_2020s = {
        "genre": "pop", "mood": "happy", "energy": 0.80,
        "likes_acoustic": False, "preferred_decade": "2020s",
        "preferred_detailed_mood": "euphoric",
    }
    print_recommendations("Euphoric 2020s Pop", euphoric_2020s, songs, mode="mood-first")


if __name__ == "__main__":
    main()
