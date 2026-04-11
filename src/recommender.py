from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: int = 0
    release_decade: str = "2020s"
    detailed_mood: str = ""

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    preferred_decade: str = ""
    preferred_detailed_mood: str = ""

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Scores all songs against the user profile and returns the top k Songs."""
        user_prefs = {
            "genre":                  user.favorite_genre,
            "mood":                   user.favorite_mood,
            "energy":                 user.target_energy,
            "likes_acoustic":         user.likes_acoustic,
            "preferred_decade":       user.preferred_decade,
            "preferred_detailed_mood": user.preferred_detailed_mood,
        }
        song_dicts = [s.__dict__ for s in self.songs]
        ranked = recommend_songs(user_prefs, song_dicts, k)
        top_ids = {rec[0]["id"] for rec in ranked}
        return [s for s in self.songs if s.id in top_ids]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a human-readable explanation of why a song was recommended."""
        user_prefs = {
            "genre":                  user.favorite_genre,
            "mood":                   user.favorite_mood,
            "energy":                 user.target_energy,
            "likes_acoustic":         user.likes_acoustic,
            "preferred_decade":       user.preferred_decade,
            "preferred_detailed_mood": user.preferred_detailed_mood,
        }
        _, reasons = score_song(user_prefs, song.__dict__)
        return ", ".join(reasons)


# ---------------------------------------------------------------------------
# Challenge 1 — load_songs with new features
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Reads songs.csv and returns a list of dicts with numeric fields cast to float/int."""
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":            int(row["id"]),
                "title":         row["title"],
                "artist":        row["artist"],
                "genre":         row["genre"],
                "mood":          row["mood"],
                "energy":        float(row["energy"]),
                "tempo_bpm":     float(row["tempo_bpm"]),
                "valence":       float(row["valence"]),
                "danceability":  float(row["danceability"]),
                "acousticness":  float(row["acousticness"]),
                # Challenge 1 — new features
                "popularity":      int(row.get("popularity", 50)),
                "release_decade":  row.get("release_decade", ""),
                "detailed_mood":   row.get("detailed_mood", ""),
            })
    print(f"Loaded {len(songs)} songs from {csv_path}")
    return songs


# ---------------------------------------------------------------------------
# Challenge 2 — Scoring Modes (Strategy pattern)
# Each mode is a dict of weights. score_song uses whichever mode is passed in.
# ---------------------------------------------------------------------------

SCORING_MODES = {
    "genre-first": {
        "genre":          3.0,   # genre dominates
        "mood":           1.0,
        "energy":         1.0,
        "acousticness":   0.5,
        "popularity":     0.5,
        "decade":         0.5,
        "detailed_mood":  0.5,
    },
    "mood-first": {
        "genre":          1.0,
        "mood":           3.0,   # mood dominates
        "energy":         1.0,
        "acousticness":   0.5,
        "popularity":     0.5,
        "decade":         0.5,
        "detailed_mood":  1.0,   # detailed mood also boosted
    },
    "energy-focused": {
        "genre":          1.0,
        "mood":           1.0,
        "energy":         3.0,   # energy proximity dominates
        "acousticness":   0.5,
        "popularity":     0.5,
        "decade":         0.5,
        "detailed_mood":  0.5,
    },
}

DEFAULT_MODE = "genre-first"


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_song(
    user_prefs: Dict,
    song: Dict,
    mode: str = DEFAULT_MODE
) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences using the selected scoring mode.

    Algorithm Recipe — base weights (genre-first mode, max ~7.5):
      genre        → 3.0 if exact match
      mood         → 1.0 if exact match
      energy       → 0.0–1.0 proximity score × weight
      acousticness → 0.5 if preference aligned
      popularity   → 0.0–0.5 normalized (popularity/100 × weight)
      decade       → 0.5 if preferred decade matches
      detailed_mood→ 0.5 if detailed mood tag matches
    """
    weights = SCORING_MODES.get(mode, SCORING_MODES[DEFAULT_MODE])
    score = 0.0
    reasons = []

    # --- Genre match ---
    if song.get("genre") == user_prefs.get("genre"):
        pts = weights["genre"]
        score += pts
        reasons.append(f"genre match ({song['genre']}) +{pts}")

    # --- Mood match ---
    if song.get("mood") == user_prefs.get("mood"):
        pts = weights["mood"]
        score += pts
        reasons.append(f"mood match ({song['mood']}) +{pts}")

    # --- Energy proximity ---
    target_energy = user_prefs.get("energy", 0.5)
    song_energy = song.get("energy", 0.5)
    proximity = round(1.0 - abs(song_energy - target_energy), 3)
    pts = round(proximity * weights["energy"], 3)
    score += pts
    reasons.append(f"energy proximity {proximity:.2f} ×{weights['energy']} = +{pts}")

    # --- Acousticness alignment ---
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    is_acoustic = song.get("acousticness", 0.0) >= 0.5
    if likes_acoustic == is_acoustic:
        pts = weights["acousticness"]
        score += pts
        reasons.append(f"acousticness aligned +{pts}")

    # --- Challenge 1: Popularity bonus (normalized 0.0–1.0 × weight) ---
    pop = song.get("popularity", 50) / 100.0
    pts = round(pop * weights["popularity"], 3)
    score += pts
    reasons.append(f"popularity {song.get('popularity', 50)}/100 → +{pts}")

    # --- Challenge 1: Release decade match ---
    preferred_decade = user_prefs.get("preferred_decade", "")
    if preferred_decade and song.get("release_decade") == preferred_decade:
        pts = weights["decade"]
        score += pts
        reasons.append(f"decade match ({song['release_decade']}) +{pts}")

    # --- Challenge 1: Detailed mood tag match ---
    preferred_dm = user_prefs.get("preferred_detailed_mood", "")
    if preferred_dm and song.get("detailed_mood") == preferred_dm:
        pts = weights["detailed_mood"]
        score += pts
        reasons.append(f"detailed mood match ({song['detailed_mood']}) +{pts}")

    return (round(score, 3), reasons)


# ---------------------------------------------------------------------------
# Challenge 3 — Diversity penalty
# Applies a score penalty if the artist or genre is already in top results
# ---------------------------------------------------------------------------

def apply_diversity_penalty(
    scored: List[Tuple[Dict, float, str]],
    artist_penalty: float = 1.5,
    genre_penalty: float = 0.5,
) -> List[Tuple[Dict, float, str]]:
    """
    Re-scores a sorted list by penalizing repeated artists and genres.

    - First occurrence of an artist/genre: no penalty
    - Each repeat artist: subtract artist_penalty from score
    - Each repeat genre (beyond 2 songs): subtract genre_penalty

    Returns a newly sorted list with adjusted scores and updated reasons.
    """
    seen_artists: Dict[str, int] = {}
    seen_genres: Dict[str, int] = {}
    adjusted = []

    for song, score, explanation in scored:
        artist = song.get("artist", "")
        genre = song.get("genre", "")
        penalty = 0.0
        penalty_notes = []

        if seen_artists.get(artist, 0) >= 1:
            penalty += artist_penalty
            penalty_notes.append(f"repeat artist -{artist_penalty}")

        if seen_genres.get(genre, 0) >= 2:
            penalty += genre_penalty
            penalty_notes.append(f"repeat genre -{genre_penalty}")

        seen_artists[artist] = seen_artists.get(artist, 0) + 1
        seen_genres[genre] = seen_genres.get(genre, 0) + 1

        new_score = round(score - penalty, 3)
        new_explanation = explanation
        if penalty_notes:
            new_explanation += " | DIVERSITY PENALTY: " + ", ".join(penalty_notes)

        adjusted.append((song, new_score, new_explanation))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


# ---------------------------------------------------------------------------
# Ranking function — wires everything together
# ---------------------------------------------------------------------------

def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: str = DEFAULT_MODE,
    diversity: bool = False,
) -> List[Tuple[Dict, float, str]]:
    """
    Scores all songs, applies optional diversity penalty, returns top k.

    Ranking Rule: score every song → sort highest-to-lowest → optionally
    penalize repeated artists/genres → slice top k.
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, mode=mode)
        explanation = ", ".join(reasons)
        scored.append((song, score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)

    if diversity:
        scored = apply_diversity_penalty(scored)

    return scored[:k]
