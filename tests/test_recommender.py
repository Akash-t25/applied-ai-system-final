"""
Unit tests for the VibeFinder recommender engine.
Tests scoring logic, diversity penalty, and edge cases — no API calls needed.
Run with: pytest
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import Song, UserProfile, Recommender, score_song, apply_diversity_penalty


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_song(**overrides):
    """Build a minimal song dict with sensible defaults."""
    base = {
        "id": 1, "title": "Test Song", "artist": "Test Artist",
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "tempo_bpm": 120, "valence": 0.8, "danceability": 0.8,
        "acousticness": 0.2, "popularity": 70,
        "release_decade": "2020s", "detailed_mood": "euphoric",
    }
    base.update(overrides)
    return base


def make_user(**overrides):
    """Build a minimal user prefs dict with sensible defaults."""
    base = {
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "likes_acoustic": False, "preferred_decade": "2020s",
        "preferred_detailed_mood": "euphoric",
    }
    base.update(overrides)
    return base


def make_recommender_songs():
    return [
        Song(id=1, title="Pop Hit", artist="Artist A", genre="pop", mood="happy",
             energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=2, title="Lofi Loop", artist="Artist B", genre="lofi", mood="chill",
             energy=0.4, tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9),
        Song(id=3, title="Rock Banger", artist="Artist C", genre="rock", mood="intense",
             energy=0.9, tempo_bpm=150, valence=0.4, danceability=0.6, acousticness=0.1),
    ]


# ---------------------------------------------------------------------------
# Genre matching
# ---------------------------------------------------------------------------

def test_genre_match_adds_points():
    song = make_song(genre="lofi")
    user = make_user(genre="lofi")
    score, reasons = score_song(user, song, mode="genre-first")
    assert any("genre match" in r for r in reasons)


def test_genre_mismatch_adds_no_genre_points():
    song = make_song(genre="metal")
    user = make_user(genre="lofi")
    score, reasons = score_song(user, song, mode="genre-first")
    assert not any("genre match" in r for r in reasons)


# ---------------------------------------------------------------------------
# Mood matching
# ---------------------------------------------------------------------------

def test_mood_match_adds_points():
    song = make_song(mood="chill")
    user = make_user(mood="chill")
    score, reasons = score_song(user, song, mode="genre-first")
    assert any("mood match" in r for r in reasons)


def test_mood_mismatch_adds_no_mood_points():
    song = make_song(mood="angry")
    user = make_user(mood="chill")
    score, reasons = score_song(user, song, mode="genre-first")
    # Use startswith to avoid matching "detailed mood match" as a false positive
    assert not any(r.startswith("mood match") for r in reasons)


# ---------------------------------------------------------------------------
# Energy proximity math
# ---------------------------------------------------------------------------

def test_perfect_energy_match_gives_max_proximity():
    song = make_song(energy=0.5)
    user = make_user(energy=0.5)
    score, reasons = score_song(user, song, mode="genre-first")
    assert any("energy proximity 1.0" in r for r in reasons)


def test_energy_far_off_gives_low_proximity():
    song = make_song(energy=0.95)
    user = make_user(energy=0.20)
    score, reasons = score_song(user, song, mode="genre-first")
    proximity_reason = [r for r in reasons if "energy proximity" in r][0]
    proximity_value = float(proximity_reason.split("energy proximity ")[1].split(" ")[0])
    assert proximity_value < 0.3


def test_energy_proximity_is_symmetric():
    song_a = make_song(energy=0.6)
    song_b = make_song(energy=0.4)
    user = make_user(energy=0.5)
    score_a, _ = score_song(user, song_a)
    score_b, _ = score_song(user, song_b)
    assert score_a == score_b


# ---------------------------------------------------------------------------
# Acousticness alignment
# ---------------------------------------------------------------------------

def test_acoustic_alignment_when_both_acoustic():
    song = make_song(acousticness=0.9)
    user = make_user(likes_acoustic=True)
    score, reasons = score_song(user, song)
    assert any("acousticness aligned" in r for r in reasons)


def test_acoustic_misalignment_adds_no_points():
    song = make_song(acousticness=0.1)
    user = make_user(likes_acoustic=True)
    score, reasons = score_song(user, song)
    assert not any("acousticness aligned" in r for r in reasons)


# ---------------------------------------------------------------------------
# Decade and detailed mood
# ---------------------------------------------------------------------------

def test_decade_match_adds_points():
    song = make_song(release_decade="2010s")
    user = make_user(preferred_decade="2010s")
    score, reasons = score_song(user, song)
    assert any("decade match" in r for r in reasons)


def test_decade_mismatch_adds_no_points():
    song = make_song(release_decade="2000s")
    user = make_user(preferred_decade="2020s")
    score, reasons = score_song(user, song)
    assert not any("decade match" in r for r in reasons)


def test_detailed_mood_match_adds_points():
    song = make_song(detailed_mood="nostalgic")
    user = make_user(preferred_detailed_mood="nostalgic")
    score, reasons = score_song(user, song)
    assert any("detailed mood match" in r for r in reasons)


# ---------------------------------------------------------------------------
# Score is always non-negative
# ---------------------------------------------------------------------------

def test_score_never_negative():
    song = make_song(genre="metal", mood="angry", energy=0.99, acousticness=0.01)
    user = make_user(genre="lofi", mood="chill", energy=0.2, likes_acoustic=True)
    score, _ = score_song(user, song)
    assert score >= 0


# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------

def test_recommend_returns_correct_count():
    rec = Recommender(make_recommender_songs())
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    results = rec.recommend(user, k=2)
    assert len(results) == 2


def test_recommend_top_result_matches_genre():
    rec = Recommender(make_recommender_songs())
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)
    results = rec.recommend(user, k=1)
    assert results[0].genre == "lofi"


def test_explain_recommendation_is_non_empty():
    rec = Recommender(make_recommender_songs())
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str) and explanation.strip() != ""


# ---------------------------------------------------------------------------
# Diversity penalty
# ---------------------------------------------------------------------------

def test_diversity_penalty_reduces_repeat_artist_score():
    song_a = make_song(title="Song A", artist="Same Artist", genre="pop")
    song_b = make_song(title="Song B", artist="Same Artist", genre="pop")
    scored = [(song_a, 5.0, "reason"), (song_b, 4.8, "reason")]
    adjusted = apply_diversity_penalty(scored)
    scores = [s[1] for s in adjusted]
    # Second occurrence of Same Artist should be penalized
    assert scores[1] < 4.8


def test_diversity_penalty_leaves_unique_artists_untouched():
    song_a = make_song(title="Song A", artist="Artist One", genre="pop")
    song_b = make_song(title="Song B", artist="Artist Two", genre="rock")
    scored = [(song_a, 5.0, "reason"), (song_b, 4.0, "reason")]
    adjusted = apply_diversity_penalty(scored)
    assert adjusted[0][1] == 5.0
    assert adjusted[1][1] == 4.0
