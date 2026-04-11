# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

This simulation builds a small music recommender that matches songs to a user based on their taste profile. It scores each song using weighted rules — rewarding genre and mood matches, and using proximity math to reward songs whose energy level is closest to what the user prefers. The top-scoring songs are returned as recommendations.

---

## How The System Works

Real-world recommenders like Spotify or YouTube learn from massive amounts of listening history — they detect patterns across millions of users and use machine learning to figure out what weights and features matter most. Our version skips the learning step and instead hard-codes those decisions manually, which makes the logic transparent and easy to reason about.

This system will prioritize **genre and mood matching** above all else, since those define the identity of a song. Energy proximity is used as a secondary signal — we reward songs that are *close* to the user's preferred energy level, not just songs that are high or low energy.

### Song Features

Each `Song` object stores the following fields:

| Feature | Type | Description |
|---|---|---|
| `id` | int | Unique identifier |
| `title` | str | Song title |
| `artist` | str | Artist name |
| `genre` | str | Music genre (e.g. pop, rock, jazz) |
| `mood` | str | Emotional mood (e.g. happy, calm, melancholic) |
| `energy` | float | Energy level from 0.0 (low) to 1.0 (high) |
| `tempo_bpm` | float | Beats per minute |
| `valence` | float | Musical positivity, 0.0 to 1.0 |
| `danceability` | float | How suitable for dancing, 0.0 to 1.0 |
| `acousticness` | float | How acoustic (vs. electronic), 0.0 to 1.0 |

### UserProfile Features

Each `UserProfile` stores:

| Feature | Type | Description |
|---|---|---|
| `favorite_genre` | str | The genre the user prefers most |
| `favorite_mood` | str | The mood the user is looking for |
| `target_energy` | float | Ideal energy level, 0.0 to 1.0 |
| `likes_acoustic` | bool | Whether the user prefers acoustic over electronic |

### Algorithm Recipe

#### Scoring Rule (one song at a time)

Each song is judged against the user profile and earns points based on how well it matches:

| Rule | Points | How it works |
|---|---|---|
| Genre match | **+2.0** | Exact string match — `song["genre"] == user["genre"]` |
| Mood match | **+1.0** | Exact string match — `song["mood"] == user["mood"]` |
| Energy proximity | **+0.0 to +1.0** | `1.0 - abs(song_energy - target_energy)` — closer = more points |
| Acousticness alignment | **+0.5** | Song's acousticness >= 0.5 must agree with `likes_acoustic` |

**Maximum possible score: 4.5**

Genre is worth the most (+2.0) because it is the hardest constraint — a hip-hop fan and a classical fan have almost no overlap. Mood is worth less (+1.0) because it is more flexible — the same user might enjoy both "happy" and "energetic" songs in the same genre. Energy uses proximity math instead of a binary match so that a song at 0.83 is not penalized the same as a song at 0.20 when the target is 0.85.

#### Ranking Rule (list of songs)

Every song in the catalog is scored using the Scoring Rule. The results are sorted from highest to lowest score. The top `k` songs (default: 5) are returned as recommendations. No song is skipped before scoring — all 20 are evaluated so the ranking is a fair comparison.

#### User Profile Used

```python
user_prefs = {
    "genre": "hip-hop",
    "mood": "energetic",
    "energy": 0.85,       # 0.0 = calm, 1.0 = intense
    "likes_acoustic": False
}
```

### Potential Biases and Limitations

- **Genre over-prioritization** — genre is worth +2.0, twice the mood weight. A song that perfectly matches mood, energy, and acousticness but has the wrong genre will still score lower than a genre-match with nothing else in common. Great songs can get buried this way.
- **Binary genre/mood matching** — there is no partial credit. A hip-hop fan gets zero genre points for an r&b song, even though the two genres are closely related. The system treats all mismatches as equally wrong.
- **Rare moods get wasted weight** — "energetic" only appears in 2 of 20 songs. If the user's mood is uncommon in the catalog, the +1.0 mood weight rarely fires and the ranking falls back almost entirely on genre and energy.
- **`likes_acoustic` penalizes broadly** — setting `likes_acoustic: False` silently down-ranks jazz, classical, lofi, and country songs without the user explicitly saying they dislike those genres. A user might want chill jazz even if they prefer electronic production overall.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

![Scoring Modes Comparison](assets/screenshot.png)

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

