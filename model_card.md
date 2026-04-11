# Model Card: Music Recommender Simulation

---

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Goal / Task

VibeFinder takes a user's taste profile — their favorite genre, mood, target energy level, and whether they prefer acoustic music — and suggests the 5 best-matching songs from a catalog. It does not learn or adapt. It applies a fixed set of hand-designed scoring rules every time it runs.

---

## 3. Data Used

- **Catalog size:** 20 songs
- **Genres covered:** pop, lofi, rock, ambient, jazz, synthwave, indie pop, r&b, hip-hop, classical, country, electronic, metal
- **Moods covered:** happy, chill, intense, relaxed, focused, moody, romantic, sad, energetic, angry
- **Features per song:** id, title, artist, genre, mood, energy (0–1), tempo_bpm, valence, danceability, acousticness (0–1)
- **Limits:** All data was manually written — no real audio was analyzed. Labels like "mood" and "genre" reflect one person's judgment. The catalog is tiny compared to real music apps, which have tens of millions of songs.

---

## 4. Algorithm Summary

Every song gets a score based on four rules:

1. **Genre match** → +2.0 points. Worth the most because genre is the biggest dividing line between types of music. A rock fan and a jazz fan have almost no overlap.
2. **Mood match** → +1.0 points. Important but more flexible — the same person might enjoy "happy" and "energetic" songs in the same genre.
3. **Energy proximity** → between 0.0 and 1.0 points. Instead of rewarding just high or low energy, we calculate how *close* the song's energy is to what the user wants. A song exactly at the target gets 1.0. A song far away gets close to 0.
4. **Acousticness alignment** → +0.5 points. If the song's acoustic feel (plugged-in vs. unplugged) matches the user's preference, it earns a bonus.

The maximum any song can score is 4.5. All 20 songs are scored, sorted highest to lowest, and the top 5 are returned with an explanation of why each one ranked where it did.

---

## 5. Observed Behavior / Biases

**Genre dominates everything.** At +2.0, genre is worth nearly half the maximum score. When a user has conflicting preferences — for example, wanting high energy (0.90) but a sad mood in r&b — the system recommended a low-energy song (energy 0.44) as #1 because genre + mood together scored +3.0, which easily buried the energy mismatch. The user asked for something that felt intense; they got something that felt slow and melancholic.

**Binary matching has no middle ground.** A hip-hop fan gets zero genre points for r&b, even though those genres are closely related. The system treats every mismatch as equally wrong, whether the user wanted hip-hop and got r&b, or wanted hip-hop and got classical.

**Missing genres are silently ignored.** When a user wanted k-pop — a genre not in the catalog — the genre weight never fired at all. The system quietly fell back on mood and energy, returning indie pop and pop songs. There is no warning or indication that the genre wasn't found.

**The same songs appear in too many lists.** High-energy songs like Storm Runner and Move the Crowd showed up across multiple unrelated profiles purely because they score well on energy proximity, not because they actually fit those profiles.

---

## 6. Evaluation Process

Five user profiles were tested:

| Profile | Top Result | Score | Surprising? |
|---|---|---|---|
| Hip-hop / energetic / energy 0.85 | Concrete Jungle | 4.48 | No — perfect match |
| Lofi / chill / energy 0.38 | Library Rain | 4.47 | No — perfect match |
| Rock / intense / energy 0.90 | Storm Runner | 4.49 | No — perfect match |
| R&B / sad / energy 0.90 (conflicting) | Heartstrings (energy 0.44) | 3.54 | Yes — high energy user got a slow song |
| K-pop / happy / energy 0.75 (missing genre) | Rooftop Lights (indie pop) | 2.49 | Somewhat — reasonable fallback but wrong genre |

One experiment was also run: genre weight was halved (+2.0 → +1.0) and energy weight was doubled. The top result stayed the same for the rock profile, but the #2 and #3 results nearly caught up in score — the system became much less genre-loyal and more energy-loyal. This showed that small weight changes have a big effect on who gets recommended what.

---

## 7. Intended Use and Non-Intended Use

**Intended use:**
- Classroom exploration of how recommendation systems work
- Learning how weighted scoring, sorting, and data flow connect together
- Demonstrating the difference between a Scoring Rule (one song) and a Ranking Rule (all songs)

**Not intended for:**
- Real users making actual music decisions — the catalog is too small and the labels are too rough
- Any production environment — there is no error handling for missing data, unknown genres, or malformed input
- Representing any real person's taste accurately — the user profile is a simplified snapshot, not a real listener model
- Making decisions that affect people — this is a simulation, not a deployed system

---

## 8. Ideas for Improvement

1. **Add partial genre credit** — r&b should score something for a hip-hop user, not zero. You could group related genres and award partial points for nearby ones.
2. **Support multiple preferred genres** — real listeners don't have one genre. Letting users say "I like hip-hop and r&b" would make results much more realistic.
3. **Add a diversity rule** — prevent the same song from appearing across too many different profiles by penalizing repeated top results or forcing variety in the top 5.

---

## 9. Personal Reflection

**Biggest learning moment:** The edge case where a user who wanted energy 0.90 got a song at energy 0.44 as their #1 recommendation. It was a perfect reminder that the algorithm does exactly what you tell it to — it doesn't understand music, it just adds up numbers. Genre + mood together scored +3.0, which mathematically crushed the energy mismatch even though the result felt completely wrong. That gap between "mathematically correct" and "actually useful" is probably the most important thing I took away from this project.

**How AI tools helped, and when I had to double-check:** AI was useful for explaining concepts quickly — things like why lambda works, what slice does, and how the proximity formula compares to a binary match. But the weights (2.0, 1.0, 0.5) were not generated by AI — those came from thinking through what actually matters in music. AI suggested structures and syntax; the design decisions had to come from understanding the problem.

**What surprised me about simple algorithms:** The results for well-matched profiles (lofi/chill, rock/intense) felt surprisingly accurate — like something a real app would return. It didn't feel like basic math, even though that's all it is. That's what makes recommenders feel smart when they work: the output looks intelligent even when the underlying logic is just addition and sorting.

**What I'd try next:** The most interesting extension would be replacing the static user profile with a listening history — track which songs the user plays or skips and update the weights automatically. That's essentially how Spotify works, and building even a basic version of that would make the system feel genuinely adaptive instead of frozen.
