# Model Card: VibeFinder — AI Music Recommender

---

## 1. Model Name

**VibeFinder 2.0** (extended from VibeFinder 1.0 / Music Recommender Simulation)

---

## 2. Goal / Task

VibeFinder takes a user's natural language music request and returns the 5 best-matching songs from a 78-song catalog. The system combines a Groq LLM agent (llama-3.3-70b-versatile) for language understanding with a deterministic rule-based scoring engine for ranking. The agent handles intent classification, RAG-grounded preference extraction, and result quality checking. The engine handles all mathematical scoring.

---

## 3. Data Used

**Song catalog (`data/songs.csv`):**
- 78 songs across 13 genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop, r&b, hip-hop, classical, country, electronic, metal
- 3 release decades: 2000s, 2010s, 2020s
- Features per song: id, title, artist, genre, mood, energy (0–1), tempo_bpm, valence, danceability, acousticness (0–1), popularity (0–100), release_decade, detailed_mood
- All data is manually authored — no real audio was analyzed

**Genre knowledge base (`data/genre_knowledge.json`):**
- 13 genre documents used for RAG retrieval
- Each document contains: description, energy range, typical moods, typical detailed moods, acoustic level, related genres
- Used to ground the agent's energy estimates in documented facts rather than training intuition

**Limits:** The catalog is fictional and small compared to real music apps. Genre and mood labels reflect one developer's judgment. The knowledge base documents are handcrafted, not derived from audio analysis.

---

## 4. Algorithm Summary

**Agent layer (Groq LLM):**
1. CLASSIFY — determines whether the request is genre-driven, mood-driven, activity-driven, or era-driven
2. RETRIEVE — fetches the relevant genre document from the knowledge base (RAG)
3. PLAN — extracts structured preferences using retrieved context and four few-shot examples
4. REFLECT — evaluates result quality and assigns a 0–100% confidence score; retries if quality is poor

**Scoring engine (rule-based):**

| Rule | Points | How it works |
|---|---|---|
| Genre match | +3.0 | Exact string match |
| Mood match | +1.0 | Exact string match |
| Energy proximity | 0.0–1.0 | `1.0 - abs(song_energy - target)` |
| Acousticness alignment | +0.5 | Acoustic preference agrees with song |
| Popularity bonus | 0.0–0.5 | `popularity / 100 × 0.5` |
| Decade match | +0.5 | Release decade equals preferred decade |
| Detailed mood match | +0.5 | Detailed mood tag matches preference |

**Maximum score: 7.5.** Three scoring modes (genre-first, mood-first, energy-focused) shift the weights based on the agent's classification. A diversity penalty deducts points from repeat artists in the top results.

---

## 5. Observed Behavior and Biases

**Genre dominates.** At +3.0, genre is worth 40% of the maximum score. A song with the right genre but wrong everything else outscores a song with the wrong genre but perfect mood, energy, and acousticness. This is an intentional design decision but creates blind spots — adjacent genres like hip-hop and r&b are treated as completely different.

**Binary matching has no middle ground.** A hip-hop fan gets zero genre points for r&b, even though the genres share tempo, production style, and culture. Every mismatch is equally penalized regardless of musical proximity.

**The RAG knowledge base reduced energy estimation errors.** Without it, the model estimated lofi energy at ~0.30; the actual dataset clusters at 0.35–0.43. With the knowledge base documenting the range as 0.20–0.50, estimates improved to ~0.38. However, the knowledge base is still handcrafted — it represents one person's understanding of genre characteristics, not measured audio data.

**Outliers can survive the REFLECT step.** During testing, a country song appeared in lofi results because it matched the energy and acoustic constraints. The REFLECT step did not flag it because the other four results were correct — the agent evaluated the set holistically, not each song individually.

**Dataset cultural bias.** Genres like K-pop, reggae, gospel, afrobeats, and bhangra are absent. Users whose musical identity centers on these genres receive poorer recommendations. This is not a technical limitation — it is a design decision that reflects whose taste was prioritized when the catalog was built.

---

## 6. Evaluation

**Automated testing:**
- 19 unit tests covering all scoring rules, edge cases, and diversity penalty behavior — 19/19 passing
- 6 reliability checks for preset genre/energy constraints — 6/6 passing
- 8-case evaluation harness with per-case engine confidence scores — 8/8 passing at 88% average confidence

**Few-shot specialization comparison:**
- `tests/test_fewshot_comparison.py` runs 3 queries through the PLAN step with and without few-shot examples and compares genre accuracy, energy accuracy, scoring mode selection, and schema completeness
- Demonstrates measurable improvement from few-shot prompting on all four criteria

**Confidence scoring:**
- The REFLECT step assigns a 0–100% confidence score to each result set
- Scores below an internal threshold trigger a retry with adjusted preferences

---

## 7. Intended Use and Non-Intended Use

**Intended use:**
- Demonstrating how a natural language interface can be layered on top of a rule-based system
- Exploring RAG, few-shot prompting, and agentic loops in a low-stakes environment
- Educational exploration of recommendation system design and AI reliability

**Not intended for:**
- Real users making actual music decisions — the catalog is too small and all titles/artists are fictional
- Any production environment — there is no authentication, rate limiting beyond what Groq provides, or content moderation
- Making decisions that affect people — this is a simulation

---

## 8. Ideas for Improvement

1. **Partial genre credit** — r&b should score something for a hip-hop user. Genre embeddings or a similarity matrix would replace binary matching with gradient scoring.
2. **User feedback loop** — track which songs users skip or replay and use that signal to update the weights automatically. This would make the system adaptive instead of static.
3. **Larger and more diverse catalog** — adding real-world genres and a much larger dataset would meaningfully reduce cultural bias and make recommendations feel authentic.
4. **Per-song REFLECT checking** — rather than evaluating the result set holistically, the agent should check each individual song against the request. This would catch outliers like the country song in lofi results.
