"""
VibeFinder — Streamlit UI
Agentic music recommender powered by Groq (llama-3.3-70b) + a custom scoring engine.
"""

import os
import sys

# Ensure src/ is on the path regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from recommender import load_songs
from agent import run_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeFinder",
    page_icon="🎵",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎵 VibeFinder")
st.caption(
    "Describe what you want to listen to in plain English. "
    "A Groq AI agent will classify, retrieve, plan, score, and reflect on your results."
)
st.divider()

# ── Load catalog once ─────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

@st.cache_data
def get_songs():
    return load_songs(DATA_PATH)

songs = get_songs()

# ── Input ─────────────────────────────────────────────────────────────────────
query = st.text_input(
    "What are you in the mood for?",
    placeholder="e.g. chill lofi music to study to from the 2020s",
)

col1, col2 = st.columns([1, 5])
with col1:
    search = st.button("Find Music", type="primary")

# ── Run agent ─────────────────────────────────────────────────────────────────
if search and query.strip():

    with st.spinner("Agent thinking..."):
        results, log = run_agent(query.strip(), songs)

    # ── Agent log ─────────────────────────────────────────────────────────────
    st.subheader("Agent Log")

    ICONS = {
        "CLASSIFY": "🔎",
        "RETRIEVE": "📚",
        "PLAN":     "🧠",
        "ACT":      "⚙️",
        "REFLECT":  "🔍",
        "RETRY":    "🔄",
        "DONE":     "✅",
        "ERROR":    "❌",
    }

    confidence_score = None
    for step, message in log:
        if step == "CONFIDENCE":
            confidence_score = message
            continue
        icon = ICONS.get(step, "•")
        if step == "ERROR":
            st.error(f"{icon} **[{step}]** {message}")
        elif step == "RETRY":
            st.warning(f"{icon} **[{step}]** {message}")
        elif step == "DONE":
            st.success(f"{icon} **[{step}]** {message}")
        else:
            st.info(f"{icon} **[{step}]** {message}")

    if confidence_score is not None:
        st.metric("Match Confidence", f"{confidence_score:.0%}")

    st.divider()

    # ── Results ───────────────────────────────────────────────────────────────
    if results:
        st.subheader("Your Recommendations")

        for i, (song, score, explanation) in enumerate(results, start=1):
            with st.container():
                left, right = st.columns([4, 1])

                with left:
                    st.markdown(f"### #{i} {song['title']}")
                    st.markdown(f"**{song['artist']}**")

                    tags = " · ".join(filter(None, [
                        song.get("genre", ""),
                        song.get("mood", ""),
                        f"energy {song.get('energy', '')}",
                        song.get("release_decade", ""),
                        song.get("detailed_mood", ""),
                    ]))
                    st.caption(tags)

                    short_explanation = (
                        explanation[:120] + "..." if len(explanation) > 120 else explanation
                    )
                    st.caption(f"_{short_explanation}_")

                with right:
                    st.metric("Score", f"{score:.2f}")

            st.divider()

    else:
        st.warning("No recommendations found. Try rephrasing your request.")

elif search and not query.strip():
    st.warning("Please enter what you are in the mood for.")
