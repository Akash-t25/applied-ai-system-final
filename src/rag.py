"""
RAG (Retrieval-Augmented Generation) module for VibeFinder.

Loads a genre knowledge base from JSON and retrieves the relevant
document(s) before the agent plans preferences. This grounds the
agent's energy/mood estimates in documented genre characteristics
rather than relying solely on training knowledge.
"""

import json
import os
from typing import Optional

KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "genre_knowledge.json"
)

_KNOWLEDGE_BASE: Optional[dict] = None


def _load_knowledge_base() -> dict:
    """Load and cache the genre knowledge base."""
    global _KNOWLEDGE_BASE
    if _KNOWLEDGE_BASE is None:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            _KNOWLEDGE_BASE = json.load(f)
    return _KNOWLEDGE_BASE


def retrieve(genre: str) -> Optional[dict]:
    """
    Retrieve the knowledge document for a specific genre.
    Returns None if the genre is not in the knowledge base.
    """
    kb = _load_knowledge_base()
    return kb.get(genre.lower())


def retrieve_as_text(genre: str) -> str:
    """
    Retrieve genre knowledge formatted as a text block ready for
    injection into an LLM prompt.
    """
    doc = retrieve(genre)
    if not doc:
        return ""

    lines = [
        f"Genre knowledge for '{genre}':",
        f"  Description:    {doc['description']}",
        f"  Energy range:   {doc['energy_range'][0]} – {doc['energy_range'][1]}",
        f"  Typical moods:  {', '.join(doc['typical_moods'])}",
        f"  Vibes:          {', '.join(doc['typical_detailed_moods'])}",
        f"  Acousticness:   {doc['acoustic_level']}",
        f"  Related genres: {', '.join(doc['related_genres'])}",
    ]
    return "\n".join(lines)


def retrieve_for_query(query: str) -> str:
    """
    Scan the query for genre keywords and return knowledge for any
    genre mentioned. Returns an empty string if nothing is found.
    """
    kb = _load_knowledge_base()
    query_lower = query.lower()

    found = []
    for genre in kb:
        if genre in query_lower:
            found.append(retrieve_as_text(genre))

    return "\n\n".join(found)
