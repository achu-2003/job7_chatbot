"""Deterministic category browse — resolve a candidate's phrase to a known job
category and format the full listing, WITHOUT relying on the LLM planner.

The weak chat model can't be trusted to pass the exact category name: it
abbreviates "Information Technology" to "IT" and often runs a fuzzy vector
search that returns 3 unrelated roles instead of the whole category. So when the
message clearly names a category we know, we resolve it here (exactly) and list
that category straight from SQL.

Both functions are pure (text in, value out) so they're trivially unit-tested.
"""
from __future__ import annotations

import re

# Filler words to ignore when matching a phrase against a category name.
_STOP = {
    "show", "me", "the", "all", "jobs", "job", "list", "open", "available",
    "roles", "role", "in", "for", "want", "i", "a", "an", "some", "please",
    "view", "see", "category", "categories", "and", "of", "any", "give",
}


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _acronym(name: str) -> str:
    """First letter of each real word — 'Information Technology' -> 'it'."""
    words = [w for w in re.findall(r"[A-Za-z]+", name) if len(w) > 1]
    return "".join(w[0] for w in words).lower()


def match_category(text: str, categories: list[str]) -> str | None:
    """Resolve a phrase to ONE known category name, or None.

    Strongest match wins: full category name present → acronym token
    ('IT' → Information Technology) → a distinctive category word / prefix
    ('admin' → Administration). Returns the actual category name to filter on,
    so the SQL match is exact (no '%IT%' matching the wrong rows).
    """
    if not text or not categories:
        return None
    tset = set(_tokens(text))
    low = " ".join(_tokens(text))
    best: str | None = None
    best_score = 0
    for cat in categories:
        cat_l = cat.lower()
        cat_tokens = [t for t in _tokens(cat) if t != "and"]
        score = 0
        meaningful = [t for t in cat_tokens if t not in _STOP]
        if cat_l in low or (meaningful and all(t in tset for t in meaningful)):
            score = 100 + len(cat_l)                      # full name
        elif len(_acronym(cat)) >= 2 and _acronym(cat) in tset:
            score = 50 + len(_acronym(cat))               # acronym (IT, HR…)
        else:
            for d in (t for t in cat_tokens if len(t) >= 4 and t not in _STOP):
                for m in tset:
                    if len(m) >= 3 and (d.startswith(m) or m.startswith(d)):
                        score = max(score, 20 + len(m))   # distinctive word / prefix
        if score > best_score:
            best_score, best = score, cat
    return best


def build_job_list_text(jobs: list[dict], category: str | None = None) -> str:
    """A compact, deterministic listing of EVERY job passed in — bullets (not
    '1.' numbering, which the bubble splitter would read as sentence breaks)."""
    n = len(jobs)
    header = f"Here are all {n} {category} roles:" if category else f"Here are all {n} roles:"
    lines = [header, ""]
    for j in jobs:
        line = f"• {j.get('title') or 'Role'}"
        if j.get("location"):
            line += f" — {j['location']}"
        if j.get("job_ref"):
            line += f"  [{j['job_ref']}]"
        lines.append(line)
    lines += ["", "Reply with a role's name or its reference to apply or get details."]
    return "\n".join(lines)
