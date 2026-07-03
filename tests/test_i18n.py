"""Multilingual translate-pivot core: detection + batched LLM translation."""
import json

from app import i18n


class _FakeLLM:
    """Captures purposes and fakes translate_in / translate_out responses."""
    def __init__(self):
        self.calls = []

    async def chat(self, *, purpose, messages, **kw):
        self.calls.append(purpose)
        if purpose == "translate_in":
            return "EN[" + messages[-1]["content"] + "]", {}
        # translate_out: echo each JSON value, prefixed, keeping keys
        payload = json.loads(messages[-1]["content"])
        return json.dumps({k: "TA::" + v for k, v in payload.items()}), {}


class _BoomLLM:
    async def chat(self, **kw):
        raise RuntimeError("llm down")


def test_detect_lang():
    assert i18n.detect_lang("வணக்கம், எனக்கு வேலை வேண்டும்") == "ta"
    assert i18n.detect_lang("मुझे नौकरी चाहिए") == "hi"
    assert i18n.detect_lang("i want a job") == "en"
    assert i18n.detect_lang("post job வேலை") == "ta"      # any Tamil char wins
    assert i18n.detect_lang("12345 !!") == "en"
    assert i18n.detect_lang("") == "en" and i18n.detect_lang(None) == "en"


async def test_translate_many_noop_for_english():
    llm = _FakeLLM()
    out = await i18n.translate_many(llm, ["Post a Job", "Buy Credits"], to_lang="en")
    assert out == ["Post a Job", "Buy Credits"]
    assert llm.calls == []                       # English never calls the LLM


async def test_translate_many_unsupported_lang_noop():
    llm = _FakeLLM()
    assert await i18n.translate_many(llm, ["x"], to_lang="fr") == ["x"]
    assert llm.calls == []


async def test_translate_many_translates_skips_nonwordy_and_caches():
    i18n._CACHE.clear()
    llm = _FakeLLM()
    # non-glossary strings, so they exercise the LLM batch path (glossary terms would
    # short-circuit before the LLM — covered by test_glossary_covers_menu_labels)
    out = await i18n.translate_many(llm, ["Good to see you", "12345", "Tell me more"], to_lang="ta")
    assert out == ["TA::Good to see you", "12345", "TA::Tell me more"]   # digits-only skipped
    assert llm.calls == ["translate_out"]        # ONE batched call for the two strings
    # a repeat is served from cache — no new LLM call
    llm2 = _FakeLLM()
    assert await i18n.translate_many(llm2, ["Good to see you"], to_lang="ta") == ["TA::Good to see you"]
    assert llm2.calls == []


async def test_translate_many_best_effort_on_error():
    i18n._CACHE.clear()
    out = await i18n.translate_many(_BoomLLM(), ["Hello", "World"], to_lang="ta")
    assert out == ["Hello", "World"]             # originals kept on error


async def test_to_english_noop_and_translate():
    llm = _FakeLLM()
    assert await i18n.to_english(llm, "hello", source_lang="en") == "hello"
    assert llm.calls == []                       # English in → no call
    # a non-glossary phrase → goes to the LLM (a glossary term would reverse-map)
    out = await i18n.to_english(llm, "எனக்கு வேலை வேண்டும்", source_lang="ta")
    assert out == "EN[எனக்கு வேலை வேண்டும்]" and llm.calls == ["translate_in"]


async def test_to_english_best_effort_on_error():
    assert await i18n.to_english(_BoomLLM(), "எனக்கு வேலை வேண்டும்", source_lang="ta") == "எனக்கு வேலை வேண்டும்"


# --- interactive-label localization -----------------------------------------

def test_collect_localizable_covers_labels_not_ids():
    result = {
        "draft_response": "Welcome!",
        "delivery_plan": [{"text": "Tap a job."}],
        "whatsapp_interactive": {"interactive": {"type": "list",
            "body": {"text": "Job categories"},
            "action": {"button": "View roles", "sections": [
                {"title": "Roles", "rows": [
                    {"id": "category:IT", "title": "Information Technology", "description": "5 roles"}]}]}}},
        "whatsapp_messages": [{"interactive": {"type": "button",
            "body": {"text": "Software Developer — Chennai"},
            "action": {"buttons": [
                {"type": "reply", "reply": {"id": "apply:r1", "title": "Apply"}},
                {"type": "reply", "reply": {"id": "save:r1", "title": "Save"}}]}}}],
    }
    texts, setters = i18n.collect_localizable(result)
    for s in ("Welcome!", "Tap a job.", "Job categories", "View roles", "Roles",
              "Information Technology", "5 roles", "Software Developer — Chennai", "Apply", "Save"):
        assert s in texts, s
    assert "category:IT" not in texts and "apply:r1" not in texts   # ids never collected
    # apply each setter — mutates in place, leaves ids untouched
    for s, t in zip(setters, texts):
        s("X:" + t)
    inter = result["whatsapp_interactive"]["interactive"]["action"]["sections"][0]["rows"][0]
    assert inter["title"].startswith("X:") and inter["id"] == "category:IT"
    assert result["whatsapp_messages"][0]["interactive"]["action"]["buttons"][0]["reply"]["id"] == "apply:r1"
    assert result["draft_response"] == "X:Welcome!"


def test_interactive_label_truncated_to_wa_cap():
    result = {"whatsapp_interactive": {"interactive": {"type": "list", "action": {
        "sections": [{"rows": [{"id": "x", "title": "short"}]}]}}}}
    _, setters = i18n.collect_localizable(result)
    setters[0]("A" * 50)                                   # an over-long translation
    title = result["whatsapp_interactive"]["interactive"]["action"]["sections"][0]["rows"][0]["title"]
    assert len(title) <= 24                                # row title cap


# --- LLM category classification --------------------------------------------

class _CatLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    async def chat(self, *, purpose, messages, **kw):
        self.calls += 1
        return self.answer, {}


_CATS = ["Logistics & Supply Chain", "Manufacturing", "Information Technology"]


async def test_classify_category_maps_synonym():
    assert await i18n.classify_category(_CatLLM("Logistics & Supply Chain"),
                                        "transport", _CATS) == "Logistics & Supply Chain"


async def test_classify_category_none_for_specific_role():
    assert await i18n.classify_category(_CatLLM("NONE"), "welder", _CATS) is None


async def test_classify_category_rejects_non_catalog_answer():
    # the model hallucinates a name not in the catalog → rejected (no false browse)
    assert await i18n.classify_category(_CatLLM("Transportation"), "transport", _CATS) is None


async def test_classify_category_empty_inputs():
    assert await i18n.classify_category(_CatLLM("X"), "", _CATS) is None
    assert await i18n.classify_category(_CatLLM("X"), "transport", []) is None


async def test_translate_many_falls_back_per_string_on_bad_batch():
    """If the batched JSON call returns garbage (smaller models do this on long /
    multi-line text), each string is rescued by an individual call — so a bad batch
    never English-dumps the turn."""
    class _FlakyLLM:
        def __init__(self): self.batch = 0; self.singles = 0
        async def chat(self, *, purpose, messages, **kw):
            if purpose == "translate_out":
                self.batch += 1
                return "sorry, I can't", {}                  # not valid JSON
            self.singles += 1
            return "TA::" + messages[-1]["content"], {}       # per-string succeeds

    i18n._CACHE.clear()
    llm = _FlakyLLM()
    out = await i18n.translate_many(
        llm, ["Upload Resume", "Don't have it handy? Skip.", "12345"], to_lang="ta")
    assert out == ["TA::Upload Resume", "TA::Don't have it handy? Skip.", "12345"]
    assert llm.batch == 1 and llm.singles == 2                # batch tried once, 2 rescued


async def test_category_from_translation_reverse_maps():
    """A category name typed in the user's language reverse-maps to the exact English
    category via the bot's OWN (glossary) translations — consistent round-trip, no
    LLM. Fixes the 'தயாரிப்பு → wrong category' / 'World Transport' cases."""
    class _NoLLM:
        async def chat(self, **kw):
            raise AssertionError("glossary terms must not call the LLM")

    i18n._CACHE.clear()
    cats = ["Manufacturing", "Healthcare", "Hospitality & Tourism"]
    assert await i18n.category_from_translation(_NoLLM(), "உற்பத்தி", cats, "ta") == "Manufacturing"
    # trailing punctuation / whitespace tolerated
    assert await i18n.category_from_translation(_NoLLM(), " சுகாதாரம். ", cats, "ta") == "Healthcare"
    assert await i18n.category_from_translation(
        _NoLLM(), "விருந்தோம்பல் & சுற்றுலா", cats, "ta") == "Hospitality & Tourism"
    # English / unknown / empty → None
    assert await i18n.category_from_translation(_NoLLM(), "anything", cats, "en") is None
    assert await i18n.category_from_translation(_NoLLM(), "எதுவுமில்லை", cats, "ta") is None


async def test_glossary_overrides_llm_for_category_names():
    """Fixed category names use the curated glossary (authoritative + consistent),
    NOT the LLM — fixing 'Hospitality & Tourism' → 'World Transport' mistranslations.
    The LLM is never called for a glossary term."""
    class _NoLLM:
        async def chat(self, **kw):
            raise AssertionError("LLM must not be called for a glossary term")

    i18n._CACHE.clear()
    cats = ["Hospitality & Tourism", "Manufacturing", "Accounting & Financee"]
    tr = await i18n.translate_many(_NoLLM(), cats, to_lang="ta")
    assert tr[0] == "விருந்தோம்பல் & சுற்றுலா"           # NOT "World Transport"
    assert tr[2] == "கணக்கியல் & நிதி"                   # the 'Financee' typo handled
    # Hindi too
    assert (await i18n.translate_many(_NoLLM(), ["Hospitality & Tourism"], to_lang="hi"))[0] == "आतिथ्य और पर्यटन"
    # reverse round-trip works off the glossary (no LLM)
    assert await i18n.category_from_translation(_NoLLM(), "விருந்தோம்பல் & சுற்றுலா", cats, "ta") \
        == "Hospitality & Tourism"


async def test_glossary_covers_menu_labels():
    """Fixed menu/button labels translate via the glossary (correct + reliable, no
    LLM) — fixes 'Application Status' → garbage and untranslated buttons."""
    class _NoLLM:
        async def chat(self, **kw):
            raise AssertionError("menu labels must not call the LLM")

    i18n._CACHE.clear()
    out = await i18n.translate_many(
        _NoLLM(), ["Job Search", "Application Status", "Recommended Jobs",
                   "View Candidates", "🪪 Credits & Wallet"], to_lang="ta")
    assert out == ["வேலை தேடல்", "விண்ணப்ப நிலை", "பரிந்துரைக்கப்பட்ட வேலைகள்",
                   "விண்ணப்பத்தார்களை பார்க்க", "🪪 கிரெடிட்கள் & வாலெட்"]
    assert (await i18n.translate_many(_NoLLM(), ["Application Status"], to_lang="hi"))[0] == "आवेदन स्थिति"


async def test_warm_cache_pretranslates_registered_strings():
    """Registered fixed strings are pre-translated into every language at startup,
    so later turns serve them from cache (no per-turn LLM) — even if the LLM later
    fails. Guards against an occasional English body for a non-English user."""
    import json as _json

    class _LLM:
        def __init__(self): self.calls = 0
        async def chat(self, *, purpose, messages, **kw):
            self.calls += 1
            payload = _json.loads(messages[-1]["content"])
            return _json.dumps({k: "X::" + v for k, v in payload.items()}), {}

    i18n._CACHE.clear()
    i18n._WARM_STRINGS.clear()
    i18n.register_warm_strings(["You're on the Job Seeker side.", "12345", ""])  # junk skipped
    assert i18n._WARM_STRINGS == {"You're on the Job Seeker side."}

    llm = _LLM()
    assert await i18n.warm_cache(llm, pace_s=0) == 2   # 1 string × ta + hi cached
    # after warming, a translation is a cache hit — the LLM must NOT be called again
    class _NoLLM:
        async def chat(self, **k):
            raise AssertionError("should be served from the warmed cache")
    out = await i18n.translate_many(_NoLLM(), ["You're on the Job Seeker side."], to_lang="ta")
    assert out == ["X::You're on the Job Seeker side."]


async def test_to_english_reverse_maps_localized_labels():
    """Typing a LOCALIZED menu/category label reverse-maps to its canonical English
    (no LLM), so it routes exactly like tapping the button. Fixes 'tap Post-a-Job
    works but typing வேலை இடுகையிடு does a candidate search'."""
    class _NoLLM:
        async def chat(self, **k):
            raise AssertionError("glossary reverse-map must not call the LLM")

    assert i18n.from_glossary("வேலை பதிவிடு", "ta") == "post a job"
    assert i18n.from_glossary("उम्मीदवार देखें", "hi") == "view candidates"
    assert i18n.from_glossary("not a label", "ta") is None           # unknown phrase
    assert i18n.from_glossary("anything", "en") is None              # English: no-op
    # to_english uses the reverse-map (no LLM) for a known label
    assert await i18n.to_english(_NoLLM(), "விண்ணப்பத்தார்களை பார்க்க", source_lang="ta") == "view candidates"
    # an unknown phrase falls through to the LLM (here stubbed to raise → best-effort original)
    assert await i18n.to_english(_NoLLM(), "சும்மா ஒரு வாக்கியம்", source_lang="ta") == "சும்மா ஒரு வாக்கியம்"


async def test_people_nouns_reverse_map_to_clean_english():
    """'applicant'/'candidate'/'job seeker' typed in Tamil/Hindi reverse-map to clean
    English (no LLM), so an employer typing 'விண்ணப்பதாரர்' reaches View Candidates
    instead of a junk-translated failed search."""
    assert i18n.from_glossary("விண்ணப்பதாரர்", "ta") == "applicant"
    assert i18n.from_glossary("வேட்பாளர்", "ta") == "candidate"
    assert i18n.from_glossary("वेலை தேடுபவர்", "ta") in (None, "job seeker")  # tolerant
    assert i18n.from_glossary("वेलै தேடுபவர்", "ta") in (None, "job seeker")
    assert i18n.from_glossary("வேலை தேடுபவர்", "ta") == "job seeker"
    assert i18n.from_glossary("आवेदक", "hi") in ("applicant", "applicants")
    assert i18n.from_glossary("उम्मीदवार", "hi") in ("candidate", "candidates")


def test_t_deterministic_glossary_translation():
    """i18n.t() translates fixed strings via the glossary with NO LLM — the basis for
    rendering the candidate card reliably (no flaky English) in the user's language."""
    assert i18n.t("*Candidates available* 👥", "ta") == "*கிடைக்கும் வேட்பாளர்கள்* 👥"
    assert i18n.t("Experience not specified", "ta") == "அனுபவம் குறிப்பிடப்படவில்லை"
    assert i18n.t("Freshers", "hi") == "फ्रेशर्स"
    assert i18n.t("years", "ta") == "ஆண்டுகள்"
    assert i18n.t("Some Unknown Phrase", "ta") == "Some Unknown Phrase"   # not in glossary → original
    assert i18n.t("anything", "en") == "anything"                        # English → no-op


def test_inject_form_i18n_builds_translator_map():
    """inject_form_i18n adds a client-side translator + a map of fixed form labels
    (from the glossary/warmed cache) before </body>; English / unsupported is a no-op;
    only catalog strings are mapped (user data is never touched)."""
    i18n._cache_put(("Company name", "ta"), "நிறுவனப் பெயர்")
    i18n._cache_put(("Email", "ta"), "மின்னஞ்சல்")
    html = "<html><body><form><label>Company name</label></form></body></html>"
    out = i18n.inject_form_i18n(html, "ta")
    assert "__FORMI18N__" in out and "நிறுவனப் பெயர்" in out
    assert out.endswith("</body></html>")                    # injected before </body>
    # the catalog is registered for warming, and English / unknown lang are no-ops
    assert "Company name" in i18n._FORM_STRINGS
    assert i18n.inject_form_i18n(html, "en") == html
    assert i18n.inject_form_i18n(html, "fr") == html


async def test_translate_many_no_perstring_storm_on_api_error():
    """An API error (e.g. a 429 rate limit) on the batch must NOT trigger a per-string
    fallback storm (that only multiplies the rate-limit hits). One batch attempt,
    zero per-string calls, originals kept. (A malformed-JSON batch still falls back —
    test_translate_many_falls_back_per_string_on_bad_batch.)"""
    class _RateLimited:
        def __init__(self): self.batch = 0; self.singles = 0
        async def chat(self, *, purpose, messages, **kw):
            if purpose == "translate_out":
                self.batch += 1
                raise RuntimeError("429 Too Many Requests")
            self.singles += 1
            raise RuntimeError("429")

    i18n._CACHE.clear()
    llm = _RateLimited()
    # non-glossary strings so the LLM batch path is exercised (glossary terms skip it)
    texts = ["Welcome to the team", "How can I help you today", "See you soon"]
    out = await i18n.translate_many(llm, texts, to_lang="ta")
    assert out == texts                                  # originals kept
    assert llm.batch == 1 and llm.singles == 0           # no per-string amplification


async def test_translate_many_json_400_recovers_via_per_string():
    """A 400 'failed to generate JSON' (Groq json_object quirk) is NOT a rate limit:
    the plain-text per-string fallback avoids JSON entirely, so those strings should
    still get translated instead of silently staying English."""
    class _JsonModeFails:
        def __init__(self): self.batch = 0; self.singles = 0
        async def chat(self, *, purpose, messages, **kw):
            if purpose == "translate_out":
                self.batch += 1
                raise RuntimeError(
                    "Error code: 400 - {'error': {'message': 'Failed to generate JSON. "
                    "Please adjust your prompt.'}}")
            self.singles += 1
            return "TA::" + messages[-1]["content"], {}   # plain-text single succeeds

    i18n._CACHE.clear()
    llm = _JsonModeFails()
    texts = ["Welcome to the team", "How can I help you today"]
    out = await i18n.translate_many(llm, texts, to_lang="ta")
    assert out == ["TA::Welcome to the team", "TA::How can I help you today"]
    assert llm.batch == 1 and llm.singles == 2           # batch failed → per-string recovered
