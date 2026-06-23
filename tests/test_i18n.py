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
    out = await i18n.translate_many(llm, ["Post a Job", "12345", "Buy Credits"], to_lang="ta")
    assert out == ["TA::Post a Job", "12345", "TA::Buy Credits"]   # digits-only skipped
    assert llm.calls == ["translate_out"]        # ONE batched call for the two strings
    # a repeat is served from cache — no new LLM call
    llm2 = _FakeLLM()
    assert await i18n.translate_many(llm2, ["Post a Job"], to_lang="ta") == ["TA::Post a Job"]
    assert llm2.calls == []


async def test_translate_many_best_effort_on_error():
    i18n._CACHE.clear()
    out = await i18n.translate_many(_BoomLLM(), ["Hello", "World"], to_lang="ta")
    assert out == ["Hello", "World"]             # originals kept on error


async def test_to_english_noop_and_translate():
    llm = _FakeLLM()
    assert await i18n.to_english(llm, "hello", source_lang="en") == "hello"
    assert llm.calls == []                       # English in → no call
    out = await i18n.to_english(llm, "வணக்கம்", source_lang="ta")
    assert out == "EN[வணக்கம்]" and llm.calls == ["translate_in"]


async def test_to_english_best_effort_on_error():
    assert await i18n.to_english(_BoomLLM(), "வணக்கம்", source_lang="ta") == "வணக்கம்"


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
