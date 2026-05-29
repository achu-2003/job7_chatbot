"""Identity extraction — name/email pulled from a turn so the bot stops
re-asking (regression for the WhatsApp name-loop)."""
from app.agent.identity import extract_email, extract_name


def test_extracts_plain_email():
    assert extract_email("sandhanapandiyanmurugan@outlook.com") == \
        "sandhanapandiyanmurugan@outlook.com"


def test_extracts_email_in_sentence():
    assert extract_email("you can reach me at Foo.Bar+jobs@gmail.com thanks") == \
        "foo.bar+jobs@gmail.com"


def test_no_email_returns_none():
    assert extract_email("I need a python developer job") is None


def test_bare_name_reply_after_name_prompt():
    # The exact screenshot case: bot asked for the name, user replied with it.
    assert extract_name(
        "Sandhanapandiyanmurugan", assistant_prompt="Need your full name to proceed."
    ) == "Sandhanapandiyanmurugan"


def test_name_lead_in_anywhere():
    assert extract_name("my name is ravi kumar") == "Ravi Kumar"
    assert extract_name("I'm Priya") == "Priya"


def test_bare_name_ignored_without_a_name_prompt():
    # Without the assistant having asked, a lone word is too risky to store.
    assert extract_name("Sandhanapandiyanmurugan", assistant_prompt=None) is None


def test_intent_phrase_not_stored_as_name():
    # The other screenshot bug: this was treated as a name. It must not be.
    assert extract_name(
        "I need a python developer job", assistant_prompt="what's your full name?"
    ) is None


def test_greeting_not_stored_as_name():
    assert extract_name("Hi", assistant_prompt="what's your full name?") is None


def test_email_reply_not_stored_as_name():
    assert extract_name(
        "sandhanapandiyanmurugan@outlook.com",
        assistant_prompt="what's your email?",
    ) is None
