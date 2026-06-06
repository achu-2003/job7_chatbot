"""Self-hosted onboarding form for new WhatsApp candidates.

A new number is asked for their name in chat, then handed a tokenised link to
this form (email, years of experience, preferred role/location). The submission
is stored in **Redis only** (never the business DB) and acts as the gate that
lets the agent start helping with jobs.

    GET  /onboard/form?token=...   → the HTML form (token resolves to the
                                     candidate; 404 if unknown/expired)
    POST /onboard/submit           → validate + store in Redis, show a thank-you

Form bodies are parsed manually (urlencoded) so we don't depend on
``python-multipart``.
"""
from __future__ import annotations

import html
from urllib.parse import parse_qs

import re

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.api.deps import get_memory
from app.config import get_settings
from app.core.logging import get_logger

router = APIRouter()
log = get_logger("onboard")


@router.get("/form", response_class=HTMLResponse)
async def onboarding_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_onboarding_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    return HTMLResponse(_form_html(token, identity.get("name") or ""))


@router.post("/submit", response_class=HTMLResponse)
async def onboarding_submit(request: Request) -> HTMLResponse:
    fields = {
        k: (v[0] if v else "")
        for k, v in parse_qs((await request.body()).decode("utf-8")).items()
    }
    token = fields.get("token", "").strip()
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)

    email = (fields.get("email") or "").strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        return HTMLResponse(
            _form_html(token, fields.get("name", ""), error="Please enter a valid email address."),
            status_code=400,
        )

    identity = await get_memory(request).save_onboarding(
        token,
        {
            "email": email,
            "years_experience": fields.get("years_experience", ""),
            "preferred_role": fields.get("preferred_role", ""),
            "location": fields.get("location", ""),
        },
    )
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    number = re.sub(r"\D", "", get_settings().whatsapp_business_number or "")
    return HTMLResponse(_success_html(identity.get("name") or "", business_number=number))


# ---------------------------------------------------------------------------
# HTML (inline; no template engine dependency)
# ---------------------------------------------------------------------------

_PAGE = """\
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Complete your profile</title>
<style>
  body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0b141a;color:#e9edef;
       margin:0;display:flex;min-height:100vh;align-items:center;justify-content:center}}
  .card{{background:#111b21;max-width:420px;width:92%;padding:28px 24px;border-radius:14px;
         box-shadow:0 10px 30px rgba(0,0,0,.4)}}
  h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:#8696a0;margin:0 0 20px;font-size:14px}}
  label{{display:block;font-size:13px;color:#aebac1;margin:14px 0 6px}}
  input{{width:100%;box-sizing:border-box;padding:11px 12px;border-radius:8px;border:1px solid #2a3942;
         background:#202c33;color:#e9edef;font-size:15px}}
  button,a.btn{{display:block;width:100%;box-sizing:border-box;margin-top:22px;padding:12px;border:0;
          border-radius:8px;background:#00a884;color:#fff;font-size:16px;font-weight:600;cursor:pointer;
          text-align:center;text-decoration:none}}
  .err{{background:#3a1d1d;color:#ffb4b4;padding:10px 12px;border-radius:8px;font-size:13px;margin-bottom:8px}}
  .ok{{text-align:center}} .ok .tick{{font-size:44px}}
</style></head><body><div class="card">{body}</div></body></html>"""


def _form_html(token: str, name: str, *, error: str = "") -> str:
    safe_name = html.escape(name)
    err = f'<div class="err">{html.escape(error)}</div>' if error else ""
    greeting = f"Hi {safe_name}, " if safe_name else ""
    body = f"""\
<h1>Complete your profile</h1>
<p class="sub">{greeting}just a few details so we can match you with the right roles.</p>
{err}
<form method="post" action="/onboard/submit">
  <input type="hidden" name="token" value="{html.escape(token)}">
  <input type="hidden" name="name" value="{safe_name}">
  <label>Email address</label>
  <input type="email" name="email" placeholder="you@example.com" required>
  <label>Years of experience</label>
  <input type="number" name="years_experience" min="0" step="1" placeholder="e.g. 3">
  <label>Preferred role</label>
  <input type="text" name="preferred_role" placeholder="e.g. Backend Engineer">
  <label>Preferred location</label>
  <input type="text" name="location" placeholder="e.g. Chennai">
  <button type="submit">Submit</button>
</form>"""
    return _PAGE.format(body=body)


def _success_html(name: str, *, business_number: str = "") -> str:
    who = f", {html.escape(name)}" if name else ""
    # A "Back to chat" button that returns the candidate to WhatsApp. With the
    # business number configured it's a wa.me deep link (pre-filled "Hi" so one
    # tap sends it and the bot replies with the welcome + menu); otherwise it's a
    # best-effort window.close(). This button only appears AFTER a submission, so
    # a candidate who merely opens the form and leaves is never let past the gate.
    if business_number:
        # Reliable: navigate back to WhatsApp (the app reopens the chat). A web
        # page cannot force-close a tab the user navigated to, so we redirect
        # rather than call window.close().
        close = (
            f'<a class="btn" href="https://wa.me/{business_number}?text=Hi">'
            "Back to chat</a>"
        )
    else:
        # No business number configured → we can't deep-link back. window.close()
        # is blocked for user-opened tabs, so we attempt it but ALSO tell the
        # candidate how to return, instead of leaving a dead button.
        close = (
            '<button onclick="window.close()">Close</button>'
            '<p class="sub" style="margin-top:14px">All done! Return to WhatsApp '
            "(tap the back arrow or &#10005; at the top) and send us a message.</p>"
        )
    body = f"""\
<div class="ok">
  <div class="tick">&#10003;</div>
  <h1>You're all set{who}!</h1>
  <p class="sub">Thanks for completing your profile. Tap below to head back to the
  chat — we'll start finding you roles right away.</p>
  {close}
</div>"""
    return _PAGE.format(body=body)


def _expired_html() -> str:
    body = """\
<h1>Link expired</h1>
<p class="sub">This form link is invalid or has expired. Please go back to
WhatsApp and message us so we can send you a fresh one.</p>"""
    return _PAGE.format(body=body)
