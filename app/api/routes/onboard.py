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

import asyncio
import html
import json
from typing import Any
from urllib.parse import parse_qs

import re

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.api.deps import get_memory
from app.config import get_settings
from app.core.logging import get_logger
from app.db.repositories import LookupRepository
from app.onboarding import prepare_registration

router = APIRouter()
log = get_logger("onboard")

# Dropdown sources fetched to render the form (id-valued options).
_OPTION_KINDS = (
    "states", "districts", "education_levels", "courses", "specializations",
    "experience_levels", "skills", "roles", "categories",
)

# Fixed enum options (stored as plain text columns on private_job_seekers).
_GENDER = (("MALE", "Male"), ("FEMALE", "Female"), ("OTHER", "Other"))
_MARITAL = (("SINGLE", "Single"), ("MARRIED", "Married"))
_ENGLISH = (("BASIC", "Basic"), ("INTERMEDIATE", "Intermediate"), ("FLUENT", "Fluent"))
_CURRENT_STATUS = (("STUDENT", "Student"), ("WORKING", "Working"))


async def _load_options() -> dict[str, list[dict[str, Any]]]:
    """Fetch every dropdown's options concurrently (read-only reference data)."""
    results = await asyncio.gather(*(LookupRepository.options(k) for k in _OPTION_KINDS))
    return dict(zip(_OPTION_KINDS, results))


@router.get("/form", response_class=HTMLResponse)
async def onboarding_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_onboarding_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    opts = await _load_options()
    return HTMLResponse(_form_html(token, identity.get("name") or "", opts))


@router.post("/submit", response_class=HTMLResponse)
async def onboarding_submit(request: Request) -> HTMLResponse:
    raw = parse_qs((await request.body()).decode("utf-8"))

    def one(key: str) -> str:
        v = raw.get(key) or []
        return v[0].strip() if v else ""

    def many(key: str) -> list[str]:
        return [x.strip() for x in (raw.get(key) or []) if x.strip()]

    token = one("token")
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)

    name = one("full_name") or one("name")

    async def _reject(message: str) -> HTMLResponse:
        return HTMLResponse(
            _form_html(token, name, await _load_options(), error=message),
            status_code=400,
        )

    email = one("email")
    if "@" not in email or "." not in email.split("@")[-1]:
        return await _reject("Please enter a valid email address.")
    if not name:
        return await _reject("Please enter your full name.")
    preferred_location_ids = many("preferred_location_ids")
    if not preferred_location_ids:
        return await _reject("Please choose at least one preferred location.")

    form = {
        # name kept under both keys (identify reads "name"; staging reads "full_name")
        "name": name,
        "full_name": name,
        "email": email,
        "gender": one("gender"),
        "marital_status": one("marital_status"),
        "state_id": one("state_id"),
        "district_id": one("district_id"),
        "current_status": one("current_status"),
        "current_year_of_study": one("current_year_of_study"),
        "education_level_id": one("education_level_id"),
        "course_id": one("course_id"),
        "specialization_id": one("specialization_id"),
        "experience_level_id": one("experience_level_id"),
        "current_salary": one("current_salary"),
        "expected_salary": one("expected_salary"),
        "english_proficiency": one("english_proficiency"),
        "skill_ids": many("skill_ids"),
        "preferred_location_ids": preferred_location_ids,
        "preferred_category_ids": many("preferred_category_ids"),
        "preferred_role_ids": many("preferred_role_ids"),
    }
    # Back-compat text fields the recommendation menu reads (first selection).
    first_role = (form["preferred_role_ids"] or [None])[0]
    form["preferred_role"] = await LookupRepository.name_for("roles", first_role) or ""
    form["location"] = await LookupRepository.name_for("districts", preferred_location_ids[0]) or ""

    memory = get_memory(request)
    identity = await memory.save_onboarding(token, form)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)

    # Build the DB-ready registration payload (private_job_seekers +
    # job_seeker_profiles + child rows, with role/location resolved to FK ids by
    # READING the lookup tables) and STAGE it in Redis. Nothing is written to the
    # business DB yet — this lets us verify the shape before enabling real INSERTs.
    try:
        payload = await prepare_registration(identity=identity, form=form)
        await memory.save_registration(
            identity["conversation_id"], payload, tenant_id=identity["tenant_id"]
        )
    except Exception as exc:  # noqa: BLE001 — staging must never fail the submission
        log.warning("registration_stage_failed", error=str(exc)[:200])

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
       margin:0;display:flex;min-height:100vh;align-items:flex-start;justify-content:center;padding:18px 0}}
  .card{{background:#111b21;max-width:460px;width:92%;padding:24px 22px;border-radius:14px;
         box-shadow:0 10px 30px rgba(0,0,0,.4)}}
  h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:#8696a0;margin:0 0 16px;font-size:14px}}
  fieldset{{border:1px solid #2a3942;border-radius:10px;margin:0 0 16px;padding:6px 14px 16px}}
  legend{{padding:0 8px;font-size:13px;font-weight:600;color:#00d3a7;text-transform:uppercase;letter-spacing:.04em}}
  label{{display:block;font-size:13px;color:#aebac1;margin:14px 0 6px}}
  label .req{{color:#ff8a8a}}
  input,select{{width:100%;box-sizing:border-box;padding:11px 12px;border-radius:8px;border:1px solid #2a3942;
         background:#202c33;color:#e9edef;font-size:15px}}
  select[multiple]{{height:auto;min-height:104px}}
  .hint{{font-size:12px;color:#6b7d88;margin:6px 0 0}}
  button,a.btn{{display:block;width:100%;box-sizing:border-box;margin-top:8px;padding:12px;border:0;
          border-radius:8px;background:#00a884;color:#fff;font-size:16px;font-weight:600;cursor:pointer;
          text-align:center;text-decoration:none}}
  .err{{background:#3a1d1d;color:#ffb4b4;padding:10px 12px;border-radius:8px;font-size:13px;margin-bottom:12px}}
  .ok{{text-align:center}} .ok .tick{{font-size:44px}}
  .ts{{position:relative}}
  .ts-box{{display:flex;flex-wrap:wrap;gap:6px;min-height:44px;padding:7px 8px;border:1px solid #2a3942;
           border-radius:8px;background:#202c33;align-items:center;cursor:text}}
  .ts-chip{{display:inline-flex;align-items:center;gap:7px;background:#005c4b;color:#e9edef;border-radius:14px;
            padding:3px 6px 3px 11px;font-size:13px}}
  .ts-chip b{{cursor:pointer;font-weight:700;opacity:.85;font-size:15px;line-height:1}}
  .ts-input{{flex:1;min-width:90px;border:0;background:transparent;color:#e9edef;font-size:15px;outline:none;padding:4px 2px}}
  .ts-menu{{position:absolute;left:0;right:0;top:calc(100% + 4px);z-index:30;background:#0b141a;border:1px solid #2a3942;
            border-radius:8px;max-height:220px;overflow:auto;box-shadow:0 8px 24px rgba(0,0,0,.5)}}
  .ts-opt{{padding:9px 12px;font-size:14px;cursor:pointer}}
  .ts-opt.active,.ts-opt:hover{{background:#202c33}}
  .ts-empty{{padding:9px 12px;color:#6b7d88;font-size:13px}}
</style></head><body><div class="card">{body}</div></body></html>"""


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _options_html(items: "list[dict[str, Any]] | tuple", *, placeholder: str | None = None) -> str:
    """``<option>`` tags. ``items`` is either id/name dicts (value=id) or
    (value, label) tuples (fixed enums). ``placeholder`` adds an empty first row."""
    out: list[str] = []
    if placeholder is not None:
        out.append(f'<option value="">{_esc(placeholder)}</option>')
    for it in items:
        val, lab = (it["id"], it["name"]) if isinstance(it, dict) else (it[0], it[1])
        out.append(f'<option value="{_esc(val)}">{_esc(lab)}</option>')
    return "".join(out)


def _js_rows(items: list[dict[str, Any]]) -> str:
    """``[[id, name, parentId], …]`` as a JS-safe JSON literal — feeds both the
    cascading selects and the type-to-search token pickers."""
    rows = [[it["id"], it["name"], it.get("parent")] for it in items]
    return json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")


# Inline, dependency-free form behaviour: (1) cascade — a parent <select> fills
# its child (state→district, course→specialization); (2) tokenSelect — a
# type-to-search multi picker that renders chips + hidden inputs (skills,
# locations, categories, roles). Raw string so Python leaves the JS untouched.
_FORM_JS = r"""
function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function fillSelect(sel, rows, ph){
  if(!sel) return;
  sel.innerHTML = "";
  var first = document.createElement("option"); first.value = ""; first.textContent = ph;
  sel.appendChild(first);
  rows.forEach(function(r){
    var o = document.createElement("option"); o.value = r[0]; o.textContent = r[1]; sel.appendChild(o);
  });
}
function cascade(parentId, childId, rows, ph){
  var p = document.getElementById(parentId), c = document.getElementById(childId);
  if(!p || !c) return;
  p.addEventListener("change", function(){
    fillSelect(c, rows.filter(function(r){ return String(r[2]) === String(p.value); }), ph);
  });
}
function tokenSelect(host, name, options, ph){
  if(!host) return;
  host.classList.add("ts");
  var box = document.createElement("div"); box.className = "ts-box";
  var input = document.createElement("input");
  input.className = "ts-input"; input.type = "text"; input.placeholder = ph; input.autocomplete = "off";
  var menu = document.createElement("div"); menu.className = "ts-menu"; menu.style.display = "none";
  box.appendChild(input); host.appendChild(box); host.appendChild(menu);
  var chosen = {}, filtered = [], active = -1;
  box.addEventListener("click", function(){ input.focus(); });
  function add(opt){
    if(chosen[opt[0]]) return;
    chosen[opt[0]] = 1;
    var chip = document.createElement("span"); chip.className = "ts-chip";
    chip.appendChild(document.createTextNode(opt[1]));
    var x = document.createElement("b"); x.textContent = "×"; chip.appendChild(x);
    var hid = document.createElement("input"); hid.type = "hidden"; hid.name = name; hid.value = opt[0];
    x.addEventListener("click", function(e){
      e.stopPropagation(); delete chosen[opt[0]]; box.removeChild(chip); host.removeChild(hid);
    });
    box.insertBefore(chip, input); host.appendChild(hid);
    input.value = ""; render(); input.focus();
  }
  function render(){
    var q = input.value.trim().toLowerCase();
    filtered = options.filter(function(o){
      if(chosen[o[0]]) return false;
      return q === "" ? true : o[1].toLowerCase().indexOf(q) >= 0;
    }).slice(0, 12);
    if(!filtered.length){ menu.innerHTML = '<div class="ts-empty">No matches</div>'; return; }
    menu.innerHTML = filtered.map(function(o, i){
      var sub = o[2] ? ' <span style="color:#6b7d88">- ' + _esc(o[2]) + '</span>' : '';
      return '<div class="ts-opt' + (i === active ? ' active' : '') + '" data-i="' + i + '">'
        + _esc(o[1]) + sub + '</div>';
    }).join("");
  }
  function openMenu(){ active = -1; render(); menu.style.display = "block"; }
  input.addEventListener("focus", openMenu);
  input.addEventListener("input", openMenu);
  input.addEventListener("keydown", function(e){
    if(e.key === "ArrowDown"){ active = Math.min(active + 1, filtered.length - 1); render(); e.preventDefault(); }
    else if(e.key === "ArrowUp"){ active = Math.max(active - 1, 0); render(); e.preventDefault(); }
    else if(e.key === "Enter"){ if(active >= 0 && filtered[active]) add(filtered[active]); e.preventDefault(); }
    else if(e.key === "Backspace" && input.value === ""){
      var c = box.querySelectorAll(".ts-chip"); if(c.length) c[c.length - 1].querySelector("b").click();
    }
  });
  menu.addEventListener("mousedown", function(e){
    var t = e.target.closest(".ts-opt"); if(t){ add(filtered[+t.getAttribute("data-i")]); e.preventDefault(); }
  });
  document.addEventListener("click", function(e){ if(!host.contains(e.target)) menu.style.display = "none"; });
}

cascade("state_id", "district_id", DISTRICTS, "Select district...");
cascade("course_id", "specialization_id", SPECS, "Select specialization...");

var STATE_NAME = {}; STATES.forEach(function(s){ STATE_NAME[s[0]] = s[1]; });
var LOC_OPTS = DISTRICTS.map(function(d){ return [d[0], d[1], STATE_NAME[d[2]] || ""]; });

tokenSelect(document.getElementById("ts_skills"), "skill_ids", SKILLS, "Type a skill...");
tokenSelect(document.getElementById("ts_locations"), "preferred_location_ids", LOC_OPTS, "Type a district...");
tokenSelect(document.getElementById("ts_categories"), "preferred_category_ids", CATEGORIES, "Type a category...");
tokenSelect(document.getElementById("ts_roles"), "preferred_role_ids", ROLES, "Type a job role...");
"""


def _form_html(
    token: str, name: str, opts: dict[str, list[dict[str, Any]]], *, error: str = ""
) -> str:
    safe_name = _esc(name)
    err = f'<div class="err">{_esc(error)}</div>' if error else ""
    greeting = f"Hi {safe_name}, " if safe_name else ""

    o = opts  # shorthand
    body = f"""\
<h1>Complete your profile</h1>
<p class="sub">{greeting}fill these so we can match you with the right roles. Fields marked <span class="req">*</span> are required.</p>
{err}
<form method="post" action="/onboard/submit">
  <input type="hidden" name="token" value="{_esc(token)}">

  <fieldset><legend>1 · Basic info</legend>
    <label>Full name <span class="req">*</span></label>
    <input type="text" name="full_name" value="{safe_name}" placeholder="Your full name" required>
    <label>Email <span class="req">*</span></label>
    <input type="email" name="email" placeholder="you@example.com" required>
    <label>Gender</label>
    <select name="gender">{_options_html(_GENDER, placeholder="Select…")}</select>
    <label>Marital status</label>
    <select name="marital_status">{_options_html(_MARITAL, placeholder="Select…")}</select>
    <label>State</label>
    <select name="state_id" id="state_id">{_options_html(o["states"], placeholder="Select state…")}</select>
    <label>District</label>
    <select name="district_id" id="district_id"><option value="">Select a state first…</option></select>
    <label>Current status</label>
    <select name="current_status">{_options_html(_CURRENT_STATUS, placeholder="Select…")}</select>
    <label>Year of study <span class="hint">(if a student)</span></label>
    <input type="number" name="current_year_of_study" min="1" max="6" placeholder="e.g. 3">
  </fieldset>

  <fieldset><legend>2 · Education</legend>
    <label>Education level</label>
    <select name="education_level_id">{_options_html(o["education_levels"], placeholder="Select…")}</select>
    <label>Course</label>
    <select name="course_id" id="course_id">{_options_html(o["courses"], placeholder="Select…")}</select>
    <label>Specialization</label>
    <select name="specialization_id" id="specialization_id"><option value="">Select a course first…</option></select>
  </fieldset>

  <fieldset><legend>3 · Experience</legend>
    <label>Experience level</label>
    <select name="experience_level_id">{_options_html(o["experience_levels"], placeholder="Select…")}</select>
    <label>Current monthly salary (₹)</label>
    <input type="number" name="current_salary" min="0" step="500" placeholder="e.g. 18000">
    <label>Expected monthly salary (₹)</label>
    <input type="number" name="expected_salary" min="0" step="500" placeholder="e.g. 25000">
    <label>English proficiency</label>
    <select name="english_proficiency">{_options_html(_ENGLISH, placeholder="Select…")}</select>
  </fieldset>

  <fieldset><legend>4 · Skills</legend>
    <label>Skills</label>
    <div id="ts_skills"></div>
    <p class="hint">Type to search, then tap a suggestion to add it.</p>
  </fieldset>

  <fieldset><legend>5 · Preferences</legend>
    <label>Preferred locations <span class="req">*</span></label>
    <div id="ts_locations"></div>
    <label>Preferred job categories</label>
    <div id="ts_categories"></div>
    <label>Preferred job roles</label>
    <div id="ts_roles"></div>
    <p class="hint">Type to search; pick at least one preferred location.</p>
  </fieldset>

  <button type="submit">Submit</button>
</form>
<script>
const SKILLS = {_js_rows(o["skills"])};
const ROLES = {_js_rows(o["roles"])};
const CATEGORIES = {_js_rows(o["categories"])};
const STATES = {_js_rows(o["states"])};
const DISTRICTS = {_js_rows(o["districts"])};
const SPECS = {_js_rows(o["specializations"])};
</script>
<script>{_FORM_JS}</script>"""
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
  <h1>Profile submitted successfully{who}!</h1>
  <p class="sub">Your details are saved. Tap below to head back to WhatsApp —
  we'll start matching you with roles right away.</p>
  {close}
</div>"""
    return _PAGE.format(body=body)


def _expired_html() -> str:
    body = """\
<h1>Link expired</h1>
<p class="sub">This form link is invalid or has expired. Please go back to
WhatsApp and message us so we can send you a fresh one.</p>"""
    return _PAGE.format(body=body)
