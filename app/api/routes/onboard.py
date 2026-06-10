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
import json
from typing import Any
from urllib.parse import parse_qs

import re

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.api.deps import get_memory
from app.chatbot import wa_format as wa
from app.config import get_settings
from app.core.logging import get_logger
from app.db.repositories import JobSeekerRepository, LookupRepository
from app.onboarding import prepare_registration
from app.whatsapp import delivery as wa_delivery

# The /onboard form is the SEEKER lane (the employer side has its own form), so
# after a submission we push the seeker hub directly — the conversation just
# continues with Job Search / Application Status / Recommended Jobs. Ids must
# match app.agent.runtime._MENU_BUTTONS.
_SEEKER_HUB_BUTTONS = (
    ("menu_search", "Job Search"),
    ("menu_status", "Application Status"),
    ("menu_recommend", "Recommended Jobs"),
)

router = APIRouter()
log = get_logger("onboard")

# Dropdown sources fetched to render the form (id-valued options).
_OPTION_KINDS = (
    "states", "districts", "education_levels", "courses", "specializations",
    "experience_levels", "skills", "roles", "categories", "other_states",
    "languages",
)

# Fixed enum options (stored as plain text / array columns on the seeker/profile).
_I_AM_A = (("STUDENT", "Student"), ("FRESHER", "Fresher - First Job"), ("EXPERIENCED", "Experienced"))
_GENDER = (("MALE", "Male"), ("FEMALE", "Female"), ("OTHER", "Other"))
_MARITAL = (("SINGLE", "Single"), ("MARRIED", "Married"))
_ENGLISH = (("BASIC", "Basic"), ("INTERMEDIATE", "Intermediate"), ("FLUENT", "Fluent"))
_JOB_TYPES = (
    ("FULL_TIME", "Full-time"), ("PART_TIME", "Part-time"), ("CONTRACT", "Contract"),
    ("INTERNSHIP", "Internship"), ("FREELANCE", "Freelance"), ("TEMPORARY", "Temporary"),
    ("WORK_FROM_HOME", "Work From Home"), ("WALK_IN", "Walk-In"),
)
_WORK_MODE = (("ON_SITE", "On-site"), ("REMOTE", "Remote"), ("HYBRID", "Hybrid"))
_YESNO = (("yes", "Yes"), ("no", "No"))
# value = the lower bound (₹/month), stored as expected_salary.
_SALARY = (
    ("5000", "₹5,000 - ₹10,000"), ("10000", "₹10,000 - ₹15,000"),
    ("15000", "₹15,000 - ₹20,000"), ("20000", "₹20,000 - ₹25,000"),
    ("25000", "₹25,000 - ₹35,000"), ("35000", "₹35,000 - ₹50,000"),
    ("50000", "₹50,000 - ₹75,000"), ("75000", "₹75,000 - ₹1,00,000"),
    ("100000", "₹1,00,000+"),
)


async def _load_options() -> dict[str, list[dict[str, Any]]]:
    """Fetch every dropdown's options (read-only reference data).

    Loaded SEQUENTIALLY, not concurrently: firing all lookups at once opened a
    Postgres connection per query and tripped 'sorry, too many clients already',
    so some lists silently came back empty (skills/categories/roles/specs blank).
    One connection at a time is plenty fast for a form render and never starves
    the pool.
    """
    return {k: await LookupRepository.options(k) for k in _OPTION_KINDS}


@router.get("/form", response_class=HTMLResponse)
async def onboarding_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_onboarding_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    opts = await _load_options()
    return HTMLResponse(
        _form_html(token, identity.get("name") or "", identity.get("customer_id") or "", opts)
    )


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
    # Languages: each can be marked Speak and/or Write (checkboxes by language id).
    speak_ids, write_ids = set(many("lang_speak")), set(many("lang_write"))
    languages = [
        {
            "languageId": lid,
            "speak": "FLUENT" if lid in speak_ids else None,
            "write": "FLUENT" if lid in write_ids else None,
        }
        for lid in (speak_ids | write_ids)
    ]

    # The wizard's "Skip for now" lets any step be skipped, so the server accepts
    # partial data — the client enforces the per-step required (*) fields on Next.
    form = {
        # name kept under both keys (identify reads "name"; staging reads "full_name")
        "name": name,
        "full_name": name,
        "email": one("email"),
        "current_status": one("current_status"),
        "gender": one("gender"),
        "marital_status": one("marital_status"),
        "date_of_birth": one("date_of_birth"),
        "state_id": one("state_id"),
        "district_id": one("district_id"),
        "city": one("city"),
        "education_level_id": one("education_level_id"),
        "course_id": one("course_id"),
        "specialization_id": one("specialization_id"),
        "institution": one("institution"),
        "year_of_passing": one("year_of_passing"),
        "experience_level_id": one("experience_level_id"),
        "current_salary": one("current_salary"),
        "expected_salary": one("expected_salary"),
        "resume": one("resume"),
        "work_mode": one("work_mode"),
        "job_types": many("job_types"),
        "interested_in_abroad": one("interested_in_abroad"),
        "skill_ids": many("skill_ids"),
        "preferred_category_ids": many("preferred_category_ids"),
        "preferred_role_ids": many("preferred_role_ids"),
        "preferred_location_ids": many("preferred_location_ids"),
        "other_state_ids": many("other_state_ids"),
        "languages": languages,
    }
    # Back-compat text fields the recommendation menu reads (first selection).
    first_role = (form["preferred_role_ids"] or [None])[0]
    form["preferred_role"] = await LookupRepository.name_for("roles", first_role) or ""
    first_loc = (form["preferred_location_ids"] or [None])[0]
    form["location"] = await LookupRepository.name_for("districts", first_loc) or ""

    memory = get_memory(request)
    identity = await memory.save_onboarding(token, form)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)

    # Build the DB-ready registration payload (private_job_seekers +
    # job_seeker_profiles + child rows) and STAGE it in Redis (always — it's the
    # safe backup / inspection copy).
    payload: dict[str, Any] | None = None
    try:
        payload = await prepare_registration(identity=identity, form=form)
        await memory.save_registration(
            identity["conversation_id"], payload, tenant_id=identity["tenant_id"]
        )
    except Exception as exc:  # noqa: BLE001 — staging must never fail the submission
        log.warning("registration_stage_failed", error=str(exc)[:200])

    # Write to the LIVE job board only when explicitly enabled (REGISTER_IN_DB).
    # A DB failure is logged but never fails the form — the Redis copy still holds.
    if payload is not None and get_settings().register_in_db:
        try:
            res = await JobSeekerRepository.create(payload, commit=True)
            log.info("registration_db_written", seeker_id=res["seeker_id"], children=res["children"])
        except Exception as exc:  # noqa: BLE001
            log.error("registration_db_write_failed", error=str(exc)[:300])

    settings = get_settings()
    # PROACTIVELY push the "registration successful" message + the seeker hub to
    # WhatsApp, so it appears the moment they return to the chat — no typing
    # needed. They reached this form via the Job Seeker lane, so we continue as a
    # seeker (search / status / recommendations) rather than re-asking the lane.
    # Mark onboarding welcomed so the bot doesn't also send its own one-time
    # success on the next message.
    phone = re.sub(r"\D", "", identity.get("customer_id") or "")
    if phone:
        first = (identity.get("name") or "").split()[0] if identity.get("name") else ""
        who = f", {first}" if first else ""
        body = (
            f"🎉 Registration successful{who}!\n\nYou're all set. Here's what I can "
            "help you with — just tap an option below."
        )
        try:
            await wa_delivery.send_message(settings, phone, wa.buttons_message(body, _SEEKER_HUB_BUTTONS))
            await memory.mark_onboarding_welcomed(
                identity["conversation_id"], tenant_id=identity["tenant_id"]
            )
        except Exception as exc:  # noqa: BLE001 — proactive push is best-effort
            log.warning("onboard_push_failed", error=str(exc)[:200])

    number = re.sub(r"\D", "", settings.whatsapp_business_number or "")
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
  .prog{{display:flex;gap:12px;margin:0 0 22px;flex-wrap:wrap;align-items:center}}
  .prog span{{width:28px;height:28px;border-radius:50%;background:#202c33;color:#8696a0;display:flex;
              align-items:center;justify-content:center;font-size:13px;font-weight:700;border:1px solid #2a3942;
              transition:background .15s}}
  .prog span.on{{background:#00a884;color:#fff;border-color:#00a884}}
  .step{{display:none}} .step.on{{display:block}}
  .chips{{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 4px}}
  .chip{{padding:9px 14px;border:1px solid #2a3942;border-radius:20px;background:#202c33;color:#e9edef;
         font-size:14px;cursor:pointer;user-select:none}}
  .chip.on{{background:#005c4b;border-color:#00a884;color:#fff}}
  .chips.invalid,.ts.invalid .ts-box,.dob.invalid select,select.invalid,input.invalid{{border-color:#ff6b6b}}
  .nav{{display:flex;gap:10px;align-items:center;margin-top:24px}}
  .nav button{{margin-top:0}}
  .nav .back{{background:#2a3942;color:#e9edef;width:auto;flex:none;padding:12px 16px}}
  .nav .skip{{background:transparent;color:#8696a0;width:auto;flex:none;padding:12px 4px;font-weight:500}}
  .nav .next{{flex:1}}
  .dob{{display:flex;gap:8px}}
  .lang{{border:1px solid #2a3942;border-radius:10px;padding:10px 12px;margin:8px 0}}
  .lang .nm{{font-weight:600;margin-bottom:8px}}
  .lang label{{display:inline-flex;align-items:center;gap:6px;margin:0 16px 0 0;color:#cfd9de;font-size:14px}}
  .lang input{{width:auto}}
  .ro{{background:#161f25;color:#8696a0}}
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

// Chrome may autofill the State select from a saved address — force it back to
// the "Select state…" placeholder so nothing is pre-chosen until the user picks.
(function(){
  var st = document.getElementById("state_id");
  if(!st) return;
  function reset(){ if(st.value){ st.value = ""; } }
  reset();
  setTimeout(reset, 200); setTimeout(reset, 600);
})();

var STATE_NAME = {}; STATES.forEach(function(s){ STATE_NAME[s[0]] = s[1]; });
var LOC_OPTS = DISTRICTS.map(function(d){ return [d[0], d[1], STATE_NAME[d[2]] || ""]; });

tokenSelect(document.getElementById("ts_skills"), "skill_ids", SKILLS, "Type a skill...");
tokenSelect(document.getElementById("ts_locations"), "preferred_location_ids", LOC_OPTS, "Type a district...");
tokenSelect(document.getElementById("ts_categories"), "preferred_category_ids", CATEGORIES, "Type a category...");
tokenSelect(document.getElementById("ts_roles"), "preferred_role_ids", ROLES, "Type a job role...");
tokenSelect(document.getElementById("ts_other_states"), "other_state_ids", OTHER_STATES, "Type a state...");

// ---- chip groups (single / multi-select → hidden inputs) ----
function chipGroup(group){
  var name = group.getAttribute("data-name"), multi = group.getAttribute("data-multi") === "1";
  group.querySelectorAll(".chip").forEach(function(chip){
    chip.addEventListener("click", function(){
      if(multi){ chip.classList.toggle("on"); }
      else { group.querySelectorAll(".chip").forEach(function(c){ c.classList.remove("on"); }); chip.classList.add("on"); }
      group.classList.remove("invalid");
      group.querySelectorAll("input.cv").forEach(function(h){ h.remove(); });
      group.querySelectorAll(".chip.on").forEach(function(c){
        var h = document.createElement("input"); h.type = "hidden"; h.className = "cv";
        h.name = name; h.value = c.getAttribute("data-val"); group.appendChild(h);
      });
    });
  });
}
document.querySelectorAll(".chips").forEach(chipGroup);

// ---- date of birth combiner (Day / Month / Year → YYYY-MM-DD) ----
(function(){
  var d = document.getElementById("dob_d"), m = document.getElementById("dob_m"),
      y = document.getElementById("dob_y"), hid = document.getElementById("date_of_birth");
  if(!d) return;
  for(var i=1;i<=31;i++){ var o=document.createElement("option"); o.value=i; o.textContent=i; d.appendChild(o); }
  ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].forEach(function(nm,i){
    var o=document.createElement("option"); o.value=i+1; o.textContent=nm; m.appendChild(o); });
  var ny=new Date().getFullYear(); for(var yr=ny-15; yr>=ny-70; yr--){ var o=document.createElement("option"); o.value=yr; o.textContent=yr; y.appendChild(o); }
  function pad(n){ return String(n).length<2 ? "0"+n : ""+n; }
  function upd(){ hid.value=(d.value&&m.value&&y.value)?(y.value+"-"+pad(m.value)+"-"+pad(d.value)):""; if(hid.value) d.parentNode.classList.remove("invalid"); }
  [d,m,y].forEach(function(s){ s.addEventListener("change", upd); });
})();

// ---- step wizard (Next validates required, Skip / Back navigate) ----
var steps = [].slice.call(document.querySelectorAll(".step"));
var dots = [].slice.call(document.querySelectorAll(".prog span"));
var cur = 0;
function show(i){
  cur = i;
  steps.forEach(function(s,k){ s.classList.toggle("on", k===i); });
  dots.forEach(function(s,k){ s.classList.toggle("on", k<=i); });
  window.scrollTo(0,0);
  document.getElementById("nextBtn").textContent = (i===steps.length-1) ? "Finish Setup" : "Next";
  document.getElementById("backBtn").style.visibility = i===0 ? "hidden" : "visible";
}
function valid(step){
  var ok = true;
  step.querySelectorAll("[data-req]").forEach(function(el){
    var bad = false;
    if(el.classList.contains("chips")) bad = !el.querySelector(".chip.on");
    else if(el.classList.contains("ts")) bad = !el.querySelector("input[type=hidden]");
    else if(el.classList.contains("dob")) bad = !document.getElementById("date_of_birth").value;
    else if(el.tagName === "SELECT") bad = !el.value;
    else bad = !((el.value||"").trim());
    el.classList.toggle("invalid", bad); if(bad) ok = false;
  });
  return ok;
}
function advance(){ if(cur===steps.length-1) document.getElementById("profForm").submit(); else show(cur+1); }
document.getElementById("nextBtn").addEventListener("click", function(){ if(valid(steps[cur])) advance(); });
document.getElementById("skipBtn").addEventListener("click", advance);
document.getElementById("backBtn").addEventListener("click", function(){ if(cur>0) show(cur-1); });
show(0);
"""


def _chips(name: str, options, *, multi: bool = False, req: bool = False) -> str:
    """A pill chip-group → hidden inputs (single or multi select)."""
    attrs = f' data-name="{_esc(name)}"'
    if multi:
        attrs += ' data-multi="1"'
    if req:
        attrs += ' data-req="1"'
    btns = "".join(f'<span class="chip" data-val="{_esc(v)}">{_esc(l)}</span>' for v, l in options)
    return f'<div class="chips"{attrs}>{btns}</div>'


def _R() -> str:
    return '<span class="req">*</span>'


def _form_html(
    token: str, name: str, phone: str, opts: dict[str, list[dict[str, Any]]], *, error: str = ""
) -> str:
    safe_name = _esc(name)
    err = f'<div class="err">{_esc(error)}</div>' if error else ""
    o = opts  # shorthand
    dots = "".join(f"<span>{i}</span>" for i in range(1, 10))
    langs = "".join(
        f'<div class="lang"><div class="nm">{_esc(l["name"])}</div>'
        f'<label><input type="checkbox" name="lang_speak" value="{_esc(l["id"])}"> Speak</label>'
        f'<label><input type="checkbox" name="lang_write" value="{_esc(l["id"])}"> Write</label></div>'
        for l in o["languages"]
    )

    body = f"""\
<h1>Complete Your Profile</h1>
<div class="prog">{dots}</div>
{err}
<form method="post" action="/onboard/submit" id="profForm" autocomplete="off">
  <input type="hidden" name="token" value="{_esc(token)}">
  <input type="hidden" name="date_of_birth" id="date_of_birth">

  <section class="step"><h1>Personal Info</h1>
    <label>I am a {_R()}</label>
    {_chips("current_status", _I_AM_A, req=True)}
    <label>Full Name {_R()}</label>
    <input type="text" name="full_name" value="{safe_name}" placeholder="Your full name" data-req>
    <label>Mobile Number</label>
    <input type="text" class="ro" value="{_esc(phone)}" readonly>
    <label>Email</label>
    <input type="email" name="email" placeholder="you@example.com">
    <label>Resume link <span class="hint">(Google Drive, Dropbox, etc.)</span></label>
    <input type="url" name="resume" placeholder="https://…">
    <label>Gender {_R()}</label>
    {_chips("gender", _GENDER, req=True)}
    <label>Marital Status {_R()}</label>
    {_chips("marital_status", _MARITAL, req=True)}
  </section>

  <section class="step"><h1>Birth &amp; Location</h1>
    <label>Date of Birth {_R()}</label>
    <div class="dob" data-req>
      <select id="dob_d"><option value="">Day</option></select>
      <select id="dob_m"><option value="">Month</option></select>
      <select id="dob_y"><option value="">Year</option></select>
    </div>
    <label>State {_R()}</label>
    <select name="state_id" id="state_id" data-req autocomplete="off">{_options_html(o["states"], placeholder="Select state…")}</select>
    <label>District {_R()}</label>
    <select name="district_id" id="district_id" data-req autocomplete="off"><option value="">Select a state first…</option></select>
    <label>City / Area</label>
    <input type="text" name="city" placeholder="Enter city name" autocomplete="off">
  </section>

  <section class="step"><h1>Education</h1>
    <label>Education Level {_R()}</label>
    <select name="education_level_id" data-req>{_options_html(o["education_levels"], placeholder="Select…")}</select>
    <label>Course / Degree</label>
    <select name="course_id" id="course_id">{_options_html(o["courses"], placeholder="Select…")}</select>
    <label>Specialization {_R()}</label>
    <select name="specialization_id" id="specialization_id" data-req><option value="">Select a course first…</option></select>
    <label>Institution / College</label>
    <input type="text" name="institution" placeholder="e.g. Anna University">
    <label>Year of Passing</label>
    <input type="number" name="year_of_passing" min="1970" max="2035" placeholder="e.g. 2024">
  </section>

  <section class="step"><h1>Skills</h1>
    <p class="sub">Select at least 1 skill to get better job matches.</p>
    <label>Selected Skills {_R()}</label>
    <div id="ts_skills" data-req></div>
  </section>

  <section class="step"><h1>Job Preferences</h1>
    <label>Preferred Job Categories {_R()}</label>
    <div id="ts_categories" data-req></div>
    <label>Job Type</label>
    {_chips("job_types", _JOB_TYPES, multi=True)}
    <label>Work Mode Preference</label>
    {_chips("work_mode", _WORK_MODE)}
  </section>

  <section class="step"><h1>Salary &amp; Experience</h1>
    <label>Expected Monthly Salary {_R()}</label>
    {_chips("expected_salary", _SALARY, req=True)}
    <label>Current Monthly Salary (₹) <span class="hint">(if working)</span></label>
    <input type="number" name="current_salary" min="0" step="500" placeholder="e.g. 18000">
    <label>Do you have work experience?</label>
    {_chips("has_experience", _YESNO)}
    <label>Experience level <span class="hint">(if experienced)</span></label>
    <select name="experience_level_id">{_options_html(o["experience_levels"], placeholder="Select…")}</select>
  </section>

  <section class="step"><h1>Preferred Roles</h1>
    <p class="sub">Pick the roles you'd like us to match you with.</p>
    <label>Preferred Job Roles</label>
    <div id="ts_roles"></div>
  </section>

  <section class="step"><h1>Preferred Work Locations</h1>
    <label>Cities / Districts {_R()}</label>
    <div id="ts_locations" data-req></div>
    <label>Other States</label>
    <div id="ts_other_states"></div>
    <label>Interested in working abroad?</label>
    {_chips("interested_in_abroad", _YESNO)}
  </section>

  <section class="step"><h1>Language Mastery</h1>
    <p class="sub">Which languages can you speak and write? This helps us match you.</p>
    {langs}
  </section>

  <div class="nav">
    <button type="button" class="back" id="backBtn">← Back</button>
    <button type="button" class="skip" id="skipBtn">Skip for now</button>
    <button type="button" class="next" id="nextBtn">Next</button>
  </div>
</form>
<script>
const SKILLS = {_js_rows(o["skills"])};
const ROLES = {_js_rows(o["roles"])};
const CATEGORIES = {_js_rows(o["categories"])};
const STATES = {_js_rows(o["states"])};
const DISTRICTS = {_js_rows(o["districts"])};
const SPECS = {_js_rows(o["specializations"])};
const OTHER_STATES = {_js_rows(o["other_states"])};
</script>
<script>{_FORM_JS}</script>"""
    return _PAGE.format(body=body)


def _success_html(name: str, *, business_number: str = "") -> str:
    who = f", {html.escape(name)}" if name else ""
    # A "Back to chat" button that returns the candidate to WhatsApp. With the
    # business number configured it's a wa.me deep link that simply reopens the
    # chat — NO pre-filled text, because the bot already pushed the registration
    # success + menu, so the conversation just continues. Otherwise it's a
    # best-effort window.close(). This button only appears AFTER a submission, so
    # a candidate who merely opens the form and leaves is never let past the gate.
    if business_number:
        # Reliable: navigate back to WhatsApp (the app reopens the chat). A web
        # page cannot force-close a tab the user navigated to, so we redirect
        # rather than call window.close().
        close = (
            f'<a class="btn" href="https://wa.me/{business_number}">'
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
  <h1>Registration successful{who}!</h1>
  <p class="sub">Your profile is saved. Tap below to head back to WhatsApp —
  your menu is already waiting in the chat.</p>
  {close}
</div>"""
    return _PAGE.format(body=body)


def _expired_html() -> str:
    body = """\
<h1>Link expired</h1>
<p class="sub">This form link is invalid or has expired. Please go back to
WhatsApp and message us so we can send you a fresh one.</p>"""
    return _PAGE.format(body=body)
