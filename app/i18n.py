"""Lightweight multilingual support (translate-pivot).

The bot's intent routing + templated replies are English. To serve Tamil / Hindi
users we translate the INBOUND message to English (so the existing English routing
and search keep working) and translate the OUTBOUND text back to the user's
language. Script-based detection (no extra dependency); LLM-backed translation with
an in-memory cache so repeated templated strings (greetings, menus) translate once.

Phase 1 covers the conversational TEXT (``draft_response`` + the paced delivery
bubbles). Interactive button/list labels, job-card bodies, and the web forms stay
English for now (Phase 2). Everything here is best-effort — a translation error
never breaks the turn (the original English text is used).
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.llm.client import LLMClient

log = get_logger(__name__)

# Supported languages: English is the pivot; add more by extending these maps.
SUPPORTED_LANGS = {"en", "ta", "hi"}
_LANG_NAME = {"en": "English", "ta": "Tamil", "hi": "Hindi"}

# Unicode script ranges used for detection.
_TAMIL = re.compile(r"[஀-௿]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")        # Hindi (and other Devanagari)
# "has a letter in any supported script" — purely-numeric/punctuation/emoji strings
# (and bare ids) are never sent to the translator.
_HAS_LETTER = re.compile(r"[A-Za-zऀ-ॿ஀-௿]")


def detect_lang(text: str | None) -> str:
    """The script's language: ``ta`` (Tamil), ``hi`` (Devanagari/Hindi), else
    ``en``. Romanized text (Tanglish/Hinglish) is treated as ``en`` for now."""
    t = text or ""
    if _TAMIL.search(t):
        return "ta"
    if _DEVANAGARI.search(t):
        return "hi"
    return "en"


def is_supported(lang: str | None) -> bool:
    return lang in SUPPORTED_LANGS


def _translatable(s: str | None) -> bool:
    return bool(s and s.strip() and _HAS_LETTER.search(s))


# Curated, hand-verified translations for the FIXED job-category names. The LLM
# mistranslates some out of context (e.g. "Hospitality & Tourism" → "World
# Transport"), so these authoritative strings are used instead — guaranteeing
# correct, CONSISTENT labels (which also makes the reverse-match reliable). Keyed
# by the exact catalog name (case-insensitive). Extendable to other fixed UI terms.
_GLOSSARY: dict[str, dict[str, str]] = {
    "ta": {
        "accounting & financee": "கணக்கியல் & நிதி",
        "accounting & finance": "கணக்கியல் & நிதி",
        "administration": "நிர்வாகம்",
        "agriculture": "விவசாயம்",
        "banking & insurance": "வங்கி & காப்பீடு",
        "construction": "கட்டுமானம்",
        "customer service": "வாடிக்கையாளர் சேவை",
        "design & creative": "வடிவமைப்பு & படைப்பாக்கம்",
        "education & training": "கல்வி & பயிற்சி",
        "engineering": "பொறியியல்",
        "healthcare": "சுகாதாரம்",
        "hospitality & tourism": "விருந்தோம்பல் & சுற்றுலா",
        "human resources": "மனிதவளம்",
        "information technology": "தகவல் தொழில்நுட்பம்",
        "legal": "சட்டம்",
        "logistics & supply chain": "தளவாடம் & விநியோகச் சங்கிலி",
        "manufacturing": "உற்பத்தி",
        "media & communications": "ஊடகம் & தொடர்பு",
        "operations": "செயல்பாடுகள்",
        "retail": "சில்லறை விற்பனை",
        "sales & marketing": "விற்பனை & சந்தைப்படுத்தல்",
        "security services": "பாதுகாப்பு சேவைகள்",
        "technician / maintenance": "தொழில்நுட்பர் / பராமரிப்பு",
        # --- fixed menu / button labels (seeker + employer) ---
        "job search": "வேலை தேடல்",
        "application status": "விண்ணப்ப நிலை",
        "recommended jobs": "பரிந்துரைக்கப்பட்ட வேலைகள்",
        "post a job": "வேலை பதிவிடு",
        "view candidates": "விண்ணப்பத்தார்களை பார்க்க",
        "menu": "பட்டியல்", "employer menu": "முதலாளர் பட்டியல்",
        "welcome back": "மீண்டும் வருக", "hi": "வணக்கம்",
        "what would you like to do today?": "இன்று நீங்கள் என்ன செய்ய விரும்புகிறீர்கள்?",
        "what are you looking for today?": "இன்று நீங்கள் எதைத் தேடுகிறீர்கள்?",
        "employer": "முதலாளர்", "welcome to jobs7.": "Jobs7-க்கு வரவேற்கிறோம்.",
        "are you here to find a job, or to hire as an employer?":
            "நீங்கள் வேலை தேட வந்தீர்களா, அல்லது முதலாளராக ஆட்களை நியமிக்க வந்தீர்களா?",
        'tap an option below (or reply "job seeker" / "employer").':
            'கீழே ஒரு விருப்பத்தைத் தட்டவும் (அல்லது "வேலை தேடுபவர்" / "முதலாளர்" எனப் பதிலளிக்கவும்).',
        # --- seeker onboarding form handover (deterministic) ---
        "thanks": "நன்றி", "open form": "படிவத்தைத் திற",
        "one quick step to finish setting up your profile — please fill this short form "
        "(email, experience, preferred role/location):":
            "உங்கள் சுயவிவரத்தை அமைப்பதை முடிக்க ஒரு விரைவான படி — இந்தக் குறுகிய படிவத்தை "
            "நிரப்பவும் (மின்னஞ்சல், அனுபவம், விருப்பமான பணி/இடம்):",
        "once you've submitted it, message me here and we'll find you some roles.":
            "சமர்ப்பித்த பிறகு, இங்கே எனக்குச் செய்தி அனுப்புங்கள், நாங்கள் உங்களுக்கு சில பணிகளைக் கண்டுபிடிப்போம்.",
        "one quick step to finish setting up your profile — tap below to fill a short form "
        "(email, experience, preferred role/location). once you're done, message me here "
        "and we'll find you some roles.":
            "உங்கள் சுயவிவரத்தை அமைப்பதை முடிக்க ஒரு விரைவான படி — கீழே ஒரு குறுகிய படிவத்தை "
            "நிரப்பத் தட்டவும் (மின்னஞ்சல், அனுபவம், விருப்பமான பணி/இடம்). முடித்ததும், இங்கே "
            "எனக்குச் செய்தி அனுப்புங்கள், நாங்கள் உங்களுக்கு சில பணிகளைக் கண்டுபிடிப்போம்.",
        # --- My Jobs / job card (deterministic) ---
        "your posted jobs": "நீங்கள் இடுகையிட்ட வேலைகள்",
        "status:": "நிலை:", "pending": "நிலுவையில்", "live": "செயலில்",
        "approved": "அங்கீகரிக்கப்பட்டது", "closed": "மூடப்பட்டது",
        "expired": "காலாவதியானது", "draft": "வரைவு",
        "full-time": "முழு நேரம்", "part-time": "பகுதி நேரம்", "on-site": "அலுவலகம்",
        "apply:": "விண்ணப்பிக்க:", "in-app": "ஆப்-இல்", "phone": "தொலைபேசி",
        "whatsapp": "வாட்ஸ்அப்", "vacancy": "காலியிடம்", "vacancies": "காலியிடங்கள்",
        "days": "நாட்கள்", "yrs": "ஆண்டுகள்",
        "you haven't posted any jobs yet. tap below to post your first one.":
            "நீங்கள் இன்னும் எந்த வேலையையும் இடவில்லை. உங்கள் முதல் வேலையை இட கீழே தட்டவும்.",
        # --- category role-list (deterministic) ---
        "roles. tap one to see the openings": "பணிகள். ஒன்றைத் தேர்ந்து விவரங்களைப் பார்க்க",
        "view roles": "பணிகளைப் பார்க்க", "roles": "பணிகள்",
        "more roles ▸": "மேலும் பணிகள் ▸", "showing": "காட்டுகிறது", "more": "மேலும்",
        "our team will review it and contact you shortly.":
            "எங்கள் குழு அதை ஆய்வு செய்து, விரைவில் உங்களைத் தொடர்பு கொள்ளும்.",
        "🗓 valid for 30 days": "🗓 30 நாட்களுக்கு பயன்படுத்தக்கூடியது",
        "✅ *job submitted!*": "✅ *வேலை சமர்ப்பிக்கப்பட்டது!*", "what next?": "அடுத்தது என்ன?",
        "covered by your plan · balance": "உங்கள் திட்டத்தில் சேர்க்கப்பட்டது · இருப்பு",
        "job credits": "வேலை கிரெடிட்கள்", "credit": "கிரெடிட்", "credits": "கிரெடிட்கள்",
        "used · balance": "பயன்படுத்தப்பட்டது · இருப்பு",
        "my jobs": "எனது வேலைகள்",
        "🪪 credits & wallet": "🪪 கிரெடிட்கள் & வாலெட்",
        "💳 buy credits": "💳 கிரெடிட்கள் வாங்கு",
        "create a new job listing": "புதிய வேலை பட்டியலை உருவாக்கு",
        "browse matching candidates": "பொருந்தும் வேட்பாளர்களைப் பார்",
        "your posted jobs": "நீங்கள் இடுகையிட்ட வேலைகள்",
        "manage your credits": "உங்கள் கிரெடிட்களை நிர்வகி",
        "view pricing and bundles": "விலை மற்றும் தொகுப்புகளைப் பார்",
        # --- people nouns: reverse-map to clean English so "applicant"/"candidate"
        # routes deterministically (employer → View Candidates, seeker → switch hint) ---
        "applicant": "விண்ணப்பதாரர்",
        "applicants": "விண்ணப்பதாரர்கள்",
        "candidate": "வேட்பாளர்",
        "candidates": "வேட்பாளர்கள்",
        "job seeker": "வேலை தேடுபவர்",
        "job seekers": "வேலை தேடுபவர்கள்",
        # --- candidate-card scaffolding (rendered deterministically, not LLM) ---
        "*candidates available* 👥": "*கிடைக்கும் வேட்பாளர்கள்* 👥",
        "*candidates — full details unlocked* 🔓": "*வேட்பாளர்கள் — முழு விவரங்கள் திறக்கப்பட்டன* 🔓",
        "candidates matching": "பொருந்தும் வேட்பாளர்கள்",
        "🔓 unlock details": "🔓 விவரம் திற",
        "experience not specified": "அனுபவம் குறிப்பிடப்படவில்லை",
        "freshers": "புதியவர்கள்",
        "fresher": "புதியவர்",
        "years": "ஆண்டுகள்",
        "year": "ஆண்டு",
        "months": "மாதங்கள்",
        "month": "மாதம்",
        "🔒 contact details and resume are locked. unlock the full profiles — "
        "phone, email & resume — on the jobs7 employer app.":
            "🔒 தொடர்பு விவரங்கள் மற்றும் ரெஸ்யூம் பூட்டப்பட்டுள்ளன. முழு விவரங்களை — "
            "தொலைபேசி, மின்னஞ்சல் & ரெஸ்யூம் — Jobs7 Employer ஆப்-இல் திறக்கவும்.",
        # --- web-form labels / options / buttons (deterministic, no LLM) ---
        "register your company": "உங்கள் நிறுவனத்தைப் பதிவு செய்யுங்கள்",
        "complete your profile": "உங்கள் சுயவிவரத்தை முடிக்கவும்",
        "personal info": "தனிப்பட்ட தகவல்", "birth & location": "பிறப்பு & இடம்",
        "education": "கல்வி", "salary & experience": "சம்பளம் & அனுபவம்",
        "job preferences": "வேலை விருப்பங்கள்", "language mastery": "மொழித் திறன்",
        "job details": "வேலை விவரங்கள்", "experience & salary": "அனுபவம் & சம்பளம்",
        "job location": "வேலை இடம்",
        "candidate location preference": "வேட்பாளர் இட விருப்பம்",
        "apply methods": "விண்ணப்பிக்கும் முறைகள்", "skills": "திறன்கள்",
        "selected skills": "தேர்ந்தெடுக்கப்பட்ட திறன்கள்",
        "upload your resume": "உங்கள் ரெஸ்யூமேயைப் பதிவேற்றவும்",
        "preferred job roles": "விருப்பமான வேலைப் பணிகள்",
        "preferred roles": "விருப்பமான பணிகள்",
        "preferred work locations": "விருப்பமான பணியிடங்கள்",
        "company name": "நிறுவனப் பெயர்", "address": "முகவரி", "state": "மாநிலம்",
        "district": "மாவட்டம்", "districts": "மாவட்டங்கள்", "city": "நகரம்",
        "city / area": "நகரம் / பகுதி", "pincode": "பின்கோடு", "description": "விளக்கம்",
        "job title": "வேலைப் பெயர்", "job category": "வேலைப் பிரிவு",
        "job type": "வேலை வகை", "experience required": "தேவையான அனுபவம்",
        "years of experience": "அனுபவ ஆண்டுகள்", "min years": "குறைந்தபட்ச ஆண்டுகள்",
        "max years": "அதிகபட்ச ஆண்டுகள்", "min (₹)": "குறைந்தபட்சம் (₹)",
        "max (₹)": "அதிகபட்சம் (₹)", "min": "குறைந்தபட்சம்", "max": "அதிகபட்சம்",
        "salary range": "சம்பள வரம்பு",
        "number of vacancies": "காலியிடங்களின் எண்ணிக்கை",
        "monthly stipend (₹)": "மாதாந்திர உதவித்தொகை (₹)",
        "training fee (₹)": "பயிற்சிக் கட்டணம் (₹)", "duration (months)": "காலம் (மாதங்கள்)",
        "intern payment type": "பயிற்சியாளர் கட்டண வகை",
        "contact phone number": "தொடர்பு தொலைபேசி எண்", "whatsapp number": "வாட்ஸ்அப் எண்",
        "full name": "முழு பெயர்", "email": "மின்னஞ்சல்", "mobile number": "கைபேசி எண்",
        "gender": "பாலினம்", "date of birth": "பிறந்த தேதி", "marital status": "திருமண நிலை",
        "i am a": "நான் ஒரு", "education level": "கல்வி நிலை",
        "course / degree": "படிப்பு / பட்டம்", "institution / college": "நிறுவனம் / கல்லூரி",
        "specialization": "சிறப்புத் துறை", "year of passing": "தேர்ச்சி ஆண்டு",
        "expected monthly salary": "எதிர்பார்க்கும் மாத சம்பளம்",
        "experience level": "அனுபவ நிலை",
        "preferred job categories": "விருப்பமான வேலைப் பிரிவுகள்",
        "interested in working abroad?": "வெளிநாட்டில் பணியாற்ற விருப்பமா?",
        "work mode preference": "பணி முறை விருப்பம்", "office address": "அலுவலக முகவரி",
        "company address": "நிறுவன முகவரி",
        # options
        "any": "ஏதேனும்", "fresher only": "புதியவர் மட்டும்", "intern": "பயிற்சியாளர்",
        "experienced": "அனுபவம் உள்ளவர்", "monthly": "மாதாந்திரம்", "annual": "ஆண்டுதோறும்",
        "specific location": "குறிப்பிட்ட இடம்", "remote": "தொலைதூரம்",
        "full time": "முழு நேரம்", "part time": "பகுதி நேரம்", "on-site / office": "அலுவலகம்",
        "hybrid": "கலப்பு", "both": "இரண்டும்", "male": "ஆண்", "female": "பெண்",
        "unmarried": "திருமணமாகாதவர்", "married": "திருமணமானவர்", "custom": "தனிப்பயன்",
        "anywhere": "எங்கும்", "phone call": "தொலைபேசி அழைப்பு", "in-app apply": "ஆப்-இல் விண்ணப்பி",
        "no need": "தேவையில்லை", "intermediate": "நடுத்தரம்", "good english": "நல்ல ஆங்கிலம்",
        "degree": "பட்டம்", "company pays (stipend)": "நிறுவனம் செலுத்தும் (உதவித்தொகை)",
        "intern pays (training fee)": "பயிற்சியாளர் செலுத்தும் (பயிற்சிக் கட்டணம்)",
        # buttons
        "create profile": "சுயவிவரத்தை உருவாக்கு", "post job": "வேலையை இடு",
        "activate now": "இப்போது செயல்படுத்து", "next": "அடுத்து", "next →": "அடுத்து →",
        "← back": "← திரும்பு", "back": "திரும்பு",
        "submit resume": "ரெஸ்யூமேயைச் சமர்ப்பி", "skip for now": "தற்போதைக்கு தவிர்",
        "close": "மூடு", "cancel": "ரத்து செய்",
        # quick-select chips + misc
        "🏢 company district": "🏢 நிறுவன மாவட்டம்", "📍 nearby": "📍 அருகில்",
        "▦ all districts": "▦ அனைத்து மாவட்டங்கள்", "🏙 top cities": "🏙 முக்கிய நகரங்கள்",
        "⚙ custom": "⚙ தனிப்பயன்", "quick select": "விரைவு தேர்வு",
        "🕒 job validity": "🕒 வேலை செல்லுபடி காலம்", "🧮 credits required": "🧮 தேவையான கிரெடிட்கள்",
        # placeholders
        "your full name": "உங்கள் முழு பெயர்",
        "role responsibilities, requirements…": "பணிப் பொறுப்புகள், தேவைகள்…",
        "enter city name": "நகரத்தின் பெயரை உள்ளிடவும்", "street, area, landmark": "தெரு, பகுதி, அடையாளம்",
        # seeker-form options + dropdown placeholders
        "day": "நாள்", "resume": "ரெஸ்யூமே",
        "select state…": "மாநிலத்தைத் தேர்ந்தெடுக்கவும்…",
        "select a state first…": "முதலில் மாநிலத்தைத் தேர்ந்தெடுக்கவும்…",
        "select a course first…": "முதலில் படிப்பைத் தேர்ந்தெடுக்கவும்…",
        "type a category...": "ஒரு பிரிவை உள்ளிடவும்...",
        "contract": "ஒப்பந்தம்", "internship": "பயிற்சி", "freelance": "சுயதொழில்",
        "temporary": "தற்காலிக", "work from home": "வீட்டிலிருந்து வேலை", "walk-in": "நேரடி வருகை",
        "student": "மாணவர்", "fresher - first job": "புதியவர் - முதல் வேலை", "other": "மற்றவை",
        "single": "திருமணமாகாதவர்", "basic": "அடிப்படை", "fluent": "சரளமான",
        # helper subtitles
        "a few details about your business so candidates know who's hiring.":
            "வேட்பாளர்கள் யார் பணியமர்த்துகிறார்கள் என அறிய உங்கள் வணிகம் பற்றிய சில விவரங்கள்.",
        "choose how candidates can reach you for this job.":
            "இந்த வேலைக்கு வேட்பாளர்கள் உங்களை எவ்வாறு தொடர்பு கொள்ளலாம் எனத் தேர்வுசெய்யவும்.",
        "select where candidates should be from.":
            "வேட்பாளர்கள் எங்கிருந்து இருக்க வேண்டும் எனத் தேர்ந்தெடுக்கவும்.",
        "pick the roles you'd like us to match you with.":
            "நாங்கள் உங்களைப் பொருத்த விரும்பும் பணிகளைத் தேர்வுசெய்யவும்.",
        "select at least 1 skill to get better job matches.":
            "சிறந்த வேலைப் பொருத்தங்களுக்கு குறைந்தது 1 திறனைத் தேர்ந்தெடுக்கவும்.",
        "which languages can you speak and write? this helps us match you.":
            "நீங்கள் எந்த மொழிகளைப் பேசவும் எழுதவும் முடியும்? இது உங்களைப் பொருத்த உதவுகிறது.",
        "choose your resume (pdf or doc) to finish applying.":
            "விண்ணப்பத்தை முடிக்க உங்கள் ரெஸ்யூமேயை (PDF அல்லது DOC) தேர்வுசெய்யவும்.",
    },
    "hi": {
        "accounting & financee": "लेखांकन और वित्त",
        "accounting & finance": "लेखांकन और वित्त",
        "administration": "प्रशासन",
        "agriculture": "कृषि",
        "banking & insurance": "बैंकिंग और बीमा",
        "construction": "निर्माण",
        "customer service": "ग्राहक सेवा",
        "design & creative": "डिज़ाइन और रचनात्मक",
        "education & training": "शिक्षा और प्रशिक्षण",
        "engineering": "इंजीनियरिंग",
        "healthcare": "स्वास्थ्य सेवा",
        "hospitality & tourism": "आतिथ्य और पर्यटन",
        "human resources": "मानव संसाधन",
        "information technology": "सूचना प्रौद्योगिकी",
        "legal": "कानूनी",
        "logistics & supply chain": "रसद और आपूर्ति श्रृंखला",
        "manufacturing": "विनिर्माण",
        "media & communications": "मीडिया और संचार",
        "operations": "संचालन",
        "retail": "खुदरा",
        "sales & marketing": "बिक्री और विपणन",
        "security services": "सुरक्षा सेवाएं",
        "technician / maintenance": "तकनीशियन / रखरखाव",
        # --- fixed menu / button labels (seeker + employer) ---
        "job search": "नौकरी खोज",
        "application status": "आवेदन स्थिति",
        "recommended jobs": "अनुशंसित नौकरियां",
        "post a job": "नौकरी पोस्ट करें",
        "view candidates": "उम्मीदवार देखें",
        "menu": "मेन्यू", "employer menu": "नियोक्ता मेन्यू",
        "welcome back": "वापसी पर स्वागत है", "hi": "नमस्ते",
        "what would you like to do today?": "आज आप क्या करना चाहेंगे?",
        "what are you looking for today?": "आज आप क्या ढूंढ रहे हैं?",
        "employer": "नियोक्ता", "welcome to jobs7.": "Jobs7 में आपका स्वागत है।",
        "are you here to find a job, or to hire as an employer?":
            "क्या आप नौकरी ढूंढने आए हैं, या नियोक्ता के रूप में भर्ती करने?",
        'tap an option below (or reply "job seeker" / "employer").':
            'नीचे एक विकल्प चुनें (या "नौकरी चाहने वाला" / "नियोक्ता" उत्तर दें)।',
        # --- seeker onboarding form handover (deterministic) ---
        "thanks": "धन्यवाद", "open form": "फ़ॉर्म खोलें",
        "one quick step to finish setting up your profile — please fill this short form "
        "(email, experience, preferred role/location):":
            "आपकी प्रोफ़ाइल सेट करना पूरा करने के लिए एक त्वरित कदम — कृपया यह छोटा फ़ॉर्म भरें "
            "(ईमेल, अनुभव, पसंदीदा भूमिका/स्थान):",
        "once you've submitted it, message me here and we'll find you some roles.":
            "इसे जमा करने के बाद, मुझे यहाँ संदेश करें और हम आपके लिए कुछ भूमिकाएँ ढूंढेंगे।",
        "one quick step to finish setting up your profile — tap below to fill a short form "
        "(email, experience, preferred role/location). once you're done, message me here "
        "and we'll find you some roles.":
            "आपकी प्रोफ़ाइल सेट करना पूरा करने के लिए एक त्वरित कदम — नीचे एक छोटा फ़ॉर्म भरने के "
            "लिए टैप करें (ईमेल, अनुभव, पसंदीदा भूमिका/स्थान)। हो जाने पर, मुझे यहाँ संदेश करें "
            "और हम आपके लिए कुछ भूमिकाएँ ढूंढेंगे।",
        # --- My Jobs / job card (deterministic) ---
        "your posted jobs": "आपकी पोस्ट की गई नौकरियां",
        "status:": "स्थिति:", "pending": "लंबित", "live": "लाइव",
        "approved": "स्वीकृत", "closed": "बंद", "expired": "समाप्त", "draft": "ड्राफ्ट",
        "full-time": "पूर्णकालिक", "part-time": "अंशकालिक", "on-site": "कार्यालय",
        "apply:": "आवेदन:", "in-app": "ऐप में", "phone": "फ़ोन",
        "whatsapp": "व्हाट्सएप", "vacancy": "रिक्ति", "vacancies": "रिक्तियां",
        "days": "दिन", "yrs": "वर्ष",
        "you haven't posted any jobs yet. tap below to post your first one.":
            "आपने अभी तक कोई नौकरी पोस्ट नहीं की है। अपनी पहली नौकरी पोस्ट करने के लिए नीचे टैप करें।",
        # --- category role-list (deterministic) ---
        "roles. tap one to see the openings": "भूमिकाएं। विवरण देखने के लिए एक चुनें",
        "view roles": "भूमिकाएं देखें", "roles": "भूमिकाएं",
        "more roles ▸": "और भूमिकाएं ▸", "showing": "दिखा रहे हैं", "more": "और",
        "our team will review it and contact you shortly.":
            "हमारी टीम इसकी समीक्षा करेगी और जल्द ही आपसे संपर्क करेगी।",
        "🗓 valid for 30 days": "🗓 30 दिनों के लिए मान्य",
        "✅ *job submitted!*": "✅ *नौकरी सबमिट हो गई!*", "what next?": "आगे क्या?",
        "covered by your plan · balance": "आपकी योजना में शामिल · शेष",
        "job credits": "नौकरी क्रेडिट", "credit": "क्रेडिट", "credits": "क्रेडिट",
        "used · balance": "उपयोग किया गया · शेष",
        "my jobs": "मेरी नौकरियां",
        "🪪 credits & wallet": "🪪 क्रेडिट और वॉलेट",
        "💳 buy credits": "💳 क्रेडिट खरीदें",
        "create a new job listing": "नई नौकरी सूची बनाएं",
        "browse matching candidates": "मिलते-जुलते उम्मीदवार देखें",
        "your posted jobs": "आपकी पोस्ट की गई नौकरियां",
        "manage your credits": "अपने क्रेडिट प्रबंधित करें",
        "view pricing and bundles": "मूल्य और बंडल देखें",
        # --- people nouns (reverse-map to clean English; see Tamil block) ---
        "applicant": "आवेदक",
        "applicants": "आवेदक",
        "candidate": "उम्मीदवार",
        "candidates": "उम्मीदवार",
        "job seeker": "नौकरी चाहने वाला",
        "job seekers": "नौकरी चाहने वाले",
        # --- candidate-card scaffolding (rendered deterministically, not LLM) ---
        "*candidates available* 👥": "*उपलब्ध उम्मीदवार* 👥",
        "*candidates — full details unlocked* 🔓": "*उम्मीदवार — पूरा विवरण अनलॉक* 🔓",
        "candidates matching": "मिलते-जुलते उम्मीदवार",
        "🔓 unlock details": "🔓 अनलॉक करें",
        "experience not specified": "अनुभव निर्दिष्ट नहीं",
        "freshers": "फ्रेशर्स",
        "fresher": "फ्रेशर",
        "years": "साल",
        "year": "साल",
        "months": "महीने",
        "month": "महीना",
        "🔒 contact details and resume are locked. unlock the full profiles — "
        "phone, email & resume — on the jobs7 employer app.":
            "🔒 संपर्क विवरण और रिज्यूमे लॉक हैं। पूरी प्रोफाइल — फोन, ईमेल और रिज्यूमे — "
            "Jobs7 Employer ऐप पर अनलॉक करें।",
        # --- web-form labels / options / buttons (deterministic, no LLM) ---
        "register your company": "अपनी कंपनी पंजीकृत करें",
        "complete your profile": "अपनी प्रोफ़ाइल पूरी करें",
        "personal info": "व्यक्तिगत जानकारी", "birth & location": "जन्म और स्थान",
        "education": "शिक्षा", "salary & experience": "वेतन और अनुभव",
        "job preferences": "नौकरी प्राथमिकताएं", "language mastery": "भाषा दक्षता",
        "job details": "नौकरी विवरण", "experience & salary": "अनुभव और वेतन",
        "job location": "नौकरी स्थान",
        "candidate location preference": "उम्मीदवार स्थान प्राथमिकता",
        "apply methods": "आवेदन के तरीके", "skills": "कौशल",
        "selected skills": "चयनित कौशल", "upload your resume": "अपना रिज्यूमे अपलोड करें",
        "preferred job roles": "पसंदीदा नौकरी भूमिकाएं", "preferred roles": "पसंदीदा भूमिकाएं",
        "preferred work locations": "पसंदीदा कार्य स्थान",
        "company name": "कंपनी का नाम", "address": "पता", "state": "राज्य",
        "district": "जिला", "districts": "जिले", "city": "शहर",
        "city / area": "शहर / क्षेत्र", "pincode": "पिनकोड", "description": "विवरण",
        "job title": "नौकरी का शीर्षक", "job category": "नौकरी श्रेणी",
        "job type": "नौकरी प्रकार", "experience required": "आवश्यक अनुभव",
        "years of experience": "अनुभव के वर्ष", "min years": "न्यूनतम वर्ष",
        "max years": "अधिकतम वर्ष", "min (₹)": "न्यूनतम (₹)", "max (₹)": "अधिकतम (₹)",
        "min": "न्यूनतम", "max": "अधिकतम", "salary range": "वेतन सीमा",
        "number of vacancies": "रिक्तियों की संख्या",
        "monthly stipend (₹)": "मासिक वजीफा (₹)", "training fee (₹)": "प्रशिक्षण शुल्क (₹)",
        "duration (months)": "अवधि (महीने)", "intern payment type": "इंटर्न भुगतान प्रकार",
        "contact phone number": "संपर्क फ़ोन नंबर", "whatsapp number": "व्हाट्सएप नंबर",
        "full name": "पूरा नाम", "email": "ईमेल", "mobile number": "मोबाइल नंबर",
        "gender": "लिंग", "date of birth": "जन्म तिथि", "marital status": "वैवाहिक स्थिति",
        "i am a": "मैं हूँ", "education level": "शिक्षा स्तर",
        "course / degree": "कोर्स / डिग्री", "institution / college": "संस्थान / कॉलेज",
        "specialization": "विशेषज्ञता", "year of passing": "उत्तीर्ण वर्ष",
        "expected monthly salary": "अपेक्षित मासिक वेतन", "experience level": "अनुभव स्तर",
        "preferred job categories": "पसंदीदा नौकरी श्रेणियां",
        "interested in working abroad?": "विदेश में काम करने में रुचि?",
        "work mode preference": "कार्य मोड प्राथमिकता", "office address": "कार्यालय का पता",
        "company address": "कंपनी का पता",
        # options
        "any": "कोई भी", "fresher only": "केवल फ्रेशर", "intern": "इंटर्न",
        "experienced": "अनुभवी", "monthly": "मासिक", "annual": "वार्षिक",
        "specific location": "विशिष्ट स्थान", "remote": "रिमोट", "full time": "पूर्णकालिक",
        "part time": "अंशकालिक", "on-site / office": "कार्यालय", "hybrid": "हाइब्रिड",
        "both": "दोनों", "male": "पुरुष", "female": "महिला", "unmarried": "अविवाहित",
        "married": "विवाहित", "custom": "कस्टम", "anywhere": "कहीं भी",
        "phone call": "फ़ोन कॉल", "in-app apply": "ऐप में आवेदन करें", "no need": "ज़रूरत नहीं",
        "intermediate": "मध्यम", "good english": "अच्छी अंग्रेज़ी", "degree": "डिग्री",
        "company pays (stipend)": "कंपनी भुगतान करती है (वजीफा)",
        "intern pays (training fee)": "इंटर्न भुगतान करता है (प्रशिक्षण शुल्क)",
        # buttons
        "create profile": "प्रोफ़ाइल बनाएं", "post job": "नौकरी पोस्ट करें",
        "activate now": "अभी सक्रिय करें", "next": "अगला", "next →": "अगला →",
        "← back": "← वापस", "back": "वापस", "submit resume": "रिज्यूमे जमा करें",
        "skip for now": "अभी के लिए छोड़ें", "close": "बंद करें", "cancel": "रद्द करें",
        # quick-select chips + misc
        "🏢 company district": "🏢 कंपनी जिला", "📍 nearby": "📍 आस-पास",
        "▦ all districts": "▦ सभी जिले", "🏙 top cities": "🏙 शीर्ष शहर",
        "⚙ custom": "⚙ कस्टम", "quick select": "त्वरित चयन",
        "🕒 job validity": "🕒 नौकरी वैधता", "🧮 credits required": "🧮 आवश्यक क्रेडिट",
        # placeholders
        "your full name": "आपका पूरा नाम",
        "role responsibilities, requirements…": "भूमिका ज़िम्मेदारियां, आवश्यकताएं…",
        "enter city name": "शहर का नाम दर्ज करें", "street, area, landmark": "सड़क, क्षेत्र, लैंडमार्क",
        # seeker-form options + dropdown placeholders
        "day": "दिन", "resume": "रिज्यूमे",
        "select state…": "राज्य चुनें…", "select a state first…": "पहले राज्य चुनें…",
        "select a course first…": "पहले कोर्स चुनें…", "type a category...": "एक श्रेणी टाइप करें...",
        "contract": "अनुबंध", "internship": "इंटर्नशिप", "freelance": "फ्रीलांस",
        "temporary": "अस्थायी", "work from home": "घर से काम", "walk-in": "वॉक-इन",
        "student": "छात्र", "fresher - first job": "फ्रेशर - पहली नौकरी", "other": "अन्य",
        "single": "अविवाहित", "basic": "बुनियादी", "fluent": "धाराप्रवाह",
        # helper subtitles
        "a few details about your business so candidates know who's hiring.":
            "कुछ विवरण आपके व्यवसाय के बारे में ताकि उम्मीदवार जानें कि कौन भर्ती कर रहा है।",
        "choose how candidates can reach you for this job.":
            "चुनें कि उम्मीदवार इस नौकरी के लिए आपसे कैसे संपर्क कर सकते हैं।",
        "select where candidates should be from.":
            "चुनें कि उम्मीदवार कहाँ से होने चाहिए।",
        "pick the roles you'd like us to match you with.":
            "उन भूमिकाओं को चुनें जिनसे आप मिलान चाहते हैं।",
        "select at least 1 skill to get better job matches.":
            "बेहतर नौकरी मिलान के लिए कम से कम 1 कौशल चुनें।",
        "which languages can you speak and write? this helps us match you.":
            "आप कौन सी भाषाएँ बोल और लिख सकते हैं? यह आपको मिलान करने में मदद करता है।",
        "choose your resume (pdf or doc) to finish applying.":
            "आवेदन पूरा करने के लिए अपना रिज्यूमे (PDF या DOC) चुनें।",
    },
}


def _glossary(text: str, lang: str) -> str | None:
    """An authoritative translation for a fixed UI/category term, or None."""
    return _GLOSSARY.get(lang, {}).get((text or "").strip().lower())


def t(text: str, lang: str) -> str:
    """Synchronous, deterministic translation of a FIXED string via the glossary
    (falling back to the warmed cache, then the original). No LLM — use this to
    render fixed labels reliably in-place (e.g. candidate-card scaffolding) instead
    of depending on a per-turn wholesale LLM translation that can flake to English."""
    if lang == "en" or not text:
        return text
    g = _glossary(text, lang)
    if g is not None:
        return g
    cached = _cache_get((text, lang))
    return cached if cached is not None else text


# --- startup cache warming ---------------------------------------------------
# Fixed user-facing bodies (lane hints, guidance nudges, …) are LLM-translated, so
# a transient LLM hiccup could leave one English for a non-English user. Modules
# register their fixed strings here; ``warm_cache`` pre-translates them into every
# supported language at startup so they're cache-served thereafter — no per-turn
# LLM dependency. Best-effort: if warming fails, the per-turn path still runs.
_WARM_STRINGS: set[str] = set()


def register_warm_strings(strings: "list[str] | tuple[str, ...]") -> None:
    """Register fixed UI strings to pre-translate at startup. Safe to call at
    import time from any module (no import cycle — this module imports nothing app)."""
    for s in strings:
        if isinstance(s, str) and _translatable(s):
            _WARM_STRINGS.add(s)


async def warm_cache(llm: "LLMClient", *, chunk: int = 12, pace_s: float = 2.0) -> int:
    """Pre-translate every registered fixed string into each supported language so
    later turns serve them from cache. Translates in SMALL chunks with a pause
    between them so it stays under the LLM's tokens-per-minute limit (a big burst
    trips a 429 on rate-limited tiers). Runs in the background, so the pacing never
    delays anything user-facing. Best-effort — never raises; un-warmed strings just
    translate lazily later (or stay English) and glossary terms are unaffected."""
    if not _WARM_STRINGS:
        return 0
    strings = list(_WARM_STRINGS)
    warmed = 0
    for lang in SUPPORTED_LANGS:
        if lang == "en":
            continue
        for i in range(0, len(strings), chunk):
            batch = strings[i:i + chunk]
            try:
                before = sum(1 for s in batch if _cache_get((s, lang)) is not None)
                await translate_many(llm, batch, to_lang=lang)
                warmed += sum(1 for s in batch if _cache_get((s, lang)) is not None) - before
            except Exception as exc:  # noqa: BLE001 — warming must never break startup
                log.warning("warm_cache_chunk_failed", lang=lang, error=str(exc)[:160])
            await asyncio.sleep(pace_s)              # pace to respect the TPM limit
    log.info("i18n_cache_warmed", strings=len(strings), cached=warmed)
    return warmed


# --- web-form localization ---------------------------------------------------
# The register / post-job / seeker-onboard pages are server-rendered HTML. Rather
# than thread a language through every label, we inject a tiny client-side pass
# that swaps known English labels for their translation (built from t() — glossary
# + warmed cache). Only strings in THIS catalog are touched, so user data (names,
# skills, place names) is never altered. Registered for warming so the cache is hot.
_FORM_STRINGS: tuple[str, ...] = (
    # headers / sections
    "Register your company", "Complete Your Profile", "Personal Info",
    "Birth & Location", "Education", "Salary & Experience", "Job Preferences",
    "Language Mastery", "Job Details", "Experience & Salary", "Job Location",
    "Candidate Location Preference", "Apply Methods", "Skills", "Upload your resume",
    "Selected Skills", "Preferred Job Roles", "Preferred Roles",
    "Preferred Work Locations",
    # field labels
    "Company name", "Address", "State", "District", "Districts", "City",
    "City / area", "City / Area", "Pincode", "Description", "Job Title",
    "Job Category", "Job Type", "Experience Required", "Years of Experience",
    "Min years", "Max years", "Min (₹)", "Max (₹)", "Min", "Max", "Salary Range",
    "Number of Vacancies", "Monthly Stipend (₹)", "Training Fee (₹)",
    "Duration (months)", "Intern Payment Type", "Contact Phone Number",
    "WhatsApp Number", "Full Name", "Email", "Mobile Number", "Gender",
    "Date of Birth", "Marital Status", "I am a", "Education Level",
    "Course / Degree", "Institution / College", "Specialization", "Year of Passing",
    "Expected Monthly Salary", "Experience level", "Preferred Job Categories",
    "Interested in working abroad?", "Work Mode Preference", "Office address",
    "Company address",
    # buttons
    "Create profile", "Post Job", "Activate Now", "Next", "Submit resume",
    "Skip for now", "Close", "Cancel",
    # placeholders
    "e.g. Acme Technologies", "Your full name", "Office address",
    "Role responsibilities, requirements…", "e.g. Software Developer",
    "Enter city name", "Street, area, landmark",
    # quick-select chips + misc
    "🏢 Company District", "📍 Nearby", "▦ All Districts", "🏙 Top Cities",
    "⚙ Custom", "Quick Select", "🕒 Job Validity", "🧮 Credits Required",
    # helper subtitles
    "A few details about your business so candidates know who's hiring.",
    "Choose how candidates can reach you for this job.",
    "Select where candidates should be from.",
    "Pick the roles you'd like us to match you with.",
    "Select at least 1 skill to get better job matches.",
    "Which languages can you speak and write? This helps us match you.",
    "Choose your resume (PDF or DOC) to finish applying.",
)
register_warm_strings(_FORM_STRINGS)

# Client-side translator: walks text nodes + placeholders + option/button labels and
# swaps any exact (trimmed) catalog match. Tolerates a trailing required-marker "*".
_FORM_I18N_JS = r"""
(function(){
  var M=window.__FORMI18N__||{}; if(!Object.keys(M).length) return;
  function look(s){ if(s==null) return null; var k=String(s).trim().toLowerCase(); if(M[k]!=null) return M[k];
    var k2=k.replace(/\s*\*\s*$/,''); return M[k2]!=null?M[k2]:null; }
  function tr(root){
    if(root.nodeType===3){ var t=(root.nodeValue||'').trim(); if(t){ var r=look(t); if(r!=null) root.nodeValue=root.nodeValue.replace(t,r); } return; }
    if(root.nodeType!==1) return;
    try{ var w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT,null),n,a=[];
      while(n=w.nextNode()) a.push(n);
      a.forEach(function(node){ var raw=node.nodeValue,t=raw.trim(); if(!t) return; var r=look(t); if(r!=null) node.nodeValue=raw.replace(t,r); });
    }catch(e){}
    root.querySelectorAll('[placeholder]').forEach(function(el){ var r=look(el.getAttribute('placeholder')); if(r!=null) el.setAttribute('placeholder',r); });
    root.querySelectorAll('option').forEach(function(el){ if(!el.children.length){ var r=look(el.textContent); if(r!=null) el.textContent=r; } });
    root.querySelectorAll('button,input[type=submit],input[type=button]').forEach(function(el){ if(el.value){ var r=look(el.value); if(r!=null) el.value=r; } });
  }
  tr(document.body);
  // Re-translate nodes added later by the form's JS (chips, cascaded dropdowns) — the
  // submit VALUES live in hidden inputs, so swapping the visible text is safe.
  try{ new MutationObserver(function(muts){ muts.forEach(function(mu){
        [].forEach.call(mu.addedNodes, function(nd){ tr(nd); }); }); })
      .observe(document.body, {childList:true, subtree:true}); }catch(e){}
})();
"""


def inject_form_i18n(html: str, lang: str) -> str:
    """Inject the client-side form translator before ``</body>``. The map is the full
    GLOSSARY for the language (lowercased keys; the JS lowercases its lookups) — so
    every hand-translated label is applied DETERMINISTICALLY (no LLM, no rate limit).
    No-op for English / unsupported lang. Only exact (case-insensitive) catalog matches
    are swapped, so user data (names, skills, free text) is never touched."""
    if lang == "en" or lang not in SUPPORTED_LANGS or "</body>" not in html:
        return html
    mapping = _GLOSSARY.get(lang)
    if not mapping:
        return html
    script = ("<script>window.__FORMI18N__=" + json.dumps(mapping, ensure_ascii=False)
              + ";" + _FORM_I18N_JS + "</script>")
    return html.replace("</body>", script + "</body>", 1)


# Module-level LRU cache: (text, to_lang) -> translation. Templated replies repeat
# across users, so this avoids re-translating the same string every turn.
_CACHE: "OrderedDict[tuple[str, str], str]" = OrderedDict()
_CACHE_MAX = 2000


def _cache_get(key: tuple[str, str]) -> str | None:
    v = _CACHE.get(key)
    if v is not None:
        _CACHE.move_to_end(key)
    return v


def _cache_put(key: tuple[str, str], val: str) -> None:
    _CACHE[key] = val
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


async def to_english(llm: "LLMClient", text: str, *, source_lang: str) -> str:
    """Translate a user's message to English (for the English routing/search).
    No-op for English / non-wordy input. Best-effort — returns the original on
    error so a translation hiccup never blocks the turn."""
    if source_lang == "en" or not _translatable(text):
        return text
    # A typed LOCALIZED label (menu item / category) reverse-maps to its canonical
    # English deterministically — so typing "வேலை இடுகையிடு" routes like tapping
    # "Post a Job", instead of the LLM guessing a phrase that misses the routing.
    glossed = from_glossary(text, source_lang)
    if glossed:
        return glossed
    try:
        content, _ = await llm.chat(
            purpose="translate_in",
            messages=[
                {"role": "system", "content":
                    "Translate the user's message to English. Reply with ONLY the "
                    "English translation — no quotes, no notes, no extra words.\n"
                    "This is a jobs chatbot. If the message clearly IS one of these "
                    "menu commands, output that EXACT label (even if worded loosely): "
                    "Post a Job, View Candidates, My Jobs, Credits & Wallet, Buy "
                    "Credits, Upgrade Plan, Job Search, Application Status, "
                    "Recommended Jobs. Otherwise translate literally (a role/skill "
                    "stays a role/skill)."},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return content.strip() or text
    except Exception as exc:  # noqa: BLE001 — never break the turn on a translation error
        log.warning("translate_in_failed", error=str(exc)[:200])
        return text


async def _batch_translate(llm: "LLMClient", items: list[str], to_lang: str) -> dict[int, str]:
    """One batched JSON call → ``{index: translation}`` for the items it returned.
    On a malformed-JSON response returns ``{}`` so the caller falls back per-string.
    An API error (e.g. a 429 rate limit) PROPAGATES — the caller must NOT then fire a
    per-string storm (that only multiplies the rate-limit hits)."""
    payload = {str(j): s for j, s in enumerate(items)}
    system = (
        f"You are a translator. Translate each VALUE in the JSON object to "
        f"{_LANG_NAME[to_lang]}. Keep the SAME keys. Preserve emoji, *bold* markers, "
        "line breaks, numbers, prices, and any code/reference tokens (slugs, ids, "
        "URLs) exactly. Return ONLY a JSON object mapping each key to its translated "
        "string."
    )
    content, _ = await llm.chat(            # API errors propagate to translate_many
        purpose="translate_out",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=1500,
    )
    try:
        data = json.loads(content)
    except Exception as exc:  # noqa: BLE001 — malformed JSON → caller falls back per-string
        log.warning("translate_batch_parse_failed", error=str(exc)[:160])
        return {}
    return {j: data[str(j)] for j in range(len(items))
            if isinstance(data.get(str(j)), str) and data[str(j)].strip()}


async def _one_translate(llm: "LLMClient", text: str, to_lang: str) -> str | None:
    """Translate a single string (robust fallback — no JSON to misparse). None on
    error so the caller keeps the English original."""
    try:
        content, _ = await llm.chat(
            purpose="translate_out_one",
            messages=[
                {"role": "system", "content":
                    f"Translate the user's text to {_LANG_NAME[to_lang]}. Preserve emoji, "
                    "*bold* markers, line breaks, numbers, prices, and code/reference tokens. "
                    "Reply with ONLY the translation — nothing else."},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        return content.strip() or None
    except Exception as exc:  # noqa: BLE001
        log.warning("translate_one_failed", error=str(exc)[:200])
        return None


async def translate_many(llm: "LLMClient", texts: list[str], *, to_lang: str) -> list[str]:
    """Translate strings to ``to_lang``, returning a same-length list. No-op for
    English / unsupported lang / empty. Per-string cached. ROBUST: tries ONE batched
    JSON call for efficiency, then falls back to a per-string call (in parallel) for
    anything the batch missed or mangled — so a single malformed batch never English-
    dumps the whole turn. Best-effort: originals are kept where translation fails."""
    if to_lang == "en" or to_lang not in SUPPORTED_LANGS or not texts:
        return list(texts)
    out: list[str | None] = [None] * len(texts)
    todo: list[tuple[int, str]] = []
    for i, s in enumerate(texts):
        if not _translatable(s):
            out[i] = s
            continue
        glossed = _glossary(s, to_lang)            # authoritative fixed-term override
        if glossed is not None:
            out[i] = glossed
            continue
        cached = _cache_get((s, to_lang))
        if cached is not None:
            out[i] = cached
        else:
            todo.append((i, s))
    if todo:
        try:
            batched = await _batch_translate(llm, [s for _, s in todo], to_lang)
        except Exception as exc:  # noqa: BLE001 — API error (e.g. 429): keep originals,
            # do NOT fire a per-string storm (it only multiplies the rate-limit hits).
            log.warning("translate_batch_failed", error=str(exc)[:160])
            for i, s in todo:
                out[i] = s
            return [o if o is not None else texts[k] for k, o in enumerate(out)]
        missing: list[tuple[int, str]] = []
        for k, (i, s) in enumerate(todo):
            tr = batched.get(k)
            if tr:
                _cache_put((s, to_lang), tr)
                out[i] = tr
            else:
                missing.append((i, s))
        if missing:                                  # per-string fallback (parallel)
            results = await asyncio.gather(
                *[_one_translate(llm, s, to_lang) for _, s in missing])
            for (i, s), tr in zip(missing, results):
                if tr:
                    _cache_put((s, to_lang), tr)
                    out[i] = tr
                else:
                    out[i] = s
    return [o if o is not None else texts[k] for k, o in enumerate(out)]


async def translate_block(llm: "LLMClient", text: str, *, to_lang: str) -> str:
    """Translate a MULTI-LINE block line-by-line, so FIXED lines hit the glossary
    deterministically (e.g. 'Our team will review it and contact you shortly.') while
    only the remaining lines need the LLM. Blank lines / structure preserved."""
    if to_lang == "en" or to_lang not in SUPPORTED_LANGS or not text:
        return text
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.strip()]
    if not idx:
        return text
    translated = await translate_many(llm, [lines[i] for i in idx], to_lang=to_lang)
    for i, tr in zip(idx, translated):
        lines[i] = tr
    return "\n".join(lines)


# --- interactive-payload label localization ---------------------------------
# WhatsApp Cloud API length caps per field — a translated label can be longer
# than its English source, so each is re-truncated to its cap (else Meta rejects
# the whole interactive message).
_WA_CAPS = {"body": 1024, "header": 60, "list_btn": 20, "section": 24,
            "row_title": 24, "row_desc": 72, "reply": 20, "cta": 20}


def _trunc(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def _interactive_slots(payload: dict | None) -> list[tuple[Callable[[str], None], str]]:
    """``(setter, text)`` for every USER-VISIBLE label in a WhatsApp interactive
    payload — body, text header, list button/section/row titles + descriptions,
    reply-button titles, and the cta display_text. Each setter re-truncates to the
    field's WhatsApp cap. IDs and URLs are NEVER included (so taps still route)."""
    out: list[tuple[Callable[[str], None], str]] = []
    inter = (payload or {}).get("interactive") or {}

    def slot(obj: Any, key: str, cap: int) -> None:
        if isinstance(obj, dict) and isinstance(obj.get(key), str) and obj[key].strip():
            out.append(((lambda v, o=obj, k=key, c=cap: o.__setitem__(k, _trunc(v, c))), obj[key]))

    slot(inter.get("body"), "text", _WA_CAPS["body"])
    header = inter.get("header")
    if isinstance(header, dict) and header.get("type") == "text":
        slot(header, "text", _WA_CAPS["header"])
    action = inter.get("action") or {}
    slot(action, "button", _WA_CAPS["list_btn"])             # list "View roles" button
    for sec in action.get("sections") or []:
        slot(sec, "title", _WA_CAPS["section"])
        for row in sec.get("rows") or []:
            slot(row, "title", _WA_CAPS["row_title"])
            slot(row, "description", _WA_CAPS["row_desc"])
    for btn in action.get("buttons") or []:
        slot(btn.get("reply"), "title", _WA_CAPS["reply"])   # NOT reply.id
    slot(action.get("parameters"), "display_text", _WA_CAPS["cta"])  # NOT parameters.url
    return out


def collect_localizable(result: dict) -> tuple[list[str], list[Callable[[str], None]]]:
    """``(texts, setters)`` for ALL user-visible text in an agent result: the reply,
    the paced delivery bubbles, and every interactive label (job cards, role/category
    lists, buttons, cta). IDs/URLs are never included. Translate ``texts`` and call
    each ``setter`` with the result to localize the whole turn in place."""
    texts: list[str] = []
    setters: list[Callable[[str], None]] = []

    def add(text: str | None, setter: Callable[[str], None]) -> None:
        if text and text.strip():
            texts.append(text)
            setters.append(setter)

    dr = result.get("draft_response")
    if dr:
        add(dr, lambda v: result.__setitem__("draft_response", v))
    for bubble in result.get("delivery_plan") or []:
        if isinstance(bubble, dict) and bubble.get("text"):
            add(bubble["text"], (lambda v, b=bubble: b.__setitem__("text", v)))
    for setter, text in _interactive_slots(result.get("whatsapp_interactive")):
        add(text, setter)
    for msg in result.get("whatsapp_messages") or []:
        for setter, text in _interactive_slots(msg):
            add(text, setter)
    return texts, setters


def _norm(s: str | None) -> str:
    """Normalize for comparison: lowercase + collapse whitespace + drop trailing
    punctuation (so 'தயாரிப்பு' matches 'தயாரிப்பு.' / 'தயாரிப்பு ')."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s.strip(" .,:;!?-—·()[]")


_REVERSE_GLOSSARY: dict[str, dict[str, str]] = {}


def from_glossary(text: str, lang: str) -> str | None:
    """Reverse-map a typed LOCALIZED label back to its canonical English (the glossary
    key). So typing a translated menu label / category name ("வேலை இடுகையிடு") routes
    EXACTLY like tapping it ("Post a Job") — no fragile LLM round-trip. None if no
    match. The reverse map is built once per language and cached."""
    if lang == "en" or not text or lang not in _GLOSSARY:
        return None
    rev = _REVERSE_GLOSSARY.get(lang)
    if rev is None:
        rev = {_norm(v): k for k, v in _GLOSSARY[lang].items()}
        _REVERSE_GLOSSARY[lang] = rev
    return rev.get(_norm(text))


async def category_from_translation(
    llm: "LLMClient", text: str, categories: list[str], lang: str,
) -> str | None:
    """Reverse-map a phrase typed in ``lang`` to its English category by matching it
    against the bot's OWN translations of the category names. This makes the round
    trip CONSISTENT: if the category list showed "Manufacturing" as "தயாரிப்பு",
    typing "தயாரிப்பு" maps straight back to "Manufacturing" — no fragile English
    round-trip. Returns the exact English category name or None."""
    if lang == "en" or not text or not categories:
        return None
    norm = _norm(text)
    if not norm:
        return None
    translated = await translate_many(llm, categories, to_lang=lang)  # cached
    pairs = list(zip(categories, translated))
    for cat, tr in pairs:                       # exact match first
        if _norm(tr) == norm:
            return cat
    for cat, tr in pairs:                       # then a contained match (multi-word)
        nt = _norm(tr)
        if nt and (nt in norm or norm in nt):
            return cat
    return None


# --- LLM category classification (synonym / translated-term fallback) --------
async def classify_category(llm: "LLMClient", text: str, categories: list[str]) -> str | None:
    """Map a free-text phrase to ONE known job category via the LLM — but only when
    it names a broad field/area/industry (so a synonym or translated term like
    "transport" → "Logistics & Supply Chain" resolves), NOT a specific role/skill.
    Returns the EXACT category name from ``categories`` or None. Best-effort."""
    text = (text or "").strip()
    if not text or not categories:
        return None
    try:
        cat_list = "\n".join(f"- {c}" for c in categories)
        content, _ = await llm.chat(
            purpose="classify_category",
            messages=[
                {"role": "system", "content":
                    "Map the user's phrase to ONE job category from the list. Reply with the "
                    "EXACT category name (copied from the list) if the phrase refers to a broad "
                    "job field, area, or industry. Reply EXACTLY 'NONE' if it's a specific job "
                    "title/role, a skill, or doesn't clearly fit a category. Output ONLY the "
                    "category name or NONE.\nCategories:\n" + cat_list},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        ans = (content or "").strip().strip('"').strip()
        if not ans or ans.upper() == "NONE":
            return None
        for c in categories:                       # accept only an exact catalog name
            if c.lower() == ans.lower():
                return c
        return None
    except Exception as exc:  # noqa: BLE001 — a classify miss must never break the turn
        log.warning("classify_category_failed", error=str(exc)[:200])
        return None
