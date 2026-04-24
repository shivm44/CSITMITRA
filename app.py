# ============================================================
#  GGU CSITmitra — Flask Web Chatbot  (Production-Ready)
#  Guru Ghasidas Vishwavidyalaya, Bilaspur
#  Department of Computer Science & Information Technology
#
#  v3.0 — deployment-safe:
#    - Per-user context stored in Flask server-side sessions
#    - NLTK with graceful fallback (no crash if data missing)
#    - gunicorn-compatible (no global mutable state per user)
#    - ggu_data.json driven — update data without code changes
#    - All intents: courses, faculty, fees, admission, placement,
#      scholarship, research, exam, comparison, syllabus, hostel
# ============================================================

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
import string
import random
import difflib
import datetime
import textwrap
import secrets

# ── NLTK (graceful degradation if data unavailable) ──────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)
    _lemmatizer = WordNetLemmatizer()
    _test = word_tokenize("test")          # fails fast if data missing
    _lemmatizer.lemmatize("running")
    _USE_NLTK = True
except Exception:
    _USE_NLTK = False

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
CORS(app, supports_credentials=True)

# ── Load knowledge data ───────────────────────────────────────────────────────
def load_data() -> dict:
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggu_data.json")
    if not os.path.exists(data_file):
        return {}
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)

DATA = load_data()   # read once at startup — immutable, safe for multi-worker

# ── Session context helpers ───────────────────────────────────────────────────
# All per-user state lives in Flask's signed cookie session.
# No global mutable dicts — safe under gunicorn with multiple workers/threads.

def _ctx() -> dict:
    """Return the current user's context dict, creating it if absent."""
    if "ctx" not in session:
        session["ctx"] = {"last_intent": None, "last_course": None, "last_faculty": None}
    return session["ctx"]

def _set_ctx(**kwargs):
    ctx = _ctx()
    ctx.update(kwargs)
    session["ctx"] = ctx          # mark session as modified
    session.modified = True

def _history() -> list:
    """Return in-memory history for this session (used for /history endpoint)."""
    if "history" not in session:
        session["history"] = []
    return session["history"]


# ── JSONBin.io persistent logging ─────────────────────────────────────────────
import urllib.request

JSONBIN_API_KEY = os.environ.get("JSONBIN_API_KEY", "")
JSONBIN_BIN_ID  = os.environ.get("JSONBIN_BIN_ID", "")
ADMIN_PASSWORD  = os.environ.get("ADMIN_PASSWORD", "csitmitra@admin")


def _jsonbin_headers(with_content=False):
    h = {"X-Master-Key": JSONBIN_API_KEY, "X-Bin-Versioning": "false"}
    if with_content:
        h["Content-Type"] = "application/json"
    return h


def _read_bin() -> list:
    if not JSONBIN_API_KEY or not JSONBIN_BIN_ID:
        return []
    try:
        url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}/latest"
        req = urllib.request.Request(url, headers=_jsonbin_headers())
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("record", {}).get("logs", [])
    except Exception as e:
        app.logger.warning(f"JSONBin read failed: {e}")
        return []


def _write_bin(logs: list):
    if not JSONBIN_API_KEY or not JSONBIN_BIN_ID:
        return
    try:
        payload = json.dumps({"logs": logs}).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}",
            data=payload, headers=_jsonbin_headers(with_content=True), method="PUT"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        app.logger.warning(f"JSONBin write failed: {e}")


def _create_bin():
    try:
        payload = json.dumps({"logs": []}).encode("utf-8")
        headers = _jsonbin_headers(with_content=True)
        headers["X-Bin-Name"] = "csitmitra-logs"
        headers["X-Bin-Private"] = "true"
        req = urllib.request.Request("https://api.jsonbin.io/v3/b", data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())["metadata"]["id"]
    except Exception as e:
        app.logger.warning(f"JSONBin create failed: {e}")
        return ""


def _ensure_bin():
    global JSONBIN_BIN_ID
    if JSONBIN_API_KEY and not JSONBIN_BIN_ID:
        JSONBIN_BIN_ID = _create_bin()
        if JSONBIN_BIN_ID:
            app.logger.info(f"JSONBin bin created: {JSONBIN_BIN_ID} — save this as JSONBIN_BIN_ID env var!")


def _log(user_msg: str, intent: str, bot_reply: str):
    """Append one turn to session cookie history AND to JSONBin."""
    ts = datetime.datetime.now()
    turn_num = len(_history()) + 1

    # Cookie history (for /history endpoint)
    h = _history()
    h.append({
        "turn":   turn_num,
        "time":   ts.strftime("%H:%M:%S"),
        "user":   user_msg,
        "intent": intent,
        "bot":    bot_reply[:400],
    })
    session["history"] = h
    session.modified = True

    # JSONBin persistent log
    try:
        _ensure_bin()
        if not JSONBIN_API_KEY:
            return

        session_id = session.get("log_file_name")
        if not session_id:
            sid_ts = ts.strftime("%Y%m%d_%H%M%S")
            sid_uid = secrets.token_hex(4)
            session_id = f"chat_{sid_ts}_{sid_uid}"
            session["log_file_name"] = session_id
            session.modified = True

        logs = _read_bin()
        session_entry = next((s for s in logs if s.get("session_id") == session_id), None)
        if not session_entry:
            session_entry = {
                "session_id": session_id,
                "started": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "turns": []
            }
            logs.append(session_entry)

        session_entry["turns"].append({
            "turn":   turn_num,
            "time":   ts.strftime("%Y-%m-%d %H:%M:%S"),
            "user":   user_msg,
            "intent": intent,
            "bot":    bot_reply[:400],
        })
        session_entry["last_active"] = ts.strftime("%Y-%m-%d %H:%M:%S")

        # Keep only last 200 sessions to stay within JSONBin free limits
        if len(logs) > 200:
            logs = logs[-200:]

        _write_bin(logs)
    except Exception as e:
        app.logger.warning(f"Log save failed: {e}")

# ── Intent keyword map ────────────────────────────────────────────────────────
INTENTS = {
    "greeting":    ["hi", "hello", "hey", "howdy", "greet", "morning", "afternoon", "evening", "namaste", "start"],
    "farewell":    ["bye", "goodbye", "exit", "quit", "later", "tata", "cya", "close", "end"],
    "thanks":      ["thanks", "thank", "appreciate", "grateful", "cheers", "helpful"],
    "about":       ["about", "overview", "history", "established", "founded",
                    "info", "ggu", "ggv", "university", "vishwavidyalaya", "department", "csit"],
    "courses":     ["course", "program", "programme", "degree", "branch", "stream", "study",
                    "offer", "available", "subject", "list", "all"],
    "mca":         ["mca", "master", "application"],
    "bca":         ["bca", "bachelor"],
    "msc_cs":      ["msc", "m.sc"],
    "bsc_cs":      ["bsc", "b.sc", "undergraduate"],
    "phd":         ["phd", "ph.d", "doctoral", "doctorate", "vret"],
    "help":        ["help", "menu", "option", "command", "guide", "assist", "support"],
    "compare":     ["compare", "difference", "vs", "versus", "better", "which", "choose",
                    "prefer", "between", "best", "suit"],
    "fees":        ["fee", "fees", "cost", "price", "tuition", "charge", "payment", "money",
                    "rupee", "expense", "afford"],
    "hostel":      ["hostel", "accommodation", "room", "mess", "stay", "residence", "dorm"],
    "admission":   ["admission", "admit", "apply", "application", "enroll", "join", "entrance",
                    "eligibility", "criteria", "requirement", "cuet", "gate", "process", "samarth"],
    "faculty":     ["faculty", "professor", "teacher", "staff", "lecturer", "instructor",
                    "who", "teach"],
    "professor":   ["babita", "santosh", "pushplata", "rajwant", "sushma", "akhilesh", "vikas",
                    "vineet", "prashant", "vivek", "abhishek", "amitesh", "majhi", "pujari",
                    "jaiswal", "shrivas", "pandey", "awasthi", "vaishnav", "sarthe", "patel", "jha", "rao"],
    "facilities":  ["facility", "facilities", "library", "lab", "wifi", "internet",
                    "sport", "cafeteria", "infrastructure", "amenity", "medical", "bus", "transport", "campus"],
    "placement":   ["placement", "job", "package", "salary", "recruit", "recruiter", "hire",
                    "lpa", "company", "career", "placed", "opportunity"],
    "scholarship": ["scholarship", "financial", "aid", "waiver", "merit", "stipend",
                    "concession", "fund", "fellowship", "nsp", "free"],
    "research":    ["research", "publication", "conference", "paper", "project", "journal",
                    "innovation", "patent", "highlight"],
    "exam":        ["exam", "examination", "test", "internal", "assessment",
                    "marks", "pattern", "grade", "cgpa", "evaluation"],
    "location":    ["location", "address", "where", "place", "city", "situated", "koni",
                    "bilaspur", "chhattisgarh", "distance", "map"],
    "contact":     ["contact", "phone", "email", "number", "reach", "call", "mail", "website", "helpline"],
    "semester":    ["semester", "syllabus", "curriculum", "sem"],
}

SMALL_TALK = {
    "greeting": [
        "Namaste! 🙏 I am **CSITmitra**, your assistant for GGU CSIT Dept.\n\nAsk me about: courses · faculty · fees · admission · placement · research",
        "Hello! I am **CSITmitra**, your GGU CSIT assistant.\n\nWhat would you like to know?",
    ],
    "farewell": [
        "Goodbye! Best of luck with your studies! 🎓 — GGU CSIT Dept.",
        "See you! Feel free to return anytime. 😊",
    ],
    "thanks": [
        "You're welcome! Any other questions about CSIT at GGU?",
        "Happy to help! Anything else you'd like to know?",
    ],
    "unknown": [
        "I'm not sure about that. Try asking about:\n\ncourses · fees · faculty · admission · placement · scholarship · research",
        "Could you rephrase? For example:\n• 'Tell me about MCA'\n• 'Who is Dr. Shrivas?'\n• 'What are the fees?'",
    ],
}

HELP_TEXT = textwrap.dedent("""\
    🤖 **GGU CSITmitra — Help**

    **Topics you can ask about:**
    • Courses & Programmes (BCA, MCA, M.Sc., B.Sc., Ph.D.)
    • Fee Structure & Hostel Fees
    • Faculty Profiles
    • Admission Process (CUET, SAMARTH Portal)
    • Placements & Recruiters
    • Scholarships & Financial Aid
    • Research & Publications
    • Exam Pattern & CGPA Grading
    • Campus Facilities
    • Contact & Location

    **Example queries:**
    • "Tell me about MCA"
    • "What are the fees for BCA?"
    • "Who is Dr. Shrivas?"
    • "MCA vs B.Sc. comparison"
    • "Ph.D. admission process"
    • "MCA syllabus"
    • "Placement highlights"
    • "Hostel fees"

    **Tips:**
    • Context is remembered per session — ask about MCA, then just type 'fees?' or 'syllabus'
    • Use a professor's last name: 'Tell me about Majhi'
    • Compare two programmes: 'MCA vs Ph.D'
""")

# ── NLP helpers ───────────────────────────────────────────────────────────────

def preprocess(text: str) -> list:
    text_lower = text.lower()
    if _USE_NLTK:
        tokens = word_tokenize(text_lower)
        tokens = [t for t in tokens if t not in string.punctuation]
        return [_lemmatizer.lemmatize(t) for t in tokens]
    return [t.strip(string.punctuation) for t in text_lower.split() if t.strip(string.punctuation)]


ALL_KEYWORDS = {kw: intent for intent, kws in INTENTS.items() for kw in kws}


def fuzzy_match(token: str, cutoff: float = 0.75):
    matches = difflib.get_close_matches(token, ALL_KEYWORDS.keys(), n=1, cutoff=cutoff)
    return ALL_KEYWORDS[matches[0]] if matches else None


def fuzzy_match_professor_name(token: str) -> str | None:
    """Return the closest professor keyword for a misspelled name token."""
    prof_keywords = INTENTS["professor"]
    matches = difflib.get_close_matches(token, prof_keywords, n=1, cutoff=0.70)
    return matches[0] if matches else None


def detect_intent(tokens: list) -> tuple:
    scores = {intent: 0 for intent in INTENTS}
    matched_professor = None

    for token in tokens:
        for intent, keywords in INTENTS.items():
            if token in keywords:
                scores[intent] += 2
                if intent == "professor":
                    matched_professor = token
        fuzzy = fuzzy_match(token)
        if fuzzy:
            scores[fuzzy] += 1
            if fuzzy == "professor" and matched_professor is None:
                # fuzzy hit on a professor keyword — resolve the actual keyword
                resolved = fuzzy_match_professor_name(token)
                if resolved:
                    matched_professor = resolved

    lower = " ".join(tokens)
    matched_course = None
    if "mca" in lower:
        matched_course = "MCA (Master of Computer Applications)"
    elif "bca" in lower:
        matched_course = "BCA (Bachelor of Computer Applications)"
    elif "bsc" in lower or "b.sc" in lower:
        matched_course = "B.Sc. (Computer Science)"
    elif "msc" in lower or "m.sc" in lower:
        matched_course = "M.Sc. (Computer Science)"
    elif "phd" in lower or "ph.d" in lower or "doctoral" in lower:
        matched_course = "Ph.D. (Computer Science / IT)"

    _course_boost = {
        "MCA (Master of Computer Applications)":   "mca",
        "BCA (Bachelor of Computer Applications)": "bca",
        "B.Sc. (Computer Science)":                "bsc_cs",
        "M.Sc. (Computer Science)":                "msc_cs",
        "Ph.D. (Computer Science / IT)":           "phd",
    }
    if matched_course and matched_course in _course_boost:
        scores[_course_boost[matched_course]] += 5

    syllabus_words = {"syllabus", "semester", "sem", "curriculum"}
    want_syllabus = bool(syllabus_words & set(tokens)) and matched_course is not None

    compare_words = {"vs", "versus", "compare", "comparison", "difference", "between"}
    if bool(compare_words & set(tokens)):
        scores["compare"] += 6

    name_tokens = [t for t in tokens if t in INTENTS["professor"]]
    if not name_tokens:
        # Fuzzy fallback: catch typos like "srivas" → "shrivas"
        for t in tokens:
            resolved = fuzzy_match_professor_name(t)
            if resolved:
                name_tokens.append(resolved)
    if name_tokens:
        matched_professor = " ".join(name_tokens)
        scores["professor"] += 6

    best = max(scores, key=scores.get)
    sub = (matched_course + ":syllabus") if want_syllabus else (matched_professor or matched_course)
    return (best if scores[best] > 0 else "unknown", sub)


def _spell_hint(user_input: str) -> str:
    tokens = preprocess(user_input)
    suggestions = set()
    for tok in tokens:
        if len(tok) < 3:
            continue
        close = difflib.get_close_matches(tok, ALL_KEYWORDS.keys(), n=2, cutoff=0.75)
        for c in close:
            if c != tok:
                suggestions.add(c)
    if suggestions:
        return "💡 Did you mean: " + " / ".join(sorted(suggestions)[:3]) + "?"
    return ""

# ── Course comparison ─────────────────────────────────────────────────────────

_COMPARE_KEYS = ("duration", "intake", "eligibility", "fee_per_semester", "total_approx_fee", "fellowship")
_COMPARE_LABELS = {
    "duration":         "Duration",
    "intake":           "Intake",
    "eligibility":      "Eligibility",
    "fee_per_semester": "Fee / Semester",
    "total_approx_fee": "Total Fee",
    "fellowship":       "Fellowship",
}

def compare_courses(lower: str):
    mapping = {
        "bca":  "BCA (Bachelor of Computer Applications)",
        "mca":  "MCA (Master of Computer Applications)",
        "msc":  "M.Sc. (Computer Science)",
        "bsc":  "B.Sc. (Computer Science)",
        "phd":  "Ph.D. (Computer Science / IT)",
    }
    found = []
    for kw, full in mapping.items():
        if kw in lower and full not in found:
            found.append(full)
    if len(found) < 2:
        return None

    courses_data = DATA.get("courses", {})
    rows = ["⚖️ **Course Comparison**\n"]
    for key in _COMPARE_KEYS:
        label = _COMPARE_LABELS[key]
        row = f"**{label}:**\n"
        for name in found:
            val = courses_data.get(name, {}).get(key, "—")
            row += f"• {name}: {val}\n"
        rows.append(row)
    rows.append(f"💡 Ask 'Tell me about {found[0]}' for full details.")
    return "\n".join(rows)

# ── Response builder ──────────────────────────────────────────────────────────

_COURSE_ONLY_INTENTS = {
    "fees", "hostel", "admission", "faculty", "professor", "placement", "scholarship",
    "research", "exam", "location", "contact", "help", "compare",
    "facilities", "semester", "about", "courses",
}

COURSE_MAP = {
    "mca":    "MCA (Master of Computer Applications)",
    "bca":    "BCA (Bachelor of Computer Applications)",
    "msc_cs": "M.Sc. (Computer Science)",
    "bsc_cs": "B.Sc. (Computer Science)",
    "phd":    "Ph.D. (Computer Science / IT)",
}

def build_response(intent: str, sub) -> str:
    ctx = _ctx()

    if intent in SMALL_TALK:
        return random.choice(SMALL_TALK[intent])

    d = DATA
    if not d:
        return "⚠️ Data file not found. Please ensure ggu_data.json is in the same folder as app.py."

    # ── About ────────────────────────────────────────────────────────────────
    if intent == "about":
        _set_ctx(last_intent="about")
        return (
            f"🏛️ **{d.get('university', 'Guru Ghasidas Vishwavidyalaya')}**\n"
            f"{d.get('department', '')}\n\n"
            f"• **Location**: {d.get('location', '')}\n"
            f"• **Founded**: {d.get('established', '')}\n"
            f"• **NAAC**: {d.get('naac', '')}\n"
            f"• **Campus**: {d.get('campus_size', '')}\n"
            f"• **Website**: {d.get('website', '')}\n\n"
            "The CSIT Department started in 1990 with PGDCA, then added M.Sc. CS & IT (1996), "
            "MCA (1998, AICTE approved), B.Sc. CS, BCA, and Ph.D. programmes. "
            "Faculty actively collaborate with institutions across India and abroad."
        )

    # ── All courses ──────────────────────────────────────────────────────────
    if intent == "courses":
        _set_ctx(last_intent="courses")
        lines = ["🎓 **Programmes offered by CSIT Dept., GGV:**\n"]
        for prog, info in d.get("courses", {}).items():
            lines.append(
                f"**{prog}**\n"
                f"Duration: {info['duration']} | Seats: {info.get('intake', 'N/A')}\n"
                f"Fee/sem: {info.get('fee_per_semester', 'N/A')}\n"
            )
        lines.append("💡 Ask 'Tell me about MCA' or 'MCA syllabus' for full details.")
        return "\n".join(lines)

    # ── Specific course detail ───────────────────────────────────────────────
    want_syllabus_only = isinstance(sub, str) and sub.endswith(":syllabus")
    if want_syllabus_only:
        sub = sub.replace(":syllabus", "")

    course_key = COURSE_MAP.get(intent) or (sub if sub and sub in d.get("courses", {}) else None)
    if not course_key and ctx.get("last_course") and intent not in _COURSE_ONLY_INTENTS:
        course_key = ctx["last_course"]
    if want_syllabus_only and not course_key and ctx.get("last_course"):
        course_key = ctx["last_course"]

    if course_key and course_key in d.get("courses", {}):
        _set_ctx(last_course=course_key, last_intent="course_detail")
        c = d["courses"][course_key]

        if want_syllabus_only:
            if "semesters" in c:
                lines = [f"📋 **Semester-wise Subjects — {course_key}:**\n"]
                for sem, subjects in c["semesters"].items():
                    lines.append(f"**{sem}:**")
                    for subj in subjects:
                        lines.append(f"• {subj}")
                    lines.append("")
                return "\n".join(lines)
            return f"Semester details not available for {course_key}."

        lines = [f"📘 **{course_key}**\n"]
        lines.append(f"• **Duration**: {c['duration']}")
        if "intake" in c:
            lines.append(f"• **Intake**: {c['intake']}")
        lines.append(f"• **Eligibility**: {c['eligibility']}")
        if "exit_option" in c:
            lines.append(f"• **Exit Option**: {c['exit_option']}")
        lines.append(f"• **Fee/Semester**: {c.get('fee_per_semester', 'N/A')}")
        if "total_approx_fee" in c:
            lines.append(f"• **Total Fee**: {c['total_approx_fee']}")
        if "fellowship" in c:
            lines.append(f"• **Fellowship**: {c['fellowship']}")
        if "research_areas" in c:
            lines.append(f"• **Research Areas**: {', '.join(c['research_areas'])}")
        if "semesters" in c:
            lines.append("\n📋 **Semester-wise Subjects:**")
            for sem, subjects in c["semesters"].items():
                lines.append(f"\n**{sem}:**")
                for subj in subjects:
                    lines.append(f"• {subj}")
        return "\n".join(lines)

    # ── Semester / syllabus ──────────────────────────────────────────────────
    if intent == "semester":
        lc = ctx.get("last_course")
        if lc and lc in d.get("courses", {}):
            c = d["courses"][lc]
            if "semesters" in c:
                lines = [f"📋 **Semester-wise Subjects — {lc}:**\n"]
                for sem, subjects in c["semesters"].items():
                    lines.append(f"**{sem}:**")
                    for subj in subjects:
                        lines.append(f"• {subj}")
                    lines.append("")
                return "\n".join(lines)
        return "Please specify a course first. Example: 'MCA syllabus' or 'BCA semesters'"

    # ── Fees ─────────────────────────────────────────────────────────────────
    if intent == "fees":
        _set_ctx(last_intent="fees")
        fee = d.get("fees", {})
        ctx_course = ctx.get("last_course")
        if ctx_course and ctx_course in d.get("courses", {}):
            c = d["courses"][ctx_course]
            lines = [f"💰 **Fees — {ctx_course}:**\n"]
            lines.append(f"• **Per Semester**: {c.get('fee_per_semester', 'N/A')}")
            lines.append(f"• **Total**: {c.get('total_approx_fee', 'N/A')}")
            if "fellowship" in c:
                lines.append(f"• **Fellowship**: {c['fellowship']}")
            h = fee.get("hostel", {})
            lines.append(f"\n🏠 **Hostel**: Boys — {h.get('boys', 'N/A')} | Girls — {h.get('girls', 'N/A')}")
            lines.append(f"\n⚠️ {fee.get('note', '')}")
            return "\n".join(lines)
        lines = ["💰 **Fee Structure — CSIT Dept., GGV:**\n"]
        for prog, info in d.get("courses", {}).items():
            lines.append(f"**{prog}**\n• Per Semester: {info.get('fee_per_semester', 'N/A')}\n• Total: {info.get('total_approx_fee', 'N/A')}\n")
        h = fee.get("hostel", {})
        lines.append(f"🏠 **Hostel:**\n• Boys: {h.get('boys', 'N/A')}\n• Girls: {h.get('girls', 'N/A')}\n• Note: {h.get('note', '')}")
        if "other_charges" in fee:
            lines.append("\n📋 **Other Charges:**")
            for charge, detail in fee["other_charges"].items():
                lines.append(f"• {charge.replace('_', ' ').title()}: {detail}")
        if "scholarships_that_cover_fees" in fee:
            lines.append("\n🎓 **Scholarships that Cover Fees:**")
            for s in fee["scholarships_that_cover_fees"]:
                lines.append(f"• {s}")
        lines.append(f"\n⚠️ {fee.get('note', '')}")
        return "\n".join(lines)

    # ── Hostel ───────────────────────────────────────────────────────────────
    if intent == "hostel":
        fee = d.get("fees", {})
        h = fee.get("hostel", {})
        return (
            "🏠 **Hostel Information — GGV:**\n\n"
            f"• **Boys Hostel**: {h.get('boys', 'N/A')}\n"
            f"• **Girls Hostel**: {h.get('girls', 'N/A')}\n"
            f"• **Note**: {h.get('note', '')}\n\n"
            "**Facilities include:** Wi-Fi, common room, reading room, mess (hygienic food), 24/7 security.\n\n"
            "📌 Apply separately via SAMARTH Portal at www.ggu.ac.in"
        )

    # ── Admission ────────────────────────────────────────────────────────────
    if intent == "admission":
        _set_ctx(last_intent="admission")
        adm = d.get("admission_process", {})
        _adm_key_map = {
            "BCA (Bachelor of Computer Applications)": "BCA",
            "B.Sc. (Computer Science)":               "B.Sc. CS",
            "M.Sc. (Computer Science)":               "M.Sc. CS",
            "MCA (Master of Computer Applications)":  "MCA",
            "Ph.D. (Computer Science / IT)":          "Ph.D.",
        }
        def _resolve(name):
            mapped = _adm_key_map.get(name, name)
            return mapped if mapped in adm else None
        ctx_course = (_resolve(sub) if sub else None) or _resolve(ctx.get("last_course", ""))
        if ctx_course:
            return (
                f"📝 **Admission — {ctx_course}:**\n\n"
                f"{adm[ctx_course]}\n\n"
                "📌 Apply online at: **www.ggu.ac.in** (SAMARTH Portal)"
            )
        lines = ["📝 **Admission Process — CSIT Dept., GGV:**\n"]
        for prog, info in adm.items():
            lines.append(f"**{prog}:**\n{info}\n")
        lines.append("📌 Apply online at: **www.ggu.ac.in** (SAMARTH Portal)")
        return "\n".join(lines)

    # ── Faculty list ─────────────────────────────────────────────────────────
    if intent == "faculty":
        _set_ctx(last_intent="faculty")
        lines = ["👨‍🏫 **Teaching Faculty — CSIT Dept., GGV:**\n"]
        for f in d.get("faculty", []):
            lines.append(f"• **{f['name']}** | {f['designation']}\n  _{f['specialization']}_\n")
        lines.append("💡 Ask 'Tell me about Dr. Babita Majhi' for a full profile.")
        return "\n".join(lines)

    # ── Specific professor ────────────────────────────────────────────────────
    if intent == "professor" or (sub and isinstance(sub, str) and any(
        sub.lower() in f["name"].lower() for f in d.get("faculty", [])
    )):
        query = (sub or "").lower()
        matched_f = next((f for f in d.get("faculty", []) if query in f["name"].lower()), None)
        if not matched_f and ctx.get("last_faculty"):
            matched_f = next((f for f in d.get("faculty", []) if ctx["last_faculty"].lower() in f["name"].lower()), None)
        if matched_f:
            _set_ctx(last_faculty=matched_f["name"])
            f = matched_f
            lines = [f"👤 **{f['name']}**\n"]
            lines.append(f"• **Designation**: {f['designation']}")
            lines.append(f"• **Qualification**: {f['qualification']}")
            lines.append(f"• **Specialization**: {f['specialization']}")
            lines.append(f"• **Subjects Taught**: {', '.join(f['subjects_taught'])}")
            lines.append(f"• **Email**: {f['email']}")
            if f.get("phone") and "N/A" not in f["phone"]:
                lines.append(f"• **Phone**: {f['phone']}")
            if f.get("notable"):
                lines.append(f"• **Notable**: {f['notable']}")
            if f.get("google_scholar"):
                lines.append(f"• **Google Scholar**: {f['google_scholar']}")
            if f.get("orcid"):
                lines.append(f"• **ORCID**: {f['orcid']}")
            if f.get("joined"):
                lines.append(f"• **Joined GGV**: {f['joined']}")
            return "\n".join(lines)
        return (
            "I couldn't find that professor. Try using their last name.\n"
            "Example: 'Tell me about Majhi' or 'Who is Dr. Shrivas?'\n"
            "Type 'faculty' to see the full list."
        )

    # ── Facilities ───────────────────────────────────────────────────────────
    if intent == "facilities":
        lines = ["🏫 **Campus & Departmental Facilities:**\n"]
        for item in d.get("facilities", []):
            lines.append(f"• {item}")
        return "\n".join(lines)

    # ── Placement ────────────────────────────────────────────────────────────
    if intent == "placement":
        p = d.get("placement", {})
        return (
            "💼 **Placement Highlights — GGV CSIT:**\n\n"
            f"• **UG 4-yr Median Package**: {p.get('ug_4yr_median', 'N/A')}\n"
            f"• **PG 2-yr Median Package**: {p.get('pg_2yr_median', 'N/A')}\n"
            f"• **Highest Package**: {p.get('highest', 'N/A')}\n"
            f"• **UG Students Placed**: {p.get('students_placed_ug', 'N/A')}\n"
            f"• **PG Students Placed**: {p.get('students_placed_pg', 'N/A')}\n"
            f"• **Placement Rate**: {p.get('placement_rate', 'N/A')}\n"
            f"• **Top Recruiters**: {', '.join(p.get('top_recruiters', []))}\n\n"
            f"📌 {p.get('placement_cell', '')}\n"
            f"{p.get('note', '')}"
        )

    # ── Scholarship ──────────────────────────────────────────────────────────
    if intent == "scholarship":
        lines = ["🏅 **Scholarships & Financial Aid:**\n"]
        for item in d.get("scholarship", []):
            lines.append(f"• {item}")
        lines.append("\n📌 Apply via: scholarships.gov.in or through SAMARTH Portal at www.ggu.ac.in")
        return "\n".join(lines)

    # ── Research ─────────────────────────────────────────────────────────────
    if intent == "research":
        r = d.get("research", {})
        lines = [f"🔬 **Research @ CSIT, GGV:**\n\n{r.get('summary', '')}\n"]
        lines.append(f"**Research Areas:** {', '.join(r.get('areas', []))}\n")
        lines.append("**Recent Highlights:**")
        for h in r.get("recent_highlights", []):
            lines.append(f"• {h}")
        return "\n".join(lines)

    # ── Exam ─────────────────────────────────────────────────────────────────
    if intent == "exam":
        return f"📋 **Examination Pattern:**\n\n{d.get('exam_pattern', 'N/A')}"

    # ── Location ─────────────────────────────────────────────────────────────
    if intent == "location":
        return (
            f"📍 **{d.get('university', '')}**\n"
            f"{d.get('department', '')}\n"
            f"{d.get('location', '')}\n\n"
            "• Located at Koni, approx. 5 km from Bilaspur city\n"
            "• River Arpa runs parallel to the campus\n"
            "• University buses connect Bilaspur city to Koni campus"
        )

    # ── Contact ──────────────────────────────────────────────────────────────
    if intent == "contact":
        return (
            "📞 **Contact — GGV CSIT Dept.:**\n\n"
            f"• **Phone**: {d.get('phone', 'N/A')}\n"
            f"• **Email**: {d.get('email', 'N/A')}\n"
            f"• **Website**: {d.get('website', 'N/A')}"
        )

    # ── Help ─────────────────────────────────────────────────────────────────
    if intent == "help":
        return HELP_TEXT

    # ── Compare ──────────────────────────────────────────────────────────────
    if intent == "compare":
        hint_lower = (sub or "").lower()
        result = compare_courses(hint_lower)
        if result:
            return result
        lc = ctx.get("last_course", "")
        if lc:
            result = compare_courses(hint_lower + " " + lc.lower()[:3])
            if result:
                return result
        return (
            "⚖️ To compare courses, mention two programme names.\n\n"
            "Examples:\n• 'MCA vs M.Sc'\n• 'Compare BCA and MCA'\n• 'Ph.D vs MCA difference'"
        )

    return random.choice(SMALL_TALK["unknown"])


# ── Main chat function ────────────────────────────────────────────────────────

def get_response(user_message: str) -> str:
    msg = user_message.strip()
    low = msg.lower()

    # Feedback shortcuts
    if any(kw in low for kw in ("good bot", "great", "awesome", "perfect", "well done", "nice work")):
        return "😊 Thank you for the kind words! Anything else I can help with?"
    if any(kw in low for kw in ("wrong", "incorrect", "not helpful", "useless")):
        return "😔 Sorry about that! Please rephrase your question or type 'help' to see example queries."

    # Compare shortcut (catches "MCA vs BCA" before full NLP)
    if any(kw in low for kw in ("vs", "versus", "compare", "difference between")):
        result = compare_courses(low)
        if result:
            _log(msg, "compare", result)
            return result

    tokens = preprocess(msg)
    intent, sub = detect_intent(tokens)
    response = build_response(intent, sub)

    if intent == "unknown":
        hint = _spell_hint(msg)
        if hint:
            response = (
                "⚠️ I couldn't find what you're looking for. Please check the name or keyword spelling.\n\n"
                + hint
                + "\n\nOr type **'help'** to see all available topics and example queries."
            )
        else:
            response = (
                "⚠️ I didn't understand that. Please check the spelling and try again.\n\n"
                "Try asking about:\n• courses · fees · faculty · admission · placement · scholarship\n\n"
                "💡 For faculty: use their **last name** e.g. *'Who is Shrivas?'*\n"
                "Type **'help'** for all example queries."
            )

    _log(msg, intent, response)
    return response


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({
                "response": "Please enter a message!",
                "timestamp": datetime.datetime.now().strftime("%I:%M %p"),
            })

        bot_response = get_response(user_message)
        return jsonify({
            "response": bot_response,
            "timestamp": datetime.datetime.now().strftime("%I:%M %p"),
        })

    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({
            "response": "Sorry, something went wrong. Please try again!",
            "timestamp": datetime.datetime.now().strftime("%I:%M %p"),
        }), 500


@app.route("/history", methods=["GET"])
def history():
    return jsonify({"history": _history()})


@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"status": "Session reset successfully."})


@app.route("/logs", methods=["GET", "POST"])
def list_logs():
    """Password-protected admin page — shows all chat sessions from JSONBin."""

    # ── Auth check ────────────────────────────────────────────────────────────
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin_auth"] = True
            session.modified = True
        else:
            return _login_page(error="Wrong password. Try again.")

    if not session.get("admin_auth"):
        return _login_page()

    # ── Show logs ─────────────────────────────────────────────────────────────
    if not JSONBIN_API_KEY:
        return _admin_html("<p style='color:#f88'>⚠️ JSONBIN_API_KEY not set. Add it in Render environment variables.</p>", 0)

    logs = _read_bin()
    logs_sorted = sorted(logs, key=lambda x: x.get("last_active", ""), reverse=True)

    rows = ""
    for s in logs_sorted:
        sid   = s.get("session_id", "?")
        start = s.get("started", "?")
        last  = s.get("last_active", "?")
        turns = len(s.get("turns", []))
        rows += (
            f"<tr>"
            f"<td>{sid}</td>"
            f"<td>{start}</td>"
            f"<td>{last}</td>"
            f"<td>{turns}</td>"
            f"<td><a href='/logs/view/{sid}'>View</a></td>"
            f"</tr>\n"
        )

    table = (
        "<table><tr><th>Session ID</th><th>Started</th><th>Last Active</th><th>Turns</th><th>Action</th></tr>"
        + rows + "</table>"
    ) if logs_sorted else "<p style='color:#888'>No logs yet. Chats will appear here once JSONBIN_API_KEY is set.</p>"

    return _admin_html(table, len(logs_sorted))


@app.route("/logs/view/<path:session_id>", methods=["GET"])
def view_log(session_id):
    """View a specific chat session."""
    if not session.get("admin_auth"):
        return _login_page()
    if not JSONBIN_API_KEY:
        return "JSONBIN_API_KEY not configured", 500

    logs = _read_bin()
    entry = next((s for s in logs if s.get("session_id") == session_id), None)
    if not entry:
        return "Session not found", 404

    turns_html = ""
    for t in entry.get("turns", []):
        turns_html += f"""
        <div class='turn'>
          <div class='meta'>Turn {t['turn']} &nbsp;·&nbsp; {t['time']} &nbsp;·&nbsp; intent: <b>{t['intent']}</b></div>
          <div class='user'>👤 {t['user']}</div>
          <div class='bot'>🤖 {t['bot']}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html><head><title>{session_id}</title>
<style>
  body {{ font-family: monospace; padding: 2rem; background:#1a1a2e; color:#eee; max-width:900px; margin:auto; }}
  a {{ color:#00d4ff; }} h2 {{ color:#00d4ff; }}
  .turn {{ background:#0f3460; border-radius:8px; padding:1rem; margin-bottom:1rem; }}
  .meta {{ color:#888; font-size:.85em; margin-bottom:.5rem; }}
  .user {{ color:#7ecfff; margin-bottom:.4rem; }}
  .bot {{ color:#b8f0b8; white-space:pre-wrap; word-break:break-word; }}
</style></head><body>
<a href="/logs">← Back to logs</a>
<h2>Session: {session_id}</h2>
<p>Started: {entry.get('started')} &nbsp;|&nbsp; Last active: {entry.get('last_active')} &nbsp;|&nbsp; {len(entry.get('turns',[]))} turns</p>
{turns_html if turns_html else "<p style='color:#888'>No turns recorded.</p>"}
</body></html>"""
    return html


@app.route("/logs/logout", methods=["GET"])
def logs_logout():
    session.pop("admin_auth", None)
    return _login_page(error="Logged out.")


def _login_page(error=""):
    err_html = f"<p style='color:#f88'>{error}</p>" if error else ""
    return f"""<!DOCTYPE html>
<html><head><title>CSITmitra Admin Login</title>
<style>
  body {{ font-family: sans-serif; background:#1a1a2e; color:#eee; display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }}
  .box {{ background:#0f3460; padding:2rem 2.5rem; border-radius:12px; min-width:320px; text-align:center; }}
  h2 {{ color:#00d4ff; margin-bottom:1rem; }}
  input {{ width:100%; padding:.6rem; border-radius:6px; border:1px solid #444; background:#1a1a2e; color:#eee; font-size:1rem; margin-bottom:1rem; box-sizing:border-box; }}
  button {{ width:100%; padding:.7rem; background:#00d4ff; color:#000; border:none; border-radius:6px; font-size:1rem; font-weight:bold; cursor:pointer; }}
  button:hover {{ background:#00b8d9; }}
</style></head><body>
<div class='box'>
  <h2>🔒 CSITmitra Admin</h2>
  {err_html}
  <form method='POST' action='/logs'>
    <input type='password' name='password' placeholder='Enter admin password' autofocus>
    <button type='submit'>Login</button>
  </form>
</div>
</body></html>"""


def _admin_html(content: str, count: int):
    return f"""<!DOCTYPE html>
<html><head><title>CSITmitra — Chat Logs</title>
<style>
  body {{ font-family: monospace; padding: 2rem; background:#1a1a2e; color:#eee; }}
  h1 {{ color:#00d4ff; }} table {{ border-collapse:collapse; width:100%; }}
  th,td {{ padding:.5rem 1rem; border:1px solid #444; text-align:left; }}
  th {{ background:#0f3460; }} a {{ color:#00d4ff; }}
  .logout {{ float:right; font-size:.85em; }}
</style></head><body>
<h1>📋 CSITmitra — Chat Logs <span class='logout'><a href='/logs/logout'>Logout</a></span></h1>
<p>{count} session(s) stored in JSONBin.</p>
{content}
</body></html>"""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "nltk": _USE_NLTK,
        "data_loaded": bool(DATA),
        "courses": len(DATA.get("courses", {})),
        "faculty": len(DATA.get("faculty", [])),
        "jsonbin_configured": bool(JSONBIN_API_KEY and JSONBIN_BIN_ID),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
