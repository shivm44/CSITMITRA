# ============================================================
#  GGU CSITmitra — Flask Web Chatbot  (Production-Ready v4.1)
#  Guru Ghasidas Vishwavidyalaya, Bilaspur
#  Department of Computer Science & Information Technology
#
#  v4.1 — Restructured for clarity + lower request-time complexity
#    - Intent lookup: O(1) dict lookup per token instead of
#      scanning every intent's keyword list per token
#    - Faculty lookup: prebuilt name/surname index instead of
#      hardcoded keyword lists + linear scans
#    - Fault-tolerant field getter: tries multiple key spellings
#      (designation/position/role, specialization/specialisation,
#      lists vs strings) so a JSON key-name mismatch no longer
#      silently blanks out a field
#    - Response building: dispatch table (intent -> handler)
#      instead of a long if/elif chain
#    - Restores v3.1 features while keeping v4's indexed architecture
#    - Fixes JSON contact-list rendering and faculty field extraction
# ============================================================

import flask
from flask_cors import CORS
import os
import re
import ast
import json
import string
import random
import difflib
import datetime
import textwrap
import secrets
from functools import lru_cache
from collections import Counter

# ── NLTK (graceful degradation if data unavailable) ──────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _lemmatizer = WordNetLemmatizer()
    word_tokenize("test")            # fails fast if data missing
    _lemmatizer.lemmatize("running")
    USE_NLTK = True
except Exception:
    USE_NLTK = False


# ============================================================
#  1. INTENT KEYWORDS  (data, kept separate from logic)
# ============================================================

INTENTS = {
    "greeting":    ["hi", "hello", "hey", "howdy", "greet", "morning", "afternoon", "evening", "namaste", "start"],
    "farewell":    ["bye", "goodbye", "exit", "quit", "later", "tata", "cya", "close", "end"],
    "thanks":      ["thanks", "thank", "appreciate", "grateful", "cheers", "helpful"],
    "about":       ["about", "overview", "history", "established", "founded", "information",
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
    "faculty":     ["faculty", "professor", "teacher", "staff", "lecturer", "instructor", "who", "teach"],
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
    # NOTE: "professor" keywords are no longer hardcoded here.
    # They are generated from ggu_data.json faculty names at load time
    # (see KnowledgeBase._build_faculty_index) so the chatbot stays in
    # sync with the JSON without any code change.
    "professor":   [],
}

COURSE_MAP = {
    "mca":    "MCA (Master of Computer Applications)",
    "bca":    "BCA (Bachelor of Computer Applications)",
    "msc_cs": "M.Sc. (Computer Science)",
    "bsc_cs": "B.Sc. (Computer Science)",
    "phd":    "Ph.D. (Computer Science / IT)",
}

_COURSE_KEYWORD_MAP = {"mca": "mca", "bca": "bca", "bsc": "bsc_cs", "b.sc": "bsc_cs",
                        "msc": "msc_cs", "m.sc": "msc_cs", "phd": "phd", "ph.d": "phd", "doctoral": "phd"}

_COMPARE_KEYS = ("duration", "intake", "eligibility", "fee_per_semester", "total_approx_fee", "fellowship")
_COMPARE_LABELS = {
    "duration": "Duration", "intake": "Intake", "eligibility": "Eligibility",
    "fee_per_semester": "Fee / Semester", "total_approx_fee": "Total Fee", "fellowship": "Fellowship",
}

_COURSE_ONLY_INTENTS = {
    "fees", "hostel", "admission", "faculty", "professor", "placement", "scholarship",
    "research", "exam", "location", "contact", "help", "compare", "facilities", "semester",
    "about", "courses",
}

_TITLE_WORDS = {"dr", "dr.", "mr", "mr.", "mrs", "mrs.", "prof", "prof.", "ms", "ms."}

SMALL_TALK = {
    "greeting": [
        "Namaste! \U0001F64F I am **CSITmitra**, your assistant for GGU CSIT Dept.\n\n"
        "Ask me about: courses \u00b7 faculty \u00b7 fees \u00b7 admission \u00b7 placement \u00b7 research",
        "Hello! I am **CSITmitra**, your GGU CSIT assistant.\n\nWhat would you like to know?",
    ],
    "farewell": [
        "Goodbye! Best of luck with your studies! \U0001F393 — GGU CSIT Dept.",
        "See you! Feel free to return anytime. \U0001F60A",
    ],
    "thanks": [
        "You're welcome! Any other questions about CSIT at GGU?",
        "Happy to help! Anything else you'd like to know?",
    ],
    "unknown": [
        "I'm not sure about that. Try asking about:\n\ncourses \u00b7 fees \u00b7 faculty \u00b7 admission \u00b7 placement \u00b7 scholarship \u00b7 research",
        "Could you rephrase? For example:\n\u2022 'Tell me about MCA'\n\u2022 'Who is Dr. Shrivas?'\n\u2022 'What are the fees?'",
    ],
}

HELP_TEXT = textwrap.dedent("""\
    \U0001F916 **GGU CSITmitra — Help**

    **Topics you can ask about:**
    \u2022 Courses & Programmes (BCA, MCA, M.Sc., B.Sc., Ph.D.)
    \u2022 Fee Structure & Hostel Fees
    \u2022 Faculty Profiles
    \u2022 Admission Process (CUET, SAMARTH Portal)
    \u2022 Placements & Recruiters
    \u2022 Scholarships & Financial Aid
    \u2022 Research & Publications
    \u2022 Exam Pattern & CGPA Grading
    \u2022 Campus Facilities
    \u2022 Contact & Location

    **Example queries:**
    \u2022 "Tell me about MCA"
    \u2022 "What are the fees for BCA?"
    \u2022 "Who is Dr. Shrivas?"
    \u2022 "MCA vs B.Sc. comparison"
    \u2022 "Ph.D. admission process"
    \u2022 "MCA syllabus"
    \u2022 "Placement highlights"
    \u2022 "Hostel fees"

    **Tips:**
    \u2022 Context is remembered per session — ask about MCA, then just type 'fees?' or 'syllabus'
    \u2022 Use a professor's last name: 'Tell me about Majhi'
    \u2022 Compare two programmes: 'MCA vs Ph.D'
""")


# ============================================================
#  2. KNOWLEDGE BASE  (loads JSON once, builds fast lookup indexes)
# ============================================================

class KnowledgeBase:
    """
    Wraps ggu_data.json and pre-builds indexes so lookups are O(1)
    instead of scanning lists at request time.
    """

    # Field-name variants we've seen in real-world scraped/edited JSON.
    # A getter tries each in order, so a spelling difference (e.g. the
    # British "specialisation") no longer silently blanks out a field.
    _DESIGNATION_KEYS = ("designation", "Designation", "position", "role", "title", "post")
    _SPECIALIZATION_KEYS = ("specialization", "specialisation", "Specialization",
                             "area_of_interest", "research_area", "field", "domain", "areas")

    def __init__(self, data: dict):
        self.data = data
        self.courses: dict = data.get("courses", {})
        self.faculty: list = data.get("faculty", [])
        self._faculty_by_key: dict = {}          # name-token -> faculty record
        self._professor_keywords: dict = {}       # keyword -> "professor" (merged into ALL_KEYWORDS)
        self._build_faculty_index()

    # -- construction -----------------------------------------------------

    def _build_faculty_index(self) -> None:
        """Index every faculty record by full name and by each surname/
        first-name token, so 'Tell me about Majhi' resolves in O(1) and
        new faculty in the JSON are searchable without touching the code."""
        for record in self.faculty:
            full_name = record.get("name", "")
            if not full_name:
                continue
            full_lower = self._normalize_name(full_name)
            self._faculty_by_key[full_lower] = record

            for token in full_lower.split():
                if token in _TITLE_WORDS or len(token) < 3:
                    continue
                self._faculty_by_key.setdefault(token, record)
                self._professor_keywords.setdefault(token, "professor")

            compact = full_lower.replace(" ", "")
            if len(compact) >= 3:
                self._faculty_by_key.setdefault(compact, record)

    @staticmethod
    def _normalize_name(text: str) -> str:
        return " ".join(text.lower().replace(".", " ").replace(",", " ").split())

    # -- generic fault-tolerant field access -------------------------------

    @staticmethod
    def _first_present(record: dict, keys: tuple, default=None):
        """Get the first non-empty value, tolerating key spelling/casing differences."""
        for key in keys:
            value = record.get(key)
            if value not in (None, "", [], {}):
                return KnowledgeBase._to_text(value)

        wanted = {key.lower().replace(" ", "_") for key in keys}
        for actual_key, value in record.items():
            normalized = str(actual_key).lower().replace(" ", "_")
            if normalized in wanted and value not in (None, "", [], {}):
                return KnowledgeBase._to_text(value)
        return default

    @staticmethod
    def _to_text(value) -> str:
        if isinstance(value, (list, tuple)):
            return ", ".join(KnowledgeBase._to_text(v) for v in value)
        if isinstance(value, dict):
            return " | ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    # -- generic MULTI-value field access ----------------------------------
    # One field "concept" (e.g. "email" or "phone") can show up in the JSON
    # in many shapes depending on who edited the data:
    #   "email": "a@x.com"                              -> single value
    #   "email_1": "a@x.com", "email_2": "b@x.com"       -> numbered siblings
    #   "phone": ["a", "b"]                              -> real list
    #   "phone": {"Registrar": "a", "Admission": "b"}    -> labeled dict
    #   "phone": "['a', 'b']"                            -> stringified list (data bug)
    #   "phone": "a, b; c"                               -> delimited string
    # collect_multi()/render_multi() handle every one of these the same
    # way, for ANY field name, so a new "phone_3" or "email_4" added to the
    # JSON later is picked up automatically with no code change.

    _MULTI_DELIM_RE = re.compile(r"\s*(?:,|;|\|)\s*")
    _NUMBERED_KEY_RE = re.compile(r"^([a-z]+)_?(\d+)$")

    @classmethod
    def _coerce_stringified_list(cls, value):
        """Defensive parse for a field that *should* be a list/dict but was
        saved as the str() of one, e.g. "['a', 'b']" instead of a real
        JSON array. Returns the parsed value, or the original unchanged."""
        if not isinstance(value, str):
            return value
        s = value.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, dict)):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return value

    @classmethod
    def _split_multi_string(cls, s: str):
        """Split "a, b; c" into ["a", "b", "c"]. Deliberately does NOT
        split on "/" so URLs (e.g. a website path) are left intact."""
        parts = [p.strip() for p in cls._MULTI_DELIM_RE.split(s) if p.strip()]
        return parts if len(parts) > 1 else [s]

    @classmethod
    def collect_multi(cls, record: dict, base_names: tuple) -> list:
        """
        Collect EVERY value for a field concept (e.g. all phone numbers,
        all emails) out of `record`, regardless of which of the shapes
        above was used. Returns an ordered, de-duplicated list of
        (label, value) pairs; label is "" when there was no natural one.
        """
        if not isinstance(record, dict):
            return []
        base_set = {b.lower() for b in base_names}
        results, seen = [], set()

        def emit(label, value):
            value = str(value).strip()
            if not value or value.lower() in ("n/a", "na", "none", "-"):
                return
            sig = (str(label).strip().lower(), value.lower())
            if sig in seen:
                return
            seen.add(sig)
            results.append((str(label).strip(), value))

        for raw_key, raw_val in record.items():
            norm_key = str(raw_key).lower().replace(" ", "_").replace("-", "_")
            m = cls._NUMBERED_KEY_RE.match(norm_key)
            base_part, num_part = (m.group(1), m.group(2)) if m else (norm_key, None)
            if base_part not in base_set:
                continue

            val = cls._coerce_stringified_list(raw_val)

            if isinstance(val, dict):
                for lbl, v in val.items():
                    emit(lbl, v)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, dict):
                        for lbl, v in item.items():
                            emit(lbl, v)
                    elif isinstance(item, str) and ":" in item:
                        lbl, _, v = item.partition(":")
                        emit(lbl, v)
                    else:
                        emit("", item)
            elif isinstance(val, str):
                parts = cls._split_multi_string(val)
                if len(parts) > 1:
                    for p in parts:
                        emit("", p)
                else:
                    label = f"{base_part.title()} {num_part}".strip() if num_part else ""
                    emit(label, val)
            elif val is not None:
                label = f"{base_part.title()} {num_part}".strip() if num_part else ""
                emit(label, val)

        return results

    @classmethod
    def render_multi(cls, record: dict, base_names: tuple, default_label: str) -> list:
        """Render collect_multi() results as '• **Label**: value' lines,
        falling back to a single 'N/A' line when nothing was found."""
        values = cls.collect_multi(record, base_names)
        if not values:
            return [f"\u2022 **{default_label}**: N/A"]
        return [f"\u2022 **{lbl or default_label}**: {val}" for lbl, val in values]

    def designation_of(self, record: dict) -> str:
        return self._first_present(record, self._DESIGNATION_KEYS, default="N/A")

    def specialization_of(self, record: dict):
        return self._first_present(record, self._SPECIALIZATION_KEYS, default=None)

    # -- lookups ------------------------------------------------------------

    def find_faculty(self, query: str):
        """O(1) average lookup by full name or any name token (surname etc.)."""
        if not query:
            return None
        query = self._normalize_name(query)
        if query in self._faculty_by_key:
            return self._faculty_by_key[query]

        for token in query.split():
            if token in self._faculty_by_key:
                return self._faculty_by_key[token]
        return None

    def professor_keywords(self) -> dict:
        return self._professor_keywords

    def resolve_course_name(self, keyword_or_name: str):
        """Accepts either a short keyword ('mca') or a full course name and
        returns the canonical course name key used in ggu_data.json."""
        if not keyword_or_name:
            return None
        if keyword_or_name in self.courses:
            return keyword_or_name
        mapped = COURSE_MAP.get(keyword_or_name)
        if mapped in self.courses:
            return mapped
        return None


# ============================================================
#  3. INTENT CLASSIFIER  (O(1) keyword lookup per token)
# ============================================================

class IntentClassifier:
    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge
        # Single flat map: keyword -> intent, built once. Looking a token
        # up here is O(1) average, versus the old approach of checking
        # "token in keyword_list" for every intent (O(total_keywords)/token).
        self.keyword_to_intent = {
            kw: intent for intent, kws in INTENTS.items() for kw in kws
        }
        self.keyword_to_intent.update(knowledge.professor_keywords())

    @staticmethod
    def preprocess(text: str) -> list:
        text_lower = text.lower()
        if USE_NLTK:
            tokens = word_tokenize(text_lower)
            tokens = [t for t in tokens if t not in string.punctuation]
            return [_lemmatizer.lemmatize(t) for t in tokens]
        return [t.strip(string.punctuation) for t in text_lower.split() if t.strip(string.punctuation)]

    @lru_cache(maxsize=2048)
    def _fuzzy_match(self, token: str, cutoff: float = 0.82):
        """Only called when an exact lookup misses, and cached, so repeated
        typos across a session cost one difflib pass instead of many."""
        matches = difflib.get_close_matches(token, self.keyword_to_intent.keys(), n=1, cutoff=cutoff)
        return self.keyword_to_intent[matches[0]] if matches else None

    def spell_hint(self, user_input: str) -> str:
        suggestions = set()
        for tok in self.preprocess(user_input):
            if len(tok) < 3:
                continue
            close = difflib.get_close_matches(tok, self.keyword_to_intent.keys(), n=2, cutoff=0.75)
            suggestions.update(c for c in close if c != tok)
        if suggestions:
            return "\U0001F4A1 Did you mean: " + " / ".join(sorted(suggestions)[:3]) + "?"
        return ""

    def detect(self, tokens: list) -> tuple:
        scores = Counter()
        matched_professor_tokens = []

        for token in tokens:
            intent = self.keyword_to_intent.get(token)          # O(1)
            if intent is None:
                intent = self._fuzzy_match(token)                # fallback only
                if intent:
                    scores[intent] += 1
            else:
                scores[intent] += 2
                if intent == "professor":
                    matched_professor_tokens.append(token)

        lower = " ".join(tokens)
        matched_course_key = next((v for k, v in _COURSE_KEYWORD_MAP.items() if k in lower), None)
        if matched_course_key:
            scores[matched_course_key] += 5

        want_syllabus = bool({"syllabus", "semester", "sem", "curriculum"} & set(tokens)) and matched_course_key is not None
        if {"vs", "versus", "compare", "comparison", "difference", "between"} & set(tokens):
            scores["compare"] += 6
        if matched_professor_tokens:
            scores["professor"] += 6

        if not scores:
            return "unknown", None

        best_intent = scores.most_common(1)[0][0]
        matched_course_name = COURSE_MAP.get(matched_course_key)
        sub = (matched_course_name + ":syllabus") if want_syllabus else (
            " ".join(matched_professor_tokens) or matched_course_name
        )
        return best_intent, sub


# ============================================================
#  4. SESSION CONTEXT + LOGGING
# ============================================================

class SessionContext:
    """Thin wrapper around Flask's per-user signed-cookie session."""

    @staticmethod
    def get() -> dict:
        if "ctx" not in flask.session:
            flask.session["ctx"] = {"last_intent": None, "last_course": None, "last_faculty": None}
        return flask.session["ctx"]

    @staticmethod
    def update(**kwargs) -> None:
        ctx = SessionContext.get()
        ctx.update(kwargs)
        flask.session["ctx"] = ctx
        flask.session.modified = True

    @staticmethod
    def history() -> list:
        if "history" not in flask.session:
            flask.session["history"] = []
        return flask.session["history"]


class ChatLogger:
    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)

    def _session_log_path(self) -> str:
        if "log_file" not in flask.session:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = secrets.token_hex(4)
            fname = f"chat_{ts}_{uid}.txt"
            log_path = os.path.join(self.logs_dir, fname)
            flask.session["log_file"] = log_path
            flask.session["log_file_name"] = fname
            flask.session.modified = True
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("=" * 60 + "\n")
                fh.write("  GGU CSITmitra — Chat Log\n")
                fh.write(f"  Session started : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
                fh.write("=" * 60 + "\n\n")
        return flask.session["log_file"]

    def log(self, user_msg: str, intent: str, bot_reply: str) -> None:
        ts = datetime.datetime.now()
        turn_num = len(SessionContext.history()) + 1

        h = SessionContext.history()
        h.append({"turn": turn_num, "time": ts.strftime("%H:%M:%S"),
                   "user": user_msg, "intent": intent, "bot": bot_reply[:400]})
        flask.session["history"] = h
        flask.session.modified = True

        try:
            log_path = self._session_log_path()
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"[{turn_num:03d}] {ts:%H:%M:%S}\n")
                fh.write(f"  User   : {user_msg}\n")
                fh.write(f"  Intent : {intent}\n")
                fh.write(f"  Bot    : {bot_reply}\n\n")
        except Exception as e:
            app.logger.warning(f"Log write failed: {e}")


# ============================================================
#  5. RESPONSE BUILDER  (dispatch table instead of if/elif chain)
# ============================================================

class ResponseBuilder:
    def __init__(self, knowledge: KnowledgeBase):
        self.k = knowledge
        self.handlers = {
            "greeting": self._small_talk, "farewell": self._small_talk, "thanks": self._small_talk,
            "about": self._about, "courses": self._courses, "semester": self._semester,
            "fees": self._fees, "hostel": self._hostel, "admission": self._admission,
            "faculty": self._faculty_list, "professor": self._professor,
            "facilities": self._facilities, "placement": self._placement,
            "scholarship": self._scholarship, "research": self._research, "exam": self._exam,
            "location": self._location, "contact": self._contact, "help": self._help,
            "compare": self._compare,
            **{k: self._course_detail for k in COURSE_MAP},   # mca/bca/msc_cs/bsc_cs/phd
        }

    def build(self, intent: str, sub) -> str:
        if not self.k.data:
            return "\u26A0\uFE0F Data file not found. Please ensure ggu_data.json is in the same folder as app.py."
        handler = self.handlers.get(intent)
        if handler is None:
            return random.choice(SMALL_TALK["unknown"])
        return handler(intent, sub)

    # -- small talk -----------------------------------------------------

    def _small_talk(self, intent, sub):
        return random.choice(SMALL_TALK[intent])

    # -- university / course info ----------------------------------------

    def _about(self, intent, sub):
        SessionContext.update(last_intent="about")
        d = self.k.data
        return (
            f"\U0001F3DB\uFE0F **{d.get('university', 'Guru Ghasidas Vishwavidyalaya')}**\n"
            f"{d.get('department', '')}\n\n"
            f"\u2022 **Location**: {d.get('location', '')}\n"
            f"\u2022 **Founded**: {d.get('established', '')}\n"
            f"\u2022 **NAAC**: {d.get('naac', '')}\n"
            f"\u2022 **Campus**: {d.get('campus_size', '')}\n"
            f"\u2022 **Website**: {d.get('website', '')}\n\n"
            f"\u2022 **About CSIT Department**: {d.get('CSIT', '')}"
        )

    def _courses(self, intent, sub):
        SessionContext.update(last_intent="courses")
        lines = ["\U0001F393 **Programmes offered by CSIT Dept., GGV:**\n"]
        for prog, info in self.k.courses.items():
            lines.append(
                f"**{prog}**\n"
                f"Duration: {info.get('duration', 'N/A')} | Seats: {info.get('intake', 'N/A')}\n"
                f"Fee/sem: {info.get('fee_per_semester', 'N/A')}\n"
            )
        lines.append("\U0001F4A1 Ask 'Tell me about MCA' or 'MCA syllabus' for full details.")
        return "\n".join(lines)

    def _resolve_course_key(self, intent, sub, ctx):
        want_syllabus_only = isinstance(sub, str) and sub.endswith(":syllabus")
        if want_syllabus_only:
            sub = sub.replace(":syllabus", "")
        course_key = self.k.resolve_course_name(intent) or self.k.resolve_course_name(sub)
        if not course_key and ctx.get("last_course") and intent not in _COURSE_ONLY_INTENTS:
            course_key = ctx["last_course"]
        if want_syllabus_only and not course_key:
            course_key = ctx.get("last_course")
        return course_key, want_syllabus_only

    def _course_detail(self, intent, sub):
        ctx = SessionContext.get()
        course_key, want_syllabus_only = self._resolve_course_key(intent, sub, ctx)
        if not course_key or course_key not in self.k.courses:
            return random.choice(SMALL_TALK["unknown"])

        SessionContext.update(last_course=course_key, last_intent="course_detail")
        c = self.k.courses[course_key]

        if want_syllabus_only:
            return self._render_syllabus(course_key, c)

        lines = [f"\U0001F4D8 **{course_key}**\n", f"\u2022 **Duration**: {c.get('duration', 'N/A')}"]
        if "intake" in c:
            lines.append(f"\u2022 **Intake**: {c['intake']}")
        lines.append(f"\u2022 **Eligibility**: {c.get('eligibility', 'N/A')}")
        if "exit_option" in c:
            lines.append(f"\u2022 **Exit Option**: {c['exit_option']}")
        lines.append(f"\u2022 **Fee/Semester**: {c.get('fee_per_semester', 'N/A')}")
        if "total_approx_fee" in c:
            lines.append(f"\u2022 **Total Fee**: {c['total_approx_fee']}")
        if "fellowship" in c:
            lines.append(f"\u2022 **Fellowship**: {c['fellowship']}")
        if "research_areas" in c:
            lines.append(f"\u2022 **Research Areas**: {', '.join(c['research_areas'])}")
        if "semesters" in c:
            lines.append("\n\U0001F4CB **Semester-wise Subjects:**")
            for sem, subjects in c["semesters"].items():
                lines.append(f"\n**{sem}:**")
                lines.extend(f"\u2022 {subj}" for subj in subjects)
        return "\n".join(lines)

    @staticmethod
    def _render_syllabus(course_key, course_data):
        if "semesters" not in course_data:
            return f"Semester details not available for {course_key}."
        lines = [f"\U0001F4CB **Semester-wise Subjects — {course_key}:**\n"]
        for sem, subjects in course_data["semesters"].items():
            lines.append(f"**{sem}:**")
            lines.extend(f"\u2022 {subj}" for subj in subjects)
            lines.append("")
        return "\n".join(lines)

    def _semester(self, intent, sub):
        ctx = SessionContext.get()
        course_key = ctx.get("last_course")
        if course_key and course_key in self.k.courses:
            return self._render_syllabus(course_key, self.k.courses[course_key])
        return "Please specify a course first. Example: 'MCA syllabus' or 'BCA semesters'"

    # -- fees / hostel / admission ----------------------------------------

    def _fees(self, intent, sub):
        SessionContext.update(last_intent="fees")
        d, fee = self.k.data, self.k.data.get("fees", {})
        ctx_course = SessionContext.get().get("last_course")

        if ctx_course and ctx_course in self.k.courses:
            c = self.k.courses[ctx_course]
            h = fee.get("hostel", {})
            lines = [
                f"\U0001F4B0 **Fees — {ctx_course}:**\n",
                f"\u2022 **Per Semester**: {c.get('fee_per_semester', 'N/A')}",
                f"\u2022 **Total**: {c.get('total_approx_fee', 'N/A')}",
            ]
            if "fellowship" in c:
                lines.append(f"\u2022 **Fellowship**: {c['fellowship']}")
            lines.append(f"\n\U0001F3E0 **Hostel**: Boys — {h.get('boys', 'N/A')} | Girls — {h.get('girls', 'N/A')}")
            lines.append(f"\n\u26A0\uFE0F {fee.get('note', '')}")
            return "\n".join(lines)

        lines = ["\U0001F4B0 **Fee Structure — CSIT Dept., GGV:**\n"]
        for prog, info in self.k.courses.items():
            lines.append(f"**{prog}**\n\u2022 Per Semester: {info.get('fee_per_semester', 'N/A')}"
                          f"\n\u2022 Total: {info.get('total_approx_fee', 'N/A')}\n")
        h = fee.get("hostel", {})
        lines.append(f"\U0001F3E0 **Hostel:**\n\u2022 Boys: {h.get('boys', 'N/A')}"
                      f"\n\u2022 Girls: {h.get('girls', 'N/A')}\n\u2022 Note: {h.get('note', '')}")
        if "other_charges" in fee:
            lines.append("\n\U0001F4CB **Other Charges:**")
            lines.extend(f"\u2022 {c.replace('_', ' ').title()}: {v}" for c, v in fee["other_charges"].items())
        if "scholarships_that_cover_fees" in fee:
            lines.append("\n\U0001F393 **Scholarships that Cover Fees:**")
            lines.extend(f"\u2022 {s}" for s in fee["scholarships_that_cover_fees"])
        lines.append(f"\n\u26A0\uFE0F {fee.get('note', '')}")
        return "\n".join(lines)

    def _hostel(self, intent, sub):
        h = self.k.data.get("fees", {}).get("hostel", {})
        return (
            "\U0001F3E0 **Hostel Information — GGV:**\n\n"
            f"\u2022 **Boys Hostel**: {h.get('boys', 'N/A')}\n"
            f"\u2022 **Girls Hostel**: {h.get('girls', 'N/A')}\n"
            f"\u2022 **Note**: {h.get('note', '')}\n\n"
            "**Facilities include:** Wi-Fi, common room, reading room, mess (hygienic food), 24/7 security.\n\n"
            "\U0001F4CC Apply separately via SAMARTH Portal at www.ggu.ac.in"
        )

    _ADM_KEY_MAP = {
        "BCA (Bachelor of Computer Applications)": "BCA",
        "B.Sc. (Computer Science)": "B.Sc. CS",
        "M.Sc. (Computer Science)": "M.Sc. CS",
        "MCA (Master of Computer Applications)": "MCA",
        "Ph.D. (Computer Science / IT)": "Ph.D.",
    }

    def _admission(self, intent, sub):
        SessionContext.update(last_intent="admission")
        adm = self.k.data.get("admission_process", {})

        def resolve(name):
            mapped = self._ADM_KEY_MAP.get(name, name)
            return mapped if mapped in adm else None

        ctx_course = (resolve(sub) if sub else None) or resolve(SessionContext.get().get("last_course", ""))
        if ctx_course:
            return (f"\U0001F4DD **Admission — {ctx_course}:**\n\n{adm[ctx_course]}\n\n"
                    "\U0001F4CC Apply online at: **www.ggu.ac.in** (SAMARTH Portal)")

        lines = ["\U0001F4DD **Admission Process — CSIT Dept., GGV:**\n"]
        for prog, info in adm.items():
            lines.append(f"**{prog}:**\n{info}\n")
        lines.append("\U0001F4CC Apply online at: **www.ggu.ac.in** (SAMARTH Portal)")
        return "\n".join(lines)

    # -- faculty ------------------------------------------------------------

    def _faculty_list(self, intent, sub):
        SessionContext.update(last_intent="faculty")
        lines = ["\U0001F468\u200D\U0001F3EB **Teaching Faculty — CSIT Dept., GGV:**\n"]
        for f in self.k.faculty:
            designation = self.k.designation_of(f)
            specialization = self.k.specialization_of(f) or "Specialization N/A"
            lines.append(
                f"\u2022 **{f.get('name', 'N/A')}**\n"
                f"  **Designation:** {designation}\n"
                f"  **Specialization:** {specialization}\n"
            )
        lines.append("\U0001F4A1 Ask 'Tell me about Dr. Babita Majhi' for a full profile.")
        return "\n".join(lines)

    def _professor(self, intent, sub):
        ctx = SessionContext.get()
        matched_f = self.k.find_faculty(sub) or (
            self.k.find_faculty(ctx.get("last_faculty")) if ctx.get("last_faculty") else None
        )
        if not matched_f:
            return ("I couldn't find that professor. Try using their last name.\n"
                     "Example: 'Tell me about Majhi' or 'Who is Dr. Shrivas?'\n"
                     "Type 'faculty' to see the full list.")

        SessionContext.update(last_faculty=matched_f["name"])
        f = matched_f
        lines = [f"\U0001F464 **{f['name']}**\n",
                 f"\u2022 **Designation**: {self.k.designation_of(f)}",
                 f"\u2022 **Qualification**: {f.get('qualification', 'N/A')}"]

        specialization = self.k.specialization_of(f)
        if specialization:
            lines.append(f"\u2022 **Specialization**: {specialization}")

        subjects = f.get("subjects_taught")
        if subjects:
            lines.append(f"\u2022 **Subjects Taught**: {', '.join(subjects)}")

        # Generic: picks up "email", "email_1"/"email_2", a list, a dict,
        # or a delimited string automatically — no hardcoded key names,
        # so a third or fourth email/phone added to the JSON just works.
        emails = self.k.collect_multi(f, ("email", "mail"))
        if emails:
            lines.extend(f"\u2022 **{lbl or 'Email'}**: {val}" for lbl, val in emails)

        phones = self.k.collect_multi(f, ("phone", "mobile", "contact_number"))
        if phones:
            lines.extend(f"\u2022 **{lbl or 'Phone'}**: {val}" for lbl, val in phones)
        if f.get("notable"):
            lines.append(f"\u2022 **Notable**: {f['notable']}")
        if f.get("google_scholar"):
            lines.append(f"\u2022 **Google Scholar**: {f['google_scholar']}")
        if f.get("orcid"):
            lines.append(f"\u2022 **ORCID**: {f['orcid']}")
        if f.get("joined"):
            lines.append(f"\u2022 **Joined GGV**: {f['joined']}")
        return "\n".join(lines)

    # -- facilities / placement / scholarship / research / exam -------------

    def _facilities(self, intent, sub):
        lines = ["\U0001F3EB **Campus & Departmental Facilities:**\n"]
        lines.extend(f"\u2022 {item}" for item in self.k.data.get("facilities", []))
        return "\n".join(lines)

    def _placement(self, intent, sub):
        p = self.k.data.get("placement", {})
        return (
            "\U0001F4BC **Placement Highlights — GGV CSIT:**\n\n"
            f"\u2022 **UG 4-yr Median Package**: {p.get('ug_4yr_median', 'N/A')}\n"
            f"\u2022 **PG 2-yr Median Package**: {p.get('pg_2yr_median', 'N/A')}\n"
            f"\u2022 **Highest Package**: {p.get('highest', 'N/A')}\n"
            f"\u2022 **UG Students Placed**: {p.get('students_placed_ug', 'N/A')}\n"
            f"\u2022 **PG Students Placed**: {p.get('students_placed_pg', 'N/A')}\n"
            f"\u2022 **Placement Rate**: {p.get('placement_rate', 'N/A')}\n"
            f"\u2022 **Top Recruiters**: {', '.join(p.get('top_recruiters', []))}\n\n"
            f"\U0001F4CC {p.get('placement_cell', '')}\n{p.get('note', '')}"
        )

    def _scholarship(self, intent, sub):
        lines = ["\U0001F3C5 **Scholarships & Financial Aid:**\n"]
        lines.extend(f"\u2022 {item}" for item in self.k.data.get("scholarship", []))
        lines.append("\n\U0001F4CC Apply via: scholarships.gov.in or through SAMARTH Portal at www.ggu.ac.in")
        return "\n".join(lines)

    def _research(self, intent, sub):
        r = self.k.data.get("research", {})
        lines = [f"\U0001F52C **Research @ CSIT, GGV:**\n\n{r.get('summary', '')}\n",
                 f"**Research Areas:** {', '.join(r.get('areas', []))}\n",
                 "**Recent Highlights:**"]
        lines.extend(f"\u2022 {h}" for h in r.get("recent_highlights", []))
        return "\n".join(lines)

    def _exam(self, intent, sub):
        return f"\U0001F4CB **Examination Pattern:**\n\n{self.k.data.get('exam_pattern', 'N/A')}"

    def _location(self, intent, sub):
        d = self.k.data
        return (
            f"\U0001F4CD **{d.get('university', '')}**\n{d.get('department', '')}\n{d.get('location', '')}\n\n"
            "\u2022 Located at Koni, approx. 5 km from Bilaspur city\n"
            "\u2022 River Arpa runs parallel to the campus\n"
            "\u2022 University buses connect Bilaspur city to Koni campus"
        )

    def _contact(self, intent, sub):
        """Render contact details generically: works no matter how many
        phones/emails the JSON has, or which shape they're stored in
        (single string, list, dict, numbered keys, or a delimited string)."""
        d = self.k.data
        lines = ["📞 **Contact — GGV CSIT Dept.:**"]
        lines.extend(self.k.render_multi(d, ("phone", "mobile", "contact_number"), "Phone"))
        lines.extend(self.k.render_multi(d, ("email", "mail"), "Email"))
        lines.extend(self.k.render_multi(d, ("website", "site", "url"), "Website"))
        return "\n".join(lines)

    def _help(self, intent, sub):
        return HELP_TEXT

    # -- compare --------------------------------------------------------

    def compare_courses(self, lower_text: str):
        found = []
        for kw, key in _COURSE_KEYWORD_MAP.items():
            if kw in lower_text:
                full = COURSE_MAP[key]
                if full not in found:
                    found.append(full)
        if len(found) < 2:
            return None

        rows = ["\u2696\uFE0F **Course Comparison**\n"]
        for key in _COMPARE_KEYS:
            row = f"**{_COMPARE_LABELS[key]}:**\n"
            for name in found:
                row += f"\u2022 {name}: {self.k.courses.get(name, {}).get(key, '—')}\n"
            rows.append(row)
        rows.append(f"\U0001F4A1 Ask 'Tell me about {found[0]}' for full details.")
        return "\n".join(rows)

    def _compare(self, intent, sub):
        hint_lower = (sub or "").lower()
        result = self.compare_courses(hint_lower)
        if result:
            return result
        last_course = SessionContext.get().get("last_course", "")
        if last_course:
            result = self.compare_courses(hint_lower + " " + last_course.lower()[:3])
            if result:
                return result
        return ("\u2696\uFE0F To compare courses, mention two programme names.\n\n"
                "Examples:\n\u2022 'MCA vs M.Sc'\n\u2022 'Compare BCA and MCA'\n\u2022 'Ph.D vs MCA difference'")


# ============================================================
#  6. CHATBOT FACADE  (ties classifier + knowledge + builder together)
# ============================================================

class ChatBot:
    _FEEDBACK_POSITIVE = ("good bot", "great", "awesome", "perfect", "well done", "nice work")
    _FEEDBACK_NEGATIVE = ("wrong", "incorrect", "not helpful", "useless")
    _COMPARE_TRIGGERS = ("vs", "versus", "compare", "difference between")

    def __init__(self, knowledge: KnowledgeBase, logger: ChatLogger):
        self.knowledge = knowledge
        self.classifier = IntentClassifier(knowledge)
        self.builder = ResponseBuilder(knowledge)
        self.logger = logger

    def respond(self, user_message: str) -> str:
        msg = user_message.strip()
        low = msg.lower()

        if any(kw in low for kw in self._FEEDBACK_POSITIVE):
            return "\U0001F60A Thank you for the kind words! Anything else I can help with?"
        if any(kw in low for kw in self._FEEDBACK_NEGATIVE):
            return "\U0001F614 Sorry about that! Please rephrase your question or type 'help' to see example queries."

        if any(kw in low for kw in self._COMPARE_TRIGGERS):
            result = self.builder.compare_courses(low)
            if result:
                self.logger.log(msg, "compare", result)
                return result

        tokens = self.classifier.preprocess(msg)
        intent, sub = self.classifier.detect(tokens)
        response = self.builder.build(intent, sub)

        if intent == "unknown":
            hint = self.classifier.spell_hint(msg)
            if hint:
                response = f"{response}\n\n{hint}"

        self.logger.log(msg, intent, response)
        return response


# ============================================================
#  7. FLASK APP  (thin routes only — all logic lives above)
# ============================================================

def load_data() -> dict:
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggu_data.json")
    if not os.path.exists(data_file):
        return {}
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


app = flask.Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
CORS(app, supports_credentials=True)

DATA = load_data()                                    # read once at startup
KNOWLEDGE = KnowledgeBase(DATA)
LOGS_DIR = os.environ.get("LOGS_DIR", os.path.join("/tmp", "csitmitra_logs"))
LOGGER = ChatLogger(LOGS_DIR)
BOT = ChatBot(KNOWLEDGE, LOGGER)


@app.route("/")
def home():
    return flask.render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = flask.request.get_json()
        if not data:
            return flask.jsonify({"error": "Invalid JSON"}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return flask.jsonify({"response": "Please enter a message!",
                             "timestamp": datetime.datetime.now().strftime("%I:%M %p")})

        bot_response = BOT.respond(user_message)
        return flask.jsonify({"response": bot_response,
                         "timestamp": datetime.datetime.now().strftime("%I:%M %p")})

    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return flask.jsonify({"response": "Sorry, something went wrong. Please try again!",
                         "timestamp": datetime.datetime.now().strftime("%I:%M %p")}), 500


@app.route("/history", methods=["GET"])
def history():
    return flask.jsonify({"history": SessionContext.history()})


@app.route("/reset", methods=["POST"])
def reset():
    flask.session.clear()
    return flask.jsonify({"status": "Session reset successfully."})


@app.route("/logs", methods=["GET"])
def list_logs():
    try:
        files = sorted((f for f in os.listdir(LOGS_DIR) if f.endswith(".txt")), reverse=True)
        rows = ""
        for fname in files:
            fpath = os.path.join(LOGS_DIR, fname)
            size_kb = os.path.getsize(fpath) / 1024
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
            rows += (f"<tr><td>{fname}</td><td>{mtime}</td><td>{size_kb:.1f} KB</td>"
                     f"<td><a href='/logs/view/{fname}'>View</a> &nbsp; "
                     f"<a href='/logs/download/{fname}'>Download</a></td></tr>\n")
        html = f"""<!DOCTYPE html>
<html><head><title>CSITmitra — Chat Logs</title>
<style>
  body {{ font-family: monospace; padding: 2rem; background:#1a1a2e; color:#eee; }}
  h1 {{ color:#00d4ff; }} table {{ border-collapse:collapse; width:100%; }}
  th,td {{ padding:.5rem 1rem; border:1px solid #444; text-align:left; }}
  th {{ background:#0f3460; }} a {{ color:#00d4ff; }}
  .empty {{ color:#888; margin-top:2rem; }}
</style></head><body>
<h1>\U0001F4CB CSITmitra — Chat Logs</h1>
<p>{len(files)} log file(s) saved in <code>logs/</code></p>
{"<table><tr><th>File</th><th>Last Modified</th><th>Size</th><th>Actions</th></tr>" + rows + "</table>" if files else "<p class='empty'>No logs yet. Chats will appear here.</p>"}
</body></html>"""
        return html
    except Exception as e:
        return f"Error listing logs: {e}", 500


@app.route("/logs/view/<path:filename>", methods=["GET"])
def view_log(filename):
    if ".." in filename or "/" in filename:
        return "Invalid filename", 400
    fpath = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(fpath):
        return "Log file not found", 404
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    html = f"""<!DOCTYPE html>
<html><head><title>{filename}</title>
<style>
  body {{ font-family: monospace; padding: 2rem; background:#1a1a2e; color:#eee; }}
  pre {{ white-space:pre-wrap; word-break:break-word; background:#0f3460; padding:1rem; border-radius:8px; }}
  a {{ color:#00d4ff; }}
</style></head><body>
<a href="/logs">\u2190 Back to logs</a> &nbsp; <a href="/logs/download/{filename}">\u2B07 Download</a>
<h2>{filename}</h2>
<pre>{content}</pre>
</body></html>"""
    return html


@app.route("/logs/download/<path:filename>", methods=["GET"])
def download_log(filename):
    if ".." in filename or "/" in filename:
        return "Invalid filename", 400
    return flask.send_from_directory(LOGS_DIR, filename, as_attachment=True)


@app.route("/health", methods=["GET"])
def health():
    log_count = len([f for f in os.listdir(LOGS_DIR) if f.endswith(".txt")]) if os.path.exists(LOGS_DIR) else 0
    return flask.jsonify({
        "status": "ok",
        "nltk": USE_NLTK,
        "data_loaded": bool(DATA),
        "courses": len(DATA.get("courses", {})),
        "faculty": len(DATA.get("faculty", [])),
        "logs_saved": log_count,
        "logs_dir": LOGS_DIR,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
