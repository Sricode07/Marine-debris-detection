"""
AI-Driven Marine Debris Detection
Upload an image, find debris with YOLO (whole image + close-up tiles),
and generate an environmental report with Gemini.
"""

import html
import io
import os
from collections import Counter

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from debris_utils import (
    PALETTE,
    detect_debris,
    draw_detections,
    make_ocean_background_b64,
)

# The report module needs the Gemini library. If that import ever fails,
# the app still works and uses a plain built-in summary instead.
try:
    from detect_and_report import generate_report
except Exception:
    generate_report = None


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI-Driven Marine Debris Detection",
    page_icon="🌊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "marine_debris_v3", "weights", "best.pt"
)
MAX_FILE_MB = 15
MAX_SIDE_PX = 3200

# Classes that should be reported together. Left = name the model uses
# (lower case), right = the name shown in the app. Add or remove lines freely.
GROUPS = {
    "plastic bag": "plastic",
    "plastic bottle": "plastic",
    "plastic container": "plastic",
    "plastic waste": "plastic",
    "plastic": "plastic",
}


def build_groups(names):
    """Fold model classes into groups.
    Returns ({old_id: new_id}, {new_id: display_name})."""
    ids, display, mapping = {}, {}, {}
    for cid, name in sorted(names.items()):
        group = GROUPS.get(str(name).lower().strip(), name)
        if group not in ids:
            ids[group] = len(ids)
            display[ids[group]] = group
        mapping[cid] = ids[group]
    return mapping, display


# ---------------------------------------------------------
# LOOK AND FEEL
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_background():
    try:
        return make_ocean_background_b64()
    except Exception:
        return None


_bg = get_background()
BG_CSS = (
    f'url("data:image/jpeg;base64,{_bg}")'
    if _bg
    else "linear-gradient(180deg,#2a3f7a 0%,#f6905f 46%,#1e78b8 48%,#0a5a96 100%)"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Quicksand:wght@400;500;600;700&display=swap');

:root {
    --navy: 12, 34, 66;
    --foam: #EAF6FF;
    --mist: #C9E2F2;
    --amber: #FFB46B;
    --coral: #FF8A65;
}

/* ---------- page background: blurred sunset sea ---------- */
.stApp {
    background-image: linear-gradient(rgba(8,24,52,.08), rgba(8,24,52,.20)), __BG__;
    background-size: cover, cover;
    background-position: center, center;
    background-attachment: fixed, fixed;
    background-repeat: no-repeat, no-repeat;
}
[data-testid="stHeader"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: transparent !important; }
footer { visibility: hidden; }

.block-container { max-width: 1500px; padding: 1.4rem 2rem 3rem; }

/* ---------- type ---------- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp small, .stApp button,
.stApp input, .stApp textarea,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    font-family: 'Quicksand', 'Segoe UI', sans-serif !important;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp small { color: var(--foam); }
[data-testid="stCaptionContainer"] p { color: var(--mist); }

h1, h2, h3, h4, .display {
    font-family: 'Fraunces', Georgia, 'Times New Roman', serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    color: #FFFFFF !important;
}
h3 { font-size: 1.45rem !important; }

/* ---------- glass panels ---------- */
.glass, [class*="st-key-glass"] {
    background: rgba(var(--navy), .60);
    -webkit-backdrop-filter: blur(14px) saturate(1.25);
    backdrop-filter: blur(14px) saturate(1.25);
    border: 1px solid rgba(255,255,255,.17);
    border-radius: 18px;
    padding: 1.15rem 1.35rem;
    box-shadow: 0 10px 34px rgba(4,16,40,.38);
}
.glass { margin-bottom: 1.1rem; }
[class*="st-key-glass"] { margin-bottom: .9rem; }

.hero { text-align: center; padding: 1.7rem 1.5rem 1.5rem; margin-bottom: 1.4rem; }
.hero .display { font-size: 2.9rem; line-height: 1.1; margin: 0 0 .55rem; }
.hero .sub { font-size: 1.1rem; max-width: 46rem; margin: 0 auto; color: var(--mist); line-height: 1.5; }

.panel-title {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 700; font-size: 1.2rem; color: #fff;
    margin: 0 0 .7rem;
}
.glass p, .glass li { line-height: 1.55; font-size: .98rem; }
.glass p { margin: 0 0 .6rem; }
.glass .fine { color: var(--mist); font-size: .9rem; }

.steps { list-style: none; padding: 0; margin: 0; counter-reset: step; }
.steps li { counter-increment: step; position: relative; padding-left: 2.3rem; margin-bottom: .95rem; }
.steps li::before {
    content: counter(step);
    position: absolute; left: 0; top: .05rem;
    width: 1.65rem; height: 1.65rem; border-radius: 50%;
    background: linear-gradient(135deg, var(--amber), var(--coral));
    color: #10243d; font-weight: 700; font-size: .85rem;
    display: flex; align-items: center; justify-content: center;
}
.steps b { display: block; color: #fff; }

.tips { padding-left: 1.1rem; margin: 0; }
.tips li { margin-bottom: .4rem; }

.chip {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .28rem .8rem; margin: .18rem .12rem;
    border-radius: 999px; font-size: .92rem;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.24);
}
.dot { width: .7rem; height: .7rem; border-radius: 50%; display: inline-block; }

.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: .2rem 0 .9rem; }
.tile { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
        border-radius: 14px; padding: .8rem 1rem; }
.tile .num { font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: 1.9rem; color: #fff; line-height: 1.1; }
.tile .lbl { color: var(--mist); font-size: .9rem; }
.tile .num.small { font-size: 1.15rem; padding-top: .45rem; }

/* ---------- controls ---------- */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, var(--amber), var(--coral));
    border: 0; border-radius: 12px; padding: .6rem 1.6rem;
    box-shadow: 0 5px 16px rgba(255,138,101,.35);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 8px 20px rgba(255,138,101,.45); border: 0;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid #fff; outline-offset: 2px;
}
.stButton > button p, .stDownloadButton > button p {
    color: #10243d !important; font-weight: 700; font-size: 1rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.07);
    border: 1.5px dashed rgba(255,214,160,.75);
    border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.38); border-radius: 10px;
}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzoneInstructions"] *, [data-testid="stFileUploaderFile"] * {
    color: var(--foam) !important;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.18); border-radius: 12px;
}
[data-testid="stAlert"] {
    background: rgba(var(--navy), .78) !important;
    border: 1px solid rgba(255,255,255,.2); border-radius: 12px;
}
[data-testid="stImage"] img { border-radius: 14px; box-shadow: 0 8px 26px rgba(4,16,40,.4); }
table { color: var(--foam); }

@media (max-width: 900px) {
    .hero .display { font-size: 2rem; }
    .block-container { padding: 1rem; }
    .tiles { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
""".replace("__BG__", BG_CSS)

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------

def _html(s):
    """Collapse HTML to one line so Markdown never mistakes it for code."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def glass(name):
    """A frosted-glass container (older Streamlit versions get a plain border)."""
    try:
        return st.container(key=f"glass_{name}")
    except TypeError:
        return st.container(border=True)


def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, width="stretch")


def dot(class_id):
    r, g, b = PALETTE[class_id % len(PALETTE)]
    return f'<span class="dot" style="background:rgb({r},{g},{b})"></span>'


def basic_report(counts):
    """Used only if the report module itself can't be loaded."""
    if not counts:
        return (
            "### Environmental Report\n\nNo debris was detected. The detector can "
            "miss small or hidden items, so a manual check is still worthwhile."
        )
    items = ", ".join(f"{n} × {k}" for k, n in counts.items())
    return (
        "### Environmental Report\n\n"
        f"**Detected:** {items}.\n\n"
        "Debris like this can be swallowed or tangle around marine animals. "
        "Collect it safely and keep monitoring the area."
    )


# ---------------------------------------------------------
# MODEL (loaded once, with an error boundary)
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from ultralytics import YOLO

        if not os.path.exists(WEIGHTS_PATH):
            return None, (
                "The trained model file was not found at "
                f"`{WEIGHTS_PATH}`. Put `best.pt` in that folder and refresh the page."
            )
        return YOLO(WEIGHTS_PATH), None
    except Exception as e:
        return None, f"The detection model could not be loaded: {e}"


with st.spinner("Loading the detection model..."):
    model, model_error = load_model()

class_names = dict(model.names) if model is not None else {}
class_map, display_names = build_groups(class_names)


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    _html("""
    <div class="glass hero">
        <div class="display">AI-Driven Marine Debris Detection</div>
        <div class="sub">Upload a photo of the water or shoreline. The model finds the
        litter, counts each kind, and writes a short environmental report.</div>
    </div>
    """),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# THREE-COLUMN LAYOUT
# ---------------------------------------------------------

left, center, right = st.columns([1, 2.5, 1], gap="large")


# ---- left: how it works -----------------------------------
with left:
    st.markdown(
        _html("""
        <div class="glass">
            <div class="panel-title">How it works</div>
            <ol class="steps">
                <li><b>Upload a photo</b>A JPG or PNG taken from the water, a beach, or a drone.</li>
                <li><b>Run the scan</b>The model checks the whole picture, then close-up sections so small pieces aren't missed.</li>
                <li><b>Read the results</b>See every item marked on the photo, a count for each kind, and a plain-language report.</li>
            </ol>
        </div>
        <div class="glass">
            <div class="panel-title">For better results</div>
            <ul class="tips">
                <li>Use sharp, well-lit photos.</li>
                <li>Keep Deep scan on for wide shots with many small items.</li>
                <li>If it finds too little, lower the minimum confidence.</li>
                <li>If it marks things that aren't debris, raise it.</li>
            </ul>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ---- right: context ---------------------------------------
with right:
    if display_names:
        chips = "".join(
            f'<span class="chip">{dot(cid)}{html.escape(str(name))}</span>'
            for cid, name in sorted(display_names.items())
        )
        grouped_note = (
            "Bags, bottles, containers and other plastic items are grouped as plastic. "
            if len(display_names) < len(class_names) else ""
        )
        classes_html = (
            f"{chips}"
            f'<p class="fine" style="margin-top:.6rem">{grouped_note}Anything outside '
            "this list can't be named, even if it is debris.</p>"
        )
    else:
        classes_html = '<p class="fine">Available once the model has loaded.</p>'

    st.markdown(
        _html(f"""
        <div class="glass">
            <div class="panel-title">Why it matters</div>
            <p>Debris in the water can be swallowed by wildlife, can tangle around
            animals, and breaks into smaller pieces that are harder to remove.</p>
            <p>Knowing what is there, and how much, is the first step in cleaning it up.</p>
        </div>
        <div class="glass">
            <div class="panel-title">What this model can name</div>
            {classes_html}
        </div>
        <div class="glass">
            <div class="panel-title">Good to know</div>
            <p class="fine">Results are AI estimates. Check anything important by eye
            before acting on it.</p>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# CENTER: upload, analyse, results
# ---------------------------------------------------------

with center:

    if model_error:
        st.error(model_error)
        st.stop()

    with glass("upload"):
        uploaded_file = st.file_uploader(
            f"Upload an image (JPG or PNG, up to {MAX_FILE_MB} MB)",
            type=["jpg", "jpeg", "png"],
        )

        with st.expander("Detection settings"):
            conf = st.slider(
                "Minimum confidence",
                min_value=0.02, max_value=0.60, value=0.10, step=0.01,
                help="Lower finds more items but may mark things that aren't "
                     "debris. Higher is stricter.",
            )
            deep_scan = st.checkbox(
                "Deep scan (slower, finds small items)", value=True,
                help="Also checks overlapping close-up sections of the photo.",
            )
            merge_overlaps = st.checkbox(
                "Merge overlapping boxes", value=True,
                help="Keeps only the best box when several land on the same object.",
            )

    if not uploaded_file:
        st.info("Upload a photo above to get started.")
        st.stop()

    # ---- validate the upload ----
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        st.error(
            f"That file is {size_mb:.1f} MB. The limit is {MAX_FILE_MB} MB. "
            "Upload a smaller or compressed copy."
        )
        st.stop()

    try:
        image = Image.open(uploaded_file)
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max(image.size) > MAX_SIDE_PX:
            image.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
    except UnidentifiedImageError:
        st.error("That file isn't a readable image. Upload a JPG or PNG.")
        st.stop()
    except Exception as e:
        st.error(f"That image couldn't be opened ({e}). Try saving it again as a JPG or PNG.")
        st.stop()

    file_key = f"{uploaded_file.name}:{uploaded_file.size}"
    analysis = st.session_state.get("analysis")
    has_result = bool(analysis and analysis["key"] == file_key)

    # show the original only until there are results to show
    if not has_result:
        show_image(image, caption="Uploaded image")

    analyze = st.button("Analyze image")

    # ---- run the analysis ----
    if analyze:
        bar = st.progress(0.0, text="Starting...")

        def on_progress(frac, msg):
            bar.progress(frac, text=msg)

        try:
            dets = detect_debris(
                model, image,
                conf=conf,
                deep_scan=deep_scan,
                merge_overlaps=merge_overlaps,
                progress_cb=on_progress,
                class_map=class_map,
            )
            annotated = draw_detections(image, dets, display_names)
            counts = Counter(display_names.get(d[5], str(d[5])) for d in dets)
        except Exception as e:
            bar.empty()
            st.error(
                "The scan failed on this image. Try again, switch off Deep scan, "
                "or use a different photo."
            )
            with st.expander("Technical details"):
                st.code(str(e))
            st.stop()

        bar.empty()

        with st.spinner("Writing the environmental report..."):
            try:
                report = generate_report(counts) if generate_report else basic_report(counts)
            except Exception:
                report = basic_report(counts)

        st.session_state["analysis"] = {
            "key": file_key,
            "annotated": annotated,
            "dets": dets,
            "counts": counts,
            "report": report,
        }
        analysis = st.session_state["analysis"]
        has_result = True

    # ---- show results ----
    if has_result:
        counts = analysis["counts"]
        dets = analysis["dets"]

        show_image(analysis["annotated"], caption="Detected debris")

        buf = io.BytesIO()
        analysis["annotated"].save(buf, format="PNG")
        st.download_button(
            "Download annotated image",
            data=buf.getvalue(),
            file_name="detected_debris.png",
            mime="image/png",
        )

        with glass("detections"):
            st.subheader("Detections")

            if counts:
                top_name, top_n = counts.most_common(1)[0]
                st.markdown(
                    _html(f"""
                    <div class="tiles">
                        <div class="tile"><div class="num">{len(dets)}</div><div class="lbl">items found</div></div>
                        <div class="tile"><div class="num">{len(counts)}</div><div class="lbl">kinds of debris</div></div>
                        <div class="tile"><div class="num small">{html.escape(top_name)}</div><div class="lbl">most common ({top_n})</div></div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

                name_to_id = {v: k for k, v in display_names.items()}
                st.markdown(
                    "".join(
                        f'<span class="chip">{dot(name_to_id.get(n, 0))}'
                        f"{html.escape(n)} × {c}</span>"
                        for n, c in counts.most_common()
                    ),
                    unsafe_allow_html=True,
                )

                with st.expander("See every detection"):
                    rows = ["| Item | Confidence |", "|---|---|"]
                    for d in dets[:60]:
                        rows.append(f"| {display_names.get(d[5], d[5])} | {d[4]:.2f} |")
                    if len(dets) > 60:
                        rows.append(f"| … and {len(dets) - 60} more | |")
                    st.markdown("\n".join(rows))
            else:
                st.info(
                    "No debris was found. Try lowering the minimum confidence "
                    "or turning on Deep scan, then analyze again."
                )

        with glass("report"):
            st.subheader("Environmental report")
            st.markdown(analysis["report"])
