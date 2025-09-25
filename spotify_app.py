#!/usr/bin/env python3
# spotify_app.py — Compare two models across six families with Low/Medium/High complexity.
from __future__ import annotations

import json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(page_title="Compare Models: Spotify Popularity", page_icon="🎧", layout="centered")

# Winner decoration: "crown" or "flames"
WINNER_BADGE = "flames"   # options: "crown" | "flames"
BADGE_EMOJI = "👑" if WINNER_BADGE == "crown" else "🔥"

# -------------------- Paths & Assets --------------------
APP_DIR = Path(__file__).parent
ARTIFACT_DIR = Path(os.environ.get("SPOTIFY_ARTIFACT_DIR", APP_DIR / "artifacts" / "spotify_v3")).resolve()
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

ASSETS_DIR = APP_DIR / "assets"
FAMILY_DIR = ASSETS_DIR / "family"
PRESET_DIR = ASSETS_DIR / "preset"
HERO_DIR   = ASSETS_DIR / "hero"

CLASS_NAMES = ["Low", "Medium", "High"]

import streamlit as st

# --- CSS override for bigger icons/emojis ---
st.markdown(
    """
    <style>
    /* Enlarge emojis in markdown */
    .stMarkdown p {
        font-size: 1.5rem;
    }

    /* Enlarge icons inside metrics */
    [data-testid="stMetricValue"] svg {
        width: 2em !important;
        height: 2em !important;
    }

    /* Enlarge icons in buttons */
    .stButton button svg {
        width: 1.5em !important;
        height: 1.5em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Example usage ---
st.write("🚀 This emoji should look bigger now")
st.metric("Downloads", "1234", "📈")
st.button("Click me 🚀")


IMAGE_MAP = {
    "family": {
        "Logistic regression": str(FAMILY_DIR / "logreg.png"),
        "Decision tree":       str(FAMILY_DIR / "tree.png"),
        "Neural network":      str(FAMILY_DIR / "nn.png"),      # or mlp.png if you prefer
        "Random forest":       str(FAMILY_DIR / "rf.png"),
        "XGBoost":             str(FAMILY_DIR / "xgb.png"),
        "K-Means":             str(FAMILY_DIR / "kmeans.png"),
    },
    "preset": {
        "Low":    str(PRESET_DIR / "low.png"),
        "Medium": str(PRESET_DIR / "medium.png"),
        "High":   str(PRESET_DIR / "high.png"),
    }
}

# -------------------- Styling --------------------
st.markdown("""
<style>
  /* Wider main area, comfy top padding */
  [data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1150px; padding-top: 3rem !important;
  }
  .hero-h1 { font-size: 46px; font-weight: 800; line-height: 1.08; margin: 0 0 10px; }
  .hero-sub { font-size: 20px; font-weight: 600; opacity: 0.95; margin-bottom: 18px; }
  .lined { border-top: 1px dashed #ddd; margin: 12px 0 18px; }
  .edu { background:#0b13241a; border:1px solid #0b132433; border-radius:12px; padding:10px 14px; }

  /* Big, simple copy for the data page */
  .jumbo { font-size: 34px; font-weight: 800; line-height: 1.15; margin: 0 0 6px; }
  .big { font-size: 22px; line-height: 1.35; margin: 4px 0; }
  .big-list { font-size: 22px; line-height: 1.5; margin: 4px 0 0 0; padding-left: 1rem; }

  /* Normalized card image sizing */
  .family-card img, .size-card img {
    width: 100% !important; height: 160px !important;
    object-fit: contain; background: transparent; display:block;
  }
  .card-emoji { font-size: 128px; text-align:center; line-height:1.1; }

  /* Winner visuals */
  .winner-chip {
    display:inline-block; padding:2px 10px; border-radius:999px;
    background:#fef3c7; color:#92400e; border:1px solid #f59e0b;
    font-weight:700; font-size:12px; margin-bottom:6px;
  }
  .winner-badge { font-size:48px; line-height:1; text-align:center; margin:2px 0 8px 0; }
  .winner-box {
    border:2px solid #f59e0b; border-radius:14px; padding:10px 12px;
    background:linear-gradient(180deg,#fff7ed 0%, #fffbeb 100%);
    margin-bottom: 8px;
  }

  /* 🔥 Flame variant (used when WINNER_BADGE == "flames") */
  .winner-box.flames {
    border-color:#fb923c;
    background:linear-gradient(180deg,#fff7ed 0%, #fff1e6 100%);
    box-shadow: 0 0 12px rgba(251,146,60,0.35), 0 0 26px rgba(239,68,68,0.22);
    animation: flameGlow 1.8s ease-in-out infinite;
  }
  @keyframes flameGlow {
    0%,100% { box-shadow: 0 0 12px rgba(251,146,60,0.35), 0 0 26px rgba(239,68,68,0.22); }
    50%     { box-shadow: 0 0 20px rgba(251,146,60,0.55), 0 0 40px rgba(239,68,68,0.35); }
  }

  .tie-chip {
    display:inline-block; padding:2px 10px; border-radius:999px;
    background:#e2e8f0; color:#334155; border:1px solid #cbd5e1;
    font-weight:700; font-size:12px; margin-bottom:6px;
  }
</style>
""", unsafe_allow_html=True)

# -------------------- Utils --------------------
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        st.error(f"Manifest not found at {MANIFEST_PATH}. Run your training script to create artifacts.")
        st.stop()
    return json.loads(MANIFEST_PATH.read_text())

def hero_image_path() -> Optional[Path]:
    for name in ("landing", "hero", "cover"):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = HERO_DIR / f"{name}{ext}"
            if p.exists():
                return p
    return None

def _img_for(group: str, label: str) -> Optional[Path]:
    mapped = IMAGE_MAP.get(group, {}).get(label)
    if mapped and Path(mapped).exists():
        return Path(mapped)
    base = FAMILY_DIR if group == "family" else PRESET_DIR
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = base / f"{label.lower()}{ext}"
        if p.exists():
            return p
    return None

def draw_card(group: str, label: str, sub: Optional[str], key: str, emoji="🎛️") -> bool:
    img = _img_for(group, label)
    wrapper_cls = "family-card" if group == "family" else "size-card"

    with st.container():
        if img:
            st.markdown(f"<div class='{wrapper_cls}'>", unsafe_allow_html=True)
            st.image(str(img), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card-emoji'>{emoji}</div>", unsafe_allow_html=True)
        st.markdown(f"**{label}**")
        if sub:
            st.caption(sub)
        return st.button("Choose", key=key, use_container_width=True)

def grid_cards(group: str, items: List[Tuple[str, str]], key_prefix: str, on_choose, cols: int = 3):
    cols_list = st.columns(cols)
    for i, (label, sub) in enumerate(items):
        with cols_list[i % cols]:
            if draw_card(group, label, sub, key=f"{key_prefix}_{i}"):
                on_choose(label)
                rerun()

def fam_key(label: str) -> str:
    return {
        "Logistic regression": "logreg",
        "Decision tree": "tree",
        "Neural network": "mlp",
        "Random forest": "rf",
        "XGBoost": "xgb",
        "K-Means": "kmeans",
    }[label]

def friendly_name(fam: str, tier: str) -> str:
    names = {
        "logreg": "Logistic regression",
        "tree":   "Decision tree",
        "mlp":    "Neural network",
        "rf":     "Random forest",
        "xgb":    "XGBoost",
        "kmeans": "K-Means",
    }
    return f"{names.get(fam, fam)} — {tier}"

def tier_reason(fam: str, tier: str) -> str:
    if fam == "logreg":
        m = {"Low":"Top 10 features — simple",
             "Medium":"L1 selects subset — balanced",
             "High":"All features + interactions — flexible"}
    elif fam == "tree":
        m = {"Low":"Heavy pruning — stable",
             "Medium":"CV-pruned — balanced",
             "High":"Unpruned — flexible"}
    elif fam == "mlp":
        m = {"Low":"Shallow — fast",
             "Medium":"Tuned best — balanced",
             "High":"Deeper — more capacity"}
    elif fam == "rf":
        m = {"Low":"Fewer trees, shallow",
             "Medium":"Tuned depth/trees",
             "High":"Many trees, deep"}
    elif fam == "xgb":
        m = {"Low":"Shallow & few trees",
             "Medium":"Tuned depth/trees/η",
             "High":"Deep & many trees"}
    else:  # kmeans
        m = {"Low":"3 clusters",
             "Medium":"6 clusters",
             "High":"12 clusters"}
    return m[tier]

def load_metrics_for(manifest: Dict[str,Any], fam: str, tier: str) -> Dict[str,Any]:
    task_dir = Path(manifest["tasks"]["pop3"]["path"])
    model_key = f"{fam}_{tier.lower()}"
    mpath = task_dir / model_key / "metrics.json"
    if not mpath.exists():
        return {}
    return json.loads(mpath.read_text())

def _winner_of(mA: dict, mB: dict) -> str:
    aA, aB = mA.get("accuracy", 0.0), mB.get("accuracy", 0.0)
    if abs(aA - aB) > 1e-9:
        return "A" if aA > aB else "B"
    fA, fB = mA.get("macro_f1", 0.0), mB.get("macro_f1", 0.0)
    if abs(fA - fB) > 1e-9:
        return "A" if fA > fB else "B"
    return "tie"

# -------------------- State & Steps --------------------
steps = [
    "landing", "data",
    "fam_A", "tier_A",
    "fam_B", "tier_B",
    "compare"
]
if "step" not in st.session_state:
    st.session_state.step = 0

st.session_state.setdefault("fam_A", None)
st.session_state.setdefault("tier_A", None)
st.session_state.setdefault("fam_B", None)
st.session_state.setdefault("tier_B", None)

def goto(name: str):
    st.session_state.step = steps.index(name)

# Optional quick reset
with st.sidebar:
    if st.button("🔄 Reset app"):
        for k in list(st.session_state.keys()):
            if k != "step":
                del st.session_state[k]
        st.session_state.step = 0
        rerun()

# -------------------- Panes --------------------
def pane_landing():
    col1, col2 = st.columns([1.1, 2])
    with col1:
        img = hero_image_path()
        if img:
            st.image(str(img), use_container_width=True)
        else:
            st.markdown("<div style='font-size:140px; line-height:1; text-align:center'>🎧</div>", unsafe_allow_html=True)
    with col2:
        # Short, punchy opening line
        st.markdown("<div class='hero-h1'>Two models compete to see who predicts Spotify popularity best.</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>Data scientists often build many models and compare to see which is best. Pick any two models. See who wins.</div>", unsafe_allow_html=True)
        if st.button("Explore the data →", use_container_width=True):
            goto("data"); rerun()

def pane_data(manifest: Dict[str,Any]):
    # Simple, big, and quick to scan — no white box
    st.markdown("<div class='jumbo'>The data</div>", unsafe_allow_html=True)
    st.markdown("<div class='big'>We split songs into <b>3 groups</b> by popularity:</div>", unsafe_allow_html=True)
    st.markdown("<ul class='big-list'><li>Low · Medium · High</li></ul>", unsafe_allow_html=True)

    st.markdown("<div class='big'>We predict the group using factors like:</div>", unsafe_allow_html=True)
    st.markdown("<ul class='big-list'><li>danceability, energy, tempo, valence, loudness</li><li>+ key, mode, genre, time signature</li></ul>", unsafe_allow_html=True)

    st.markdown("<div class='big'>We compare two models. <b>Highest accuracy</b> wins.</div>", unsafe_allow_html=True)
    st.markdown("<div class='big'>Baseline guess: <b>33%</b>. These should do better.</div>", unsafe_allow_html=True)

    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    c1, _ = st.columns(2)
    if c1.button("Pick Model A →"):
        goto("fam_A"); rerun()

def pane_fam(slot: str, manifest: Dict[str,Any]):
    st.subheader(f"Pick a model family — Model {slot}")
    available = list(manifest["tasks"]["pop3"]["families"].keys())  # 'logreg','tree','mlp','rf','xgb','kmeans'
    label_map = {
        "logreg":"Logistic regression",
        "tree":"Decision tree",
        "mlp":"Neural network",
        "rf":"Random forest",
        "xgb":"XGBoost",
        "kmeans":"K-Means",
    }
    subtitle = {
        "logreg":"Clear & fast; probabilities.",
        "tree":"IF–THEN rules.",
        "mlp":"Learns complex patterns.",
        "rf":"Many decision trees (bagging).",
        "xgb":"Boosted trees.",
        "kmeans":"Cluster-based guess.",
    }
    items = [(label_map[k], subtitle[k]) for k in ["logreg","tree","mlp","rf","xgb","kmeans"] if k in available]

    def on_pick(label: str):
        fam = fam_key(label)
        st.session_state[f"fam_{slot}"] = fam
        st.session_state[f"tier_{slot}"] = None
        goto("tier_A" if slot == "A" else "tier_B")

    grid_cards("family", items, key_prefix=f"fam_{slot}", on_choose=on_pick, cols=3)
    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    if st.button("⬅️ Back"):
        goto("data" if slot == "A" else "fam_A"); rerun()

def pane_tier(slot: str):
    fam = st.session_state.get(f"fam_{slot}")
    if fam is None:
        goto("fam_A" if slot == "A" else "fam_B"); rerun(); return
    st.subheader(f"Choose flexibility — Model {slot}")

    items = [("Low",    tier_reason(fam, "Low")),
             ("Medium", tier_reason(fam, "Medium")),
             ("High",   tier_reason(fam, "High"))]

    def on_pick(tier_label: str):
        st.session_state[f"tier_{slot}"] = tier_label
        goto("fam_B" if slot == "A" else "compare")

    grid_cards("preset", items, key_prefix=f"tier_{slot}", on_choose=on_pick, cols=3)
    st.caption("Tip: **Medium** is often the sweet spot — flexible enough to learn patterns, but regularized to avoid memorizing noise.")
    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    if st.button("⬅️ Back"):
        goto("fam_A" if slot == "A" else "fam_B"); rerun()

def pane_compare(manifest: Dict[str,Any]):
    fam_A, tier_A = st.session_state.get("fam_A"), st.session_state.get("tier_A")
    fam_B, tier_B = st.session_state.get("fam_B"), st.session_state.get("tier_B")
    if not (fam_A and tier_A and fam_B and tier_B):
        goto("fam_A"); rerun(); return

    mA = load_metrics_for(manifest, fam_A, tier_A)
    mB = load_metrics_for(manifest, fam_B, tier_B)
    nameA = friendly_name(fam_A, tier_A)
    nameB = friendly_name(fam_B, tier_B)
    winner = _winner_of(mA, mB)  # "A" | "B" | "tie"

    st.subheader("Compare your two choices")
    c1, c2 = st.columns(2)

    def panel(col, name, fam, tier, metrics, is_winner: bool, is_tie: bool):
        with col:
            if is_tie:
                st.markdown("<div class='tie-chip'>🤝 Tie</div>", unsafe_allow_html=True)
            elif is_winner:
                st.markdown(f"<div class='winner-badge'>{BADGE_EMOJI}</div>", unsafe_allow_html=True)
                st.markdown("<div class='winner-chip'>Winner</div>", unsafe_allow_html=True)
                winner_cls = "winner-box flames" if WINNER_BADGE == "flames" else "winner-box"
                st.markdown(f"<div class='{winner_cls}'>", unsafe_allow_html=True)

            st.markdown(f"**{name}**")
            st.caption(tier_reason(fam, tier))
            st.metric("Accuracy (test)", f"{metrics.get('accuracy',0)*100:.1f}%")
            st.metric("Macro-F1 (test)", f"{metrics.get('macro_f1',0):.3f}")

            if is_winner and not is_tie:
                st.markdown("</div>", unsafe_allow_html=True)

    panel(c1, nameA, fam_A, tier_A, mA, winner == "A", winner == "tie")
    panel(c2, nameB, fam_B, tier_B, mB, winner == "B", winner == "tie")

    st.markdown("<div class='edu'>", unsafe_allow_html=True)
    st.markdown("**Takeaway**  \nLow = simpler (risk: underfitting) · High = very flexible (risk: overfitting) · **Medium** often balances the two.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("⬅️ Change Model A"): goto("fam_A"); rerun()
    if b2.button("⬅️ Change Tier A"):  goto("tier_A"); rerun()
    if b3.button("⬅️ Change Model B"): goto("fam_B"); rerun()
    if b4.button("⬅️ Change Tier B"):  goto("tier_B"); rerun()

# -------------------- Router --------------------
manifest = load_manifest()
pane = steps[st.session_state.step]

if pane == "landing":
    pane_landing()
elif pane == "data":
    pane_data(manifest)
elif pane == "fam_A":
    pane_fam("A", manifest)
elif pane == "tier_A":
    pane_tier("A")
elif pane == "fam_B":
    pane_fam("B", manifest)
elif pane == "tier_B":
    pane_tier("B")
else:
    pane_compare(manifest)
