"""Design system for the Streamlit app: palette, typography, motion, icons and small HTML builders.

Everything visual lives here so app.py stays focused on layout and logic.
Icons are hand-picked Lucide outlines (ISC licence, https://lucide.dev) rendered inline as SVG.
"""
from __future__ import annotations

import html as _html
import re

import streamlit as st

# ---------------------------------------------------------------------------
# Palette — warm paper, near-black ink, one accent. Tiers re-use the same ramp
# so only a "Hit" ever gets the accent colour.
# ---------------------------------------------------------------------------
PAPER = "#F3F0E8"
CARD = "#FBFAF6"
INK = "#1E1D1B"
INK_SOFT = "#4A4741"
MUTED = "#857F74"
LINE = "#DDD6C8"
ACCENT = "#C8512A"
ACCENT_SOFT = "#F1DDD3"
STONE = "#BDB5A6"
TEAL = "#3E6F6B"   # only used to separate clusters on the map

TIER_COLOURS = {"Low": STONE, "Moderate": INK_SOFT, "Hit": ACCENT}
CLUSTER_COLOURS = [ACCENT, INK, TEAL, "#B89A5E", "#8C6A9E", "#6B8F4E"]

# ---------------------------------------------------------------------------
# Icons (Lucide). Every shape gets pathLength=1 so the "draw-on" animation is uniform.
# ---------------------------------------------------------------------------
_ICONS = {
    "gamepad": '<line x1="6" x2="10" y1="11" y2="11"/><line x1="8" x2="8" y1="9" y2="13"/>'
               '<line x1="15" x2="15.01" y1="12" y2="12"/><line x1="18" x2="18.01" y1="10" y2="10"/>'
               '<path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
              '<path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
              '<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
              '<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "trending": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "archive": '<rect width="20" height="5" x="2" y="3" rx="1"/><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8"/>'
               '<path d="M10 12h4"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>'
             '<path d="M2 12h20"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/>'
           '<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/>'
           '<path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
                '<path d="M3 12A9 3 0 0 0 21 12"/>',
    "broom": '<path d="m13 11 9-9"/><path d="M14.6 12.6c.8.8.9 2.1.2 3L10 22l-8-8 6.4-4.8c.9-.7 2.2-.6 3 .2Z"/>'
             '<path d="m6.8 10.4 6.8 6.8"/><path d="m5 17 1.4-1.4"/>',
    "split": '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/>'
             '<path d="m15 9 6-6"/>',
    "layers": '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>'
              '<path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "scatter": '<circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/><circle cx="18.5" cy="5.5" r=".5" fill="currentColor"/>'
               '<circle cx="11.5" cy="11.5" r=".5" fill="currentColor"/><circle cx="7.5" cy="16.5" r=".5" fill="currentColor"/>'
               '<circle cx="17.5" cy="14.5" r=".5" fill="currentColor"/><path d="M3 3v16a2 2 0 0 0 2 2h16"/>',
    "star": '<path d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "arrow_up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "arrow_down": '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>',
}

_SHAPES = re.compile(r"<(path|line|polyline|circle|rect|ellipse)\s")


def icon(name: str, size: int = 20, cls: str = "") -> str:
    """Inline SVG icon; `cls` adds animation classes (e.g. 'draw', 'bob')."""
    body = _SHAPES.sub(lambda m: f'<{m.group(1)} pathLength="1" ', _ICONS[name])
    return (f'<svg class="ico {cls}" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>')


TIER_ICON = {"Hit": "trophy", "Moderate": "trending", "Low": "archive"}


def esc(text) -> str:
    return _html.escape(str(text))


def render(markup: str) -> None:
    """Render an HTML fragment. Whitespace is collapsed so Markdown never mistakes indentation for code."""
    st.markdown(re.sub(r">\s+<", "><", markup.strip()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&family=Geist+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap');

:root {{
  --paper:{PAPER}; --card:{CARD}; --ink:{INK}; --ink-soft:{INK_SOFT}; --muted:{MUTED};
  --line:{LINE}; --accent:{ACCENT}; --accent-soft:{ACCENT_SOFT}; --stone:{STONE};
  --ease: cubic-bezier(.22,.8,.24,1);
}}

/* Registered properties make numbers and rings animatable */
@property --n   {{ syntax: '<integer>'; inherits: false; initial-value: 0; }}
@property --deg {{ syntax: '<number>';  inherits: false; initial-value: 0; }}

/* ---------- base ---------- */
.stApp {{ background: var(--paper); }}
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button,
.stApp [data-testid="stWidgetLabel"], .stApp [data-baseweb="select"] div, .stApp table {{
  font-family: 'Geist', ui-sans-serif, system-ui, sans-serif; color: var(--ink);
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }}
footer {{ visibility: hidden; }}
::selection {{ background: var(--accent-soft); }}

/* subtle paper grain via layered gradients (no image assets) */
.stApp::before {{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.5;
  background-image: radial-gradient(rgba(30,29,27,.035) 1px, transparent 1px);
  background-size: 3px 3px;
}}

/* ---------- tabs as an underline nav ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.75rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{
  padding: .6rem 0 .75rem; background: transparent; color: var(--muted);
  font-weight: 500; letter-spacing: .01em; transition: color .25s var(--ease);
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--ink); }}
.stTabs [data-baseweb="tab"] [data-testid="stIconMaterial"] {{ transition: transform .35s var(--ease); }}
.stTabs [data-baseweb="tab"]:hover [data-testid="stIconMaterial"] {{ transform: translateY(-2px) rotate(-6deg); }}
.stTabs [aria-selected="true"] {{ color: var(--ink); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--ink); height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.6rem; }}

/* ---------- input panel ---------- */
.st-key-input_panel {{
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.4rem 1.4rem .6rem; animation: rise .6s var(--ease) both;
}}
.stApp [data-testid="stWidgetLabel"] p {{ font-size: .82rem; color: var(--ink-soft); font-weight: 500; }}
.stSlider [data-baseweb="slider"] [role="slider"] {{ transition: transform .2s var(--ease), box-shadow .2s; }}
.stSlider [data-baseweb="slider"] [role="slider"]:hover {{ transform: scale(1.2); box-shadow: 0 0 0 6px var(--accent-soft); }}
.stButton > button, .stDownloadButton > button {{
  border-radius: 999px; border: 1px solid var(--ink); background: transparent; color: var(--ink);
  font-weight: 500; transition: background .25s var(--ease), color .25s var(--ease), translate .25s var(--ease);
}}
.stButton > button:hover {{ background: var(--ink); color: var(--paper); translate: 0 -1px; }}
.stButton > button:hover p {{ color: var(--paper); }}

/* ---------- typography helpers ---------- */
.eyebrow {{ font: 500 .72rem/1 'Geist Mono', monospace; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }}
.display {{ font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; letter-spacing: -.01em; color: var(--ink); }}
.mono {{ font-family: 'Geist Mono', ui-monospace, monospace; }}
.muted {{ color: var(--muted); }}

/* ---------- masthead ---------- */
.mast {{ display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; margin-bottom: 1.4rem; animation: rise .7s var(--ease) both; }}
.mast .brand {{ display:flex; align-items:center; gap:.65rem; margin-bottom:.9rem; }}
.mast .brand .logo {{ width:38px; height:38px; border-radius:11px; display:grid; place-items:center; background: var(--ink); color: var(--paper); transition: rotate .5s var(--ease); }}
.mast .brand:hover .logo {{ rotate: -10deg; }}
.mast .brand b {{ font: 600 1rem 'Geist', sans-serif; letter-spacing: -.01em; }}
.mast h1 {{ font-size: clamp(2.3rem, 4.4vw, 3.6rem); line-height: 1.02; margin: 0; }}
.mast h1 em {{ color: var(--accent); }}
.mast p.lede {{ max-width: 34rem; color: var(--ink-soft); margin: .8rem 0 0; font-size: 1rem; line-height: 1.55; }}
.mast .meta {{ display:flex; gap:1.6rem; padding-bottom:.35rem; }}
.mast .meta div {{ text-align:right; }}
.mast .meta .v {{ font: 500 1.35rem 'Geist Mono', monospace; color: var(--ink); }}

/* ---------- cards ---------- */
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1.25rem 1.35rem; animation: rise .6s var(--ease) both; }}
.card + .card {{ margin-top: 1rem; }}
.card h3 {{ font: 500 1rem 'Geist', sans-serif; margin: 0; letter-spacing: -.005em; }}
.card-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom: 1rem; }}
.card-head .t {{ display:flex; align-items:center; gap:.55rem; }}
.card-head .t .ico {{ color: var(--muted); transition: color .3s, rotate .4s var(--ease); }}
.card:hover .card-head .t .ico {{ color: var(--accent); rotate: -8deg; }}
.d1 {{ animation-delay: .06s; }} .d2 {{ animation-delay: .12s; }} .d3 {{ animation-delay: .18s; }} .d4 {{ animation-delay: .24s; }}

/* ---------- verdict ---------- */
.verdict {{ display:grid; grid-template-columns: auto 1fr; gap: 1.6rem; align-items:center; }}
.ring {{
  --size: 148px; width: var(--size); height: var(--size); border-radius: 50%; position: relative;
  display:grid; place-items:center;
  background: conic-gradient(var(--tier) calc(var(--deg) * 1%), #ECE7DC 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #000 calc(100% - 11px));
          mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #000 calc(100% - 11px));
  animation: ring 1.3s var(--ease) both;
}}
.ring-wrap {{ position:relative; width:148px; height:148px; }}
.ring-wrap progress {{ position:absolute; width:1px; height:1px; opacity:0; }}
.ring-center {{ position:absolute; inset:0; display:grid; place-items:center; text-align:center; }}
.ring-center .pct {{ font: 500 2rem/1 'Geist Mono', monospace; counter-reset: n var(--n); animation: count 1.3s var(--ease) both; }}
.ring-center .pct::before {{ content: counter(n); }}
.ring-center .pct small {{ font-size: .9rem; color: var(--muted); }}
.ring-center .lbl {{ font: 500 .65rem 'Geist Mono', monospace; letter-spacing:.12em; text-transform:uppercase; color: var(--muted); margin-top:.25rem; }}
.verdict .tier-name {{ font-size: 3.4rem; line-height: .95; margin: .35rem 0 .5rem; display:flex; align-items:center; gap:.7rem; }}
.verdict .tier-name .ico {{ color: var(--tier); }}
.verdict .tier-name span {{ display:inline-block; animation: slide-in .7s var(--ease) both .1s; }}
.verdict p {{ margin:0; color: var(--ink-soft); line-height:1.5; max-width: 30rem; }}
.chip {{ display:inline-flex; align-items:center; gap:.35rem; font: 500 .7rem 'Geist Mono', monospace; letter-spacing:.06em;
        text-transform: uppercase; padding:.28rem .6rem; border-radius:999px; border:1px solid var(--line); color: var(--ink-soft); background: var(--paper); }}
.chip.accent {{ border-color: transparent; background: var(--accent-soft); color: var(--accent); }}
.chip .dot {{ width:6px; height:6px; border-radius:50%; background: currentColor; }}
.chip.live .dot {{ animation: pulse 1.8s ease-in-out infinite; }}

/* ---------- probability bars ---------- */
.bars {{ display:grid; gap: .95rem; }}
.bar-row {{ display:grid; grid-template-columns: 6.5rem 1fr 3.6rem; align-items:center; gap: .9rem; }}
.bar-row .name {{ display:flex; align-items:center; gap:.45rem; font-weight:500; font-size:.92rem; }}
.bar-row .name .sw {{ width:9px; height:9px; border-radius:3px; background: var(--c); }}
.bar-track {{ height: 10px; border-radius: 999px; background: #ECE7DC; overflow: hidden; }}
.bar-fill {{ height:100%; width: var(--w); background: var(--c); border-radius: 999px; transform-origin: left;
             animation: grow 1.1s var(--ease) both; animation-delay: var(--delay, 0s); }}
.bar-row .val {{ font: 500 .92rem 'Geist Mono', monospace; text-align:right; counter-reset: n var(--n); animation: count 1.1s var(--ease) both; }}
.bar-row .val::before {{ content: counter(n) "%"; }}
.bar-row.top .name {{ color: var(--ink); }}

/* ---------- stat tiles ---------- */
.stats {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; }}
.stat {{ border:1px solid var(--line); border-radius: 14px; padding: .9rem 1rem; background: var(--paper);
         transition: translate .3s var(--ease), border-color .3s; animation: rise .6s var(--ease) both; }}
.stat:hover {{ translate: 0 -3px; border-color: var(--ink-soft); }}
.stat .k {{ display:flex; align-items:center; gap:.4rem; }}
.stat .k .ico {{ color: var(--muted); transition: color .3s, scale .3s var(--ease); }}
.stat:hover .k .ico {{ color: var(--accent); scale: 1.12; }}
.stat .v {{ font: 500 1.45rem 'Geist Mono', monospace; margin-top: .45rem; letter-spacing: -.02em; }}
.stat .s {{ font-size: .78rem; color: var(--muted); margin-top: .15rem; }}

/* ---------- lists / tables ---------- */
.titles {{ display:grid; gap: 0; }}
.title-row {{ display:grid; grid-template-columns: 1.6rem 1fr auto auto; gap: .9rem; align-items:center;
              padding: .7rem 0; border-top: 1px solid var(--line); animation: rise .5s var(--ease) both; }}
.title-row:first-child {{ border-top: 0; }}
.title-row .idx {{ font: 500 .75rem 'Geist Mono', monospace; color: var(--muted); }}
.title-row .nm {{ font-weight: 500; }}
.title-row .sub {{ font-size: .78rem; color: var(--muted); }}
.title-row .sales {{ font: 500 .9rem 'Geist Mono', monospace; }}
.title-row:hover .nm {{ color: var(--accent); }}
.tierchip {{ font: 500 .66rem 'Geist Mono', monospace; letter-spacing:.08em; text-transform:uppercase; padding:.22rem .5rem; border-radius:6px; }}
.tierchip.Hit {{ background: var(--accent-soft); color: var(--accent); }}
.tierchip.Moderate {{ background: #E4E0D8; color: var(--ink-soft); }}
.tierchip.Low {{ background: #EEEAE2; color: var(--muted); }}

table.board {{ width:100%; border-collapse: collapse; font-size: .9rem; }}
table.board th {{ font: 500 .68rem 'Geist Mono', monospace; letter-spacing:.12em; text-transform: uppercase; color: var(--muted);
                  text-align:right; padding: .55rem .6rem; border-bottom: 1px solid var(--line); }}
table.board th:nth-child(-n+3) {{ text-align:left; }}
table.board td {{ padding: .7rem .6rem; border-bottom: 1px solid var(--line); text-align:right; font-family:'Geist Mono', monospace; }}
table.board td:nth-child(-n+3) {{ text-align:left; font-family:'Geist', sans-serif; }}
table.board tbody tr {{ transition: background .2s; animation: rise .5s var(--ease) both; }}
table.board tbody tr:hover {{ background: #F5F1E8; }}
table.board tr.deployed td {{ background: #FAEEE7; }}
table.board tr.deployed td:first-child {{ box-shadow: inset 3px 0 0 var(--accent); }}
.f1cell {{ display:flex; align-items:center; justify-content:flex-end; gap:.6rem; }}
.f1bar {{ width: 90px; height: 6px; background:#ECE7DC; border-radius: 99px; overflow:hidden; }}
.f1bar i {{ display:block; height:100%; width: var(--w); background: var(--ink); transform-origin:left; animation: grow 1s var(--ease) both; }}
tr.deployed .f1bar i {{ background: var(--accent); }}
.delta-up {{ color: #3E6F4B; }} .delta-down {{ color: var(--accent); }}

/* ---------- segments ---------- */
.segs {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: .9rem; }}
.seg {{ background: var(--card); border:1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem;
        transition: translate .3s var(--ease), border-color .3s; animation: rise .6s var(--ease) both; }}
.seg:hover {{ translate: 0 -3px; border-color: var(--c); }}
.seg .hd {{ display:flex; align-items:center; gap:.5rem; }}
.seg .hd i {{ width:10px; height:10px; border-radius: 50%; background: var(--c); }}
.seg h4 {{ margin:.55rem 0 .2rem; font-family: 'Instrument Serif', serif; font-weight:400; font-size: 1.55rem; line-height:1.1; }}
.seg .kv {{ display:grid; grid-template-columns: 1fr 1fr; gap:.35rem .8rem; margin-top:.7rem; font-size:.8rem; }}
.seg .kv b {{ display:block; font: 500 .95rem 'Geist Mono', monospace; color: var(--ink); }}
.seg .ex {{ margin-top:.75rem; font-size:.78rem; color: var(--muted); line-height:1.45; }}

/* ---------- about: pipeline steps ---------- */
.steps {{ display:grid; grid-template-columns: repeat(5, 1fr); gap: .8rem; }}
.step {{ border-top: 2px solid var(--ink); padding-top: .9rem; animation: rise .6s var(--ease) both; }}
.step .ico {{ color: var(--ink); transition: rotate .4s var(--ease), color .3s; }}
.step:hover .ico {{ rotate: -10deg; color: var(--accent); }}
.step b {{ display:block; margin:.6rem 0 .25rem; font-weight:500; }}
.step span {{ font-size: .82rem; color: var(--ink-soft); line-height:1.45; }}
.prose p {{ color: var(--ink-soft); line-height: 1.65; }}

/* ---------- keyframes (transform / opacity only, plus registered props) ---------- */
@keyframes rise     {{ from {{ opacity: 0; translate: 0 10px; }} }}
@keyframes slide-in {{ from {{ opacity: 0; translate: -14px 0; }} }}
@keyframes grow     {{ from {{ transform: scaleX(0); }} }}
@keyframes ring     {{ from {{ --deg: 0; }} }}
@keyframes count    {{ from {{ --n: 0; }} }}
@keyframes draw     {{ from {{ stroke-dashoffset: 1; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes bob      {{ 0%,100% {{ translate: 0 0; }} 50% {{ translate: 0 -3px; }} }}
@keyframes pulse    {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .35; }} }}

.ico.draw * {{ stroke-dasharray: 1; stroke-dashoffset: 0; animation: draw 1.2s var(--ease) both .15s; }}
.ico.bob {{ animation: bob 2.8s ease-in-out infinite 1.3s; }}

@media (prefers-reduced-motion: reduce) {{
  .card, .stat, .seg, .step, .mast, .title-row, table.board tbody tr, .st-key-input_panel,
  .bar-fill, .f1bar i, .ring, .ico.draw *, .ico.bob, .chip.live .dot, .verdict .tier-name span {{ animation: none !important; }}
  .ring-center .pct, .bar-row .val {{ animation: none !important; }}
}}

@media (max-width: 900px) {{
  .mast {{ flex-direction: column; align-items: flex-start; }}
  .mast .meta div {{ text-align:left; }}
  .verdict {{ grid-template-columns: 1fr; }}
  .stats {{ grid-template-columns: 1fr; }}
  .steps {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
