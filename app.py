"""Interface web Streamlit — Agent de Repositionnement IA."""

import json
import re
from typing import Dict, List
import streamlit as st
import anthropic

from config import MODEL, SYSTEM_PROMPT, ANTHROPIC_API_KEY, AI_ROLES
from agent import UserProfile, AnalysisResult, build_profile_dict
from database import save_profile, init_database, get_profile
from linkedin import extract_linkedin_profile, validate_linkedin_url

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quel est votre rôle dans l'IA ?",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
/* Reset Streamlit chrome */
#MainMenu, footer, header { visibility:hidden !important; }
[data-testid="collapsedControl"] { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }
.block-container {
    max-width:1120px !important;
    padding:0 48px 80px !important;
    margin:0 auto !important;
}
[data-testid="stAppViewContainer"] { background:#fff; }
html, body, [class*="css"], * {
    font-family:'Inter','SF Pro Display',-apple-system,BlinkMacSystemFont,sans-serif !important;
    box-sizing:border-box;
}
h1, h2, h3, h4, h5 {
    font-family:'Inter','SF Pro Display',-apple-system,sans-serif !important;
    letter-spacing:-.5px;
}

/* Nav */
.nav {
    display:flex; align-items:center; justify-content:space-between;
    padding:24px 0 20px; border-bottom:1px solid #f1f5f9; margin-bottom:0;
}
.nav-logo { font-size:15px; font-weight:700; color:#0f172a; letter-spacing:-.2px; }
.nav-right { font-size:13px; color:#94a3b8; }

/* Hero */
.hero {
    padding:32px 0 12px; text-align:center !important;
    position:relative;
}
.hero::before {
    content:''; position:absolute; top:-20px; left:50%; transform:translateX(-50%);
    width:900px; height:500px; z-index:-1; pointer-events:none;
    background:radial-gradient(ellipse at center, rgba(91,91,214,.10) 0%, rgba(91,91,214,0) 65%);
}
.hero * { text-align:center !important; }

/* Social proof row au-dessus du titre */
.social-proof {
    display:inline-flex; align-items:center; gap:12px;
    background:#fff; border:1px solid #f1f5f9; border-radius:100px;
    padding:6px 18px 6px 8px; margin-bottom:22px;
    box-shadow:0 2px 8px rgba(0,0,0,.04);
}
.avatar-stack { display:flex; }
.avatar-stack img {
    width:28px; height:28px; border-radius:50%;
    border:2px solid #fff; object-fit:cover;
    margin-left:-8px;
}
.avatar-stack img:first-child { margin-left:0; }
.social-proof-text { font-size:12.5px; color:#475569; font-weight:500; }
.social-proof-text strong { color:#0f172a; font-weight:700; }

.hero-eyebrow {
    display:inline-block; background:#f0eeff; color:#5b5bd6;
    border-radius:100px; padding:5px 14px; font-size:11px; font-weight:700;
    letter-spacing:.8px; text-transform:uppercase; margin-bottom:24px;
}
.hero h1 {
    font-size:46px !important; font-weight:900 !important; line-height:1.08 !important;
    letter-spacing:-2px !important; color:#0f172a !important; margin:0 0 16px !important;
}
.hero h1 span { color:#5b5bd6 !important; }
.hero-sub {
    font-size:16px; color:#64748b; line-height:1.6;
    max-width:460px; margin-left:auto !important; margin-right:auto !important;
    margin-bottom:8px; font-weight:400; display:block;
}

/* Input URL dans hero */
.hero-note { font-size:12px; color:#94a3b8; margin-top:10px; text-align:center; }

/* Séparateur */
.sep { border:none; border-top:1px solid #f1f5f9; margin:60px 0; }

/* Features */
.feats { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }
.feat-card {
    padding:28px 22px; border:1px solid #f1f5f9;
    border-radius:16px; background:#fafafa;
}
.feat-icon { font-size:22px; margin-bottom:14px; display:block; }
.feat-t { font-size:14px; font-weight:700; color:#0f172a; margin-bottom:8px; }
.feat-d { font-size:13px; color:#64748b; line-height:1.7; }

/* Section header */
.s-label {
    font-size:11px; font-weight:700; letter-spacing:1.2px;
    color:#94a3b8; text-transform:uppercase; margin-bottom:10px;
}
.s-title {
    font-size:30px; font-weight:800; letter-spacing:-1px;
    color:#0f172a; margin-bottom:10px; line-height:1.2;
}
.s-sub { font-size:15px; color:#64748b; line-height:1.65; }

/* Steps */
.steps { display:flex; flex-direction:column; }
.step-li {
    display:flex; gap:18px; padding:18px 0;
    border-bottom:1px solid #f8f8f8;
}
.step-li:last-child { border-bottom:none; }
.s-num {
    flex-shrink:0; width:26px; height:26px; border-radius:50%;
    background:#5b5bd6; color:#fff; font-size:12px; font-weight:700;
    display:flex; align-items:center; justify-content:center; margin-top:3px;
}
.s-t { font-size:14px; font-weight:700; color:#0f172a; margin-bottom:4px; }
.s-d { font-size:13px; color:#64748b; line-height:1.65; }

/* Roles */
.roles-wrap { display:flex; flex-wrap:wrap; gap:8px; padding-top:8px; }
.r-pill {
    background:#f5f3ff; color:#5b5bd6; border:1px solid #ddd6fe;
    border-radius:100px; padding:6px 15px; font-size:12.5px; font-weight:500;
}

/* Testimonials */
.testis { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }
.testi {
    background:#fff; border:1px solid #f1f5f9; border-radius:16px;
    padding:24px 22px; transition:box-shadow .2s;
}
.testi:hover { box-shadow:0 8px 24px rgba(0,0,0,.06); }
.testi-quote {
    font-size:14px; color:#334155; line-height:1.65;
    margin-bottom:20px; font-style:italic;
}
.testi-who { display:flex; align-items:center; gap:12px; }
.testi-img {
    width:42px; height:42px; border-radius:50%;
    object-fit:cover; flex-shrink:0;
}
.testi-name { font-size:13.5px; font-weight:700; color:#0f172a; margin:0; }
.testi-role { font-size:12px; color:#64748b; margin:2px 0 0; }
.testi-arrow { color:#5b5bd6; font-weight:600; }

/* Product preview mockup */
.preview-wrap {
    margin:40px auto 0; max-width:720px; position:relative;
    background:linear-gradient(135deg,#f5f3ff 0%,#fafafa 100%);
    border-radius:20px; padding:36px 24px 0; border:1px solid #f1f5f9;
}
.preview-label {
    position:absolute; top:-14px; left:50%; transform:translateX(-50%);
    background:#fff; border:1px solid #e2e8f0; border-radius:100px;
    padding:5px 14px; font-size:11px; font-weight:600; color:#64748b;
    letter-spacing:.5px; text-transform:uppercase;
}
.preview-card {
    background:#fff; border-radius:14px; padding:24px 28px;
    box-shadow:0 8px 32px rgba(91,91,214,.12);
    border:1px solid #ede9fe; margin-bottom:-24px;
}
.preview-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; padding-bottom:14px; border-bottom:1px solid #f8f8f8; }
.preview-score-pill {
    display:inline-flex; align-items:center; gap:8px;
    background:linear-gradient(135deg,#22c55e,#16a34a); color:#fff;
    border-radius:100px; padding:6px 14px; font-size:13px; font-weight:700;
}
.preview-role { font-size:15px; font-weight:700; color:#5b5bd6; }
.preview-row { display:flex; gap:10px; margin-bottom:10px; flex-wrap:wrap; }
.preview-chip {
    background:#f5f3ff; color:#5b21b6; border-radius:100px;
    padding:4px 12px; font-size:12px; font-weight:500;
}
.preview-step {
    background:#fafafa; border-left:3px solid #5b5bd6;
    border-radius:0 6px 6px 0; padding:10px 14px; font-size:12.5px;
    color:#0f172a; margin-bottom:6px;
}
.preview-step-meta { color:#94a3b8; font-size:11px; margin-top:2px; }

/* Visual split section */
.split-row { display:grid; grid-template-columns:1fr 1fr; gap:40px; align-items:center; }
.split-img {
    width:100%; height:340px; object-fit:cover;
    border-radius:20px; box-shadow:0 8px 32px rgba(0,0,0,.08);
}
.split-text .s-label { margin-bottom:14px; }
.split-text .s-title { font-size:26px; margin-bottom:16px; }

/* Personas / For who */
.personas { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }
.persona {
    text-align:center; padding:20px 16px;
    border:1px solid #f1f5f9; border-radius:16px; background:#fafafa;
}
.persona-img {
    width:84px; height:84px; border-radius:50%; object-fit:cover;
    margin:0 auto 14px; border:3px solid #fff;
    box-shadow:0 4px 16px rgba(0,0,0,.08);
}
.persona-tag {
    display:inline-block; background:#ede9fe; color:#5b21b6;
    border-radius:100px; padding:3px 10px; font-size:11px; font-weight:600;
    margin-bottom:8px; letter-spacing:.3px;
}
.persona-name { font-size:14px; font-weight:700; color:#0f172a; margin-bottom:4px; }
.persona-desc { font-size:12.5px; color:#64748b; line-height:1.55; }

/* CTA band */
.cta-band {
    background:#5b5bd6; border-radius:20px;
    padding:60px 48px; text-align:center; margin:60px 0 0;
}
.cta-band h2 {
    font-size:30px; font-weight:800; color:#fff;
    margin-bottom:10px; letter-spacing:-.5px;
}
.cta-band p { font-size:15px; color:rgba(255,255,255,.7); margin-bottom:0; }

/* Boutons Streamlit — violet cohérent (inclut form_submit_button) */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
button[kind="secondaryFormSubmit"],
button[kind="primaryFormSubmit"] {
    background:#5b5bd6 !important; color:#fff !important;
    border:1px solid #5b5bd6 !important; border-radius:10px !important;
    font-weight:600 !important; font-size:14px !important;
    padding:11px 24px !important; letter-spacing:-.1px !important;
    transition:opacity .15s !important; box-shadow:none !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
button[kind="primary"]:hover,
button[kind="secondaryFormSubmit"]:hover,
button[kind="primaryFormSubmit"]:hover {
    opacity:.85 !important; background:#5b5bd6 !important; color:#fff !important;
}
.stButton > button:focus,
[data-testid="stFormSubmitButton"] > button:focus {
    box-shadow:none !important; outline:none !important;
}

/* Bouton CTA band (blanc) */
.cta-btn .stButton > button {
    background:#fff !important; color:#5b5bd6 !important;
    font-size:15px !important; padding:13px 32px !important;
}

/* Results */
.score-wrap { text-align:center; padding:28px 20px; border-radius:16px; color:#fff; }
.score-num  { font-size:76px; font-weight:900; line-height:1; letter-spacing:-2px; }
.score-lbl  { font-size:13px; opacity:.85; margin-top:8px; }
.role-badge {
    background:#f5f3ff; border:2px solid #5b5bd6;
    border-radius:12px; padding:14px 20px;
    font-size:18px; font-weight:700; color:#4c4cbe; text-align:center;
}
.res-step {
    background:#fafafa; border-left:3px solid #5b5bd6;
    border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;
}
.res-step-t { font-weight:700; color:#0f172a; font-size:13.5px; }
.res-step-m { color:#64748b; font-size:12px; margin-top:3px; }
.tag     { display:inline-block; background:#ede9fe; color:#5b21b6;
           border-radius:20px; padding:4px 12px; margin:3px; font-size:12.5px; }
.tag-red { background:#fef2f2; color:#b91c1c; }
.li-card {
    background:#fafafa; border:1px solid #e2e8f0;
    border-radius:12px; padding:18px 22px; margin-bottom:12px;
}
.li-name  { font-size:20px; font-weight:700; color:#0f172a; }
.li-title { font-size:14px; color:#5b5bd6; margin-top:3px; }
.li-meta  { font-size:13px; color:#64748b; margin-top:6px; }
.saved-bar {
    background:#f0fdf4; border:1px solid #bbf7d0;
    border-radius:10px; padding:12px 18px; color:#166534; font-weight:600;
    font-size:14px;
}

/* ── Training dashboard ── */
.xp-bar-wrap {
    background:linear-gradient(135deg,#5b5bd6,#7c3aed);
    border-radius:20px; padding:28px 32px; color:#fff; margin-bottom:32px;
}
.xp-bar-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
.xp-label { font-size:12px; font-weight:600; letter-spacing:.8px; text-transform:uppercase; opacity:.85; }
.xp-value { font-size:30px; font-weight:900; letter-spacing:-.5px; }
.xp-bar-bg {
    width:100%; height:10px; background:rgba(255,255,255,.2);
    border-radius:100px; overflow:hidden;
}
.xp-bar-fill {
    height:100%; background:#fff; border-radius:100px;
    transition:width .4s;
}
.xp-stats { display:flex; gap:24px; margin-top:14px; font-size:13px; opacity:.9; }

.mission-card {
    background:#fff; border:1.5px solid #e2e8f0; border-radius:16px;
    padding:22px 26px; margin-bottom:14px;
    display:flex; align-items:center; gap:20px;
    transition:border-color .2s, box-shadow .2s;
}
.mission-card.done { border-color:#22c55e; background:#f0fdf4; }
.mission-card.active { border-color:#5b5bd6; box-shadow:0 4px 20px rgba(91,91,214,.15); }
.mission-card.locked { opacity:.5; }
.mission-num {
    flex-shrink:0; width:44px; height:44px; border-radius:12px;
    background:#f5f3ff; color:#5b5bd6; font-size:17px; font-weight:800;
    display:flex; align-items:center; justify-content:center;
}
.mission-card.done .mission-num { background:#22c55e; color:#fff; }
.mission-body { flex:1; }
.mission-title { font-size:15px; font-weight:700; color:#0f172a; margin-bottom:4px; }
.mission-meta { font-size:12.5px; color:#64748b; }
.mission-badge {
    background:#22c55e; color:#fff; padding:5px 12px; border-radius:100px;
    font-size:12px; font-weight:700;
}
.mission-xp { color:#5b5bd6; font-weight:700; font-size:13px; margin-top:3px; }

/* Mission runner */
.mission-brief {
    background:#f5f3ff; border:1px solid #ddd6fe; border-radius:14px;
    padding:20px 24px; margin-bottom:20px;
}
.mission-brief-label { font-size:11px; font-weight:700; letter-spacing:.8px; color:#5b21b6; text-transform:uppercase; margin-bottom:8px; }
.mission-brief-title { font-size:18px; font-weight:800; color:#0f172a; margin-bottom:10px; }
.mission-brief-desc { font-size:14px; color:#475569; line-height:1.65; }

.score-result {
    text-align:center; padding:32px; background:#fff;
    border:2px solid #5b5bd6; border-radius:16px; margin-top:20px;
}
.score-result-num { font-size:60px; font-weight:900; color:#5b5bd6; line-height:1; }
.score-result-lbl { font-size:13px; color:#64748b; margin-top:8px; }
.score-result-fb { font-size:14px; color:#334155; margin-top:16px; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase": "landing",
        "linkedin_data": None,
        "profile": None,
        "result": None,
        "profile_id": None,
        "api_history": [],
        # Gamification
        "mission_status": {},    # {0: {"status":"done","score":85,"feedback":"..."}, ...}
        "mission_chat": {},      # {0: [{"role":"user","content":"..."}], ...}
        "active_mission": None,  # index de la mission en cours
        "total_xp": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_client():
    if not ANTHROPIC_API_KEY:
        st.error("Clé API Anthropic manquante — ajoutez ANTHROPIC_API_KEY dans les secrets Streamlit.")
        st.stop()
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def run_analysis(profile: UserProfile) -> AnalysisResult:
    client = get_client()
    roles_list = "\n".join(f"- {r}" for r in AI_ROLES)
    prompt = f"""Analyse ce profil et retourne UNIQUEMENT un JSON valide.

PROFIL :
- Nom : {profile.nom}
- Poste : {profile.poste_actuel}
- Expérience : {profile.annees_experience} ans
- Secteur : {profile.secteur}
- Compétences : {", ".join(profile.competences)}
- Formation : {profile.formation}
- Objectif : {profile.objectif_reconversion}

RÔLES IA :
{roles_list}

IMPORTANT pour le plan_formation :
- PAS de cours en ligne (Coursera, Udemy, etc.)
- UNIQUEMENT des exercices pratiques concrets à réaliser DIRECTEMENT avec Claude Opus 4.6 (claude.ai) ou ChatGPT
- Chaque étape = une mission pratique que la personne exécute elle-même en dialoguant avec l'IA, appliquée à SON métier actuel ({profile.poste_actuel} / {profile.secteur})
- Exemples de formats attendus : "Faire auditer tes 5 derniers emails par Claude", "Créer 10 prompts-types pour ton workflow X", "Automatiser une tâche Y avec Claude Projects"
- Les ressources = liens vers claude.ai / documentation Anthropic / exemples de prompts, PAS vers des plateformes de cours

JSON :
{{
  "role_cible": "rôle parmi la liste",
  "score": entier_0_100,
  "score_justification": "2 phrases",
  "competences_transferables": ["c1","c2","c3"],
  "competences_a_acquerir": ["c1","c2","c3"],
  "plan_formation": [
    {{"etape":"1. Mission pratique concrète appliquée à son secteur","duree":"X jours","ressources":"Outil IA à utiliser + type de prompts/exercice"}},
    {{"etape":"2. Mission pratique concrète","duree":"X jours","ressources":"Outil + exercice"}},
    {{"etape":"3. Projet pratique plus ambitieux","duree":"X semaines","ressources":"Outil + livrable attendu"}},
    {{"etape":"4. Projet final démontrable","duree":"X semaines","ressources":"Outil + format du livrable portfolio"}}
  ],
  "salaire_estime": "fourchette en France",
  "perspectives": "2-3 phrases",
  "message_encouragement": "2-3 phrases personnalisées"
}}"""
    resp = client.messages.create(
        model=MODEL, max_tokens=2048,
        system="Tu es expert RH et formation IA. Retourne uniquement du JSON valide.",
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"},
    )
    # Récupérer le texte (éviter les blocs thinking)
    raw = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text" or hasattr(block, "text"):
            candidate = getattr(block, "text", "") or ""
            if candidate.strip():
                raw = candidate.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```", "", raw).strip()

    # Parser en trouvant le premier JSON valide dans le texte
    def parse_json(text: str) -> Dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Scan : essayer tous les points où ça commence par {
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
        raise json.JSONDecodeError("Aucun JSON valide trouvé", text, 0)

    data = parse_json(raw)
    return AnalysisResult(**data)

def mission_coach_system(profile: UserProfile, result: AnalysisResult, mission: Dict) -> str:
    return f"""Tu es le coach d'entraînement IA de {profile.nom}, qui travaille actuellement comme {profile.poste_actuel} dans le secteur {profile.secteur} avec {profile.annees_experience} ans d'expérience.
Son rôle cible : {result.role_cible}.

Ta mission : l'accompagner dans l'exercice suivant, appliqué à SON contexte métier réel.

EXERCICE EN COURS :
- Titre : {mission['etape']}
- Durée indicative : {mission['duree']}
- Méthode : {mission['ressources']}

Ton rôle :
1. Présente brièvement l'objectif concret en 2 phrases
2. Pose UNE question à la fois pour l'aider à démarrer l'exercice
3. Accompagne-le pas à pas, adapte à son métier
4. Reste motivant, concret, pragmatique
5. Quand il a produit un livrable satisfaisant, dis-lui qu'il peut valider l'exercice"""


def evaluate_mission(profile: UserProfile, mission: Dict, chat_history: List[Dict]) -> Dict:
    """Demande à Claude d'évaluer la mission et de donner un score + feedback."""
    client = get_client()
    conversation_text = "\n\n".join(
        f"[{m['role'].upper()}]\n{m['content']}" for m in chat_history
    )
    eval_prompt = f"""Tu évalues l'accomplissement d'une mission d'entraînement IA.

UTILISATEUR : {profile.nom}, {profile.poste_actuel} ({profile.secteur})

MISSION DEMANDÉE :
- {mission['etape']}
- Méthode : {mission['ressources']}

CONVERSATION AVEC LE COACH :
{conversation_text[:6000]}

Évalue la qualité du travail réalisé. Retourne UNIQUEMENT ce JSON :
{{
  "score": entier_0_100,
  "feedback": "2-3 phrases : ce qui a été bien fait, ce qu'il faudrait approfondir",
  "badge": "un nom de badge court valorisant (ex: 'Explorateur IA', 'Tacticien de prompts', 'Stratège métier')"
}}

Critères de scoring :
- 85-100 : livrable concret, appliqué au métier, qualité pro
- 60-84 : solide mais peut être plus détaillé/personnalisé
- 40-59 : esquissé mais manque de profondeur
- 0-39 : insuffisant, à reprendre

Sois exigeant mais encourageant. UNIQUEMENT le JSON."""

    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        system="Tu es un évaluateur exigeant et juste. Retourne uniquement du JSON valide.",
        messages=[{"role": "user", "content": eval_prompt}],
    )
    raw = ""
    for block in resp.content:
        candidate = getattr(block, "text", "") or ""
        if candidate.strip():
            raw = candidate.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(raw[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
        return {"score": 60, "feedback": "Évaluation indisponible, relance possible.", "badge": "Apprenti IA"}


def render_training_dashboard(profile: UserProfile, result: AnalysisResult):
    """Dashboard de gamification avec XP et missions."""
    missions = result.plan_formation
    n_total = len(missions)
    statuses = st.session_state.mission_status or {}
    n_done = sum(1 for i in range(n_total) if statuses.get(i, {}).get("status") == "done")
    total_xp = st.session_state.total_xp
    max_xp = n_total * 500
    progress_pct = int((n_done / n_total) * 100) if n_total else 0

    st.markdown(f"""
    <div class="xp-bar-wrap">
        <div class="xp-bar-top">
            <div>
                <div class="xp-label">Votre progression</div>
                <div class="xp-value">{total_xp} XP</div>
            </div>
            <div style="text-align:right;">
                <div class="xp-label">{n_done} / {n_total} missions</div>
                <div style="font-size:22px;font-weight:800;margin-top:4px;">{progress_pct}%</div>
            </div>
        </div>
        <div class="xp-bar-bg">
            <div class="xp-bar-fill" style="width:{progress_pct}%;"></div>
        </div>
        <div class="xp-stats">
            <span>🎯 Rôle cible : {result.role_cible}</span>
            <span>🏆 Niveau max : {max_xp} XP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Vos missions d'entraînement")
    st.caption("Chaque mission est un exercice pratique réalisé directement avec Claude Opus 4.6.")

    for i, mission in enumerate(missions):
        status = statuses.get(i, {}).get("status", "todo")
        locked = i > 0 and statuses.get(i - 1, {}).get("status") != "done"

        card_class = "mission-card"
        if status == "done":
            card_class += " done"
        elif locked:
            card_class += " locked"

        col_card, col_btn = st.columns([5, 1.5])
        with col_card:
            status_badge = ""
            if status == "done":
                s = statuses[i]
                status_badge = f'<div class="mission-xp">✓ {s.get("score",0)} / 100 · {s.get("badge","")}</div>'
            elif locked:
                status_badge = '<div class="mission-meta" style="color:#cbd5e1;">🔒 Débloquée après la mission précédente</div>'
            else:
                status_badge = '<div class="mission-xp">💰 500 XP à gagner</div>'

            st.markdown(f"""
            <div class="{card_class}">
                <div class="mission-num">{i+1}</div>
                <div class="mission-body">
                    <div class="mission-title">{mission['etape']}</div>
                    <div class="mission-meta">⏱ {mission['duree']} &nbsp;·&nbsp; {mission['ressources']}</div>
                    {status_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_btn:
            if status == "done":
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Revoir", key=f"review_{i}", use_container_width=True):
                    st.session_state.active_mission = i
                    st.session_state.phase = "mission"
                    st.rerun()
            elif locked:
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("🔒", key=f"locked_{i}", use_container_width=True, disabled=True)
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Démarrer →", key=f"start_{i}", type="primary", use_container_width=True):
                    st.session_state.active_mission = i
                    st.session_state.phase = "mission"
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Retour aux résultats", use_container_width=False):
        st.session_state.phase = "done"
        st.rerun()


def render_mission_runner(profile: UserProfile, result: AnalysisResult):
    """Interface chat pour réaliser une mission avec Claude."""
    idx = st.session_state.active_mission
    mission = result.plan_formation[idx]

    st.markdown(f"""
    <div class="mission-brief">
        <div class="mission-brief-label">Mission {idx+1} / {len(result.plan_formation)}</div>
        <div class="mission-brief-title">{mission['etape']}</div>
        <div class="mission-brief-desc">
            <strong>Durée :</strong> {mission['duree']}<br>
            <strong>Méthode :</strong> {mission['ressources']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_back, col_validate = st.columns([1, 1])
    with col_back:
        if st.button("← Retour au tableau de bord", use_container_width=True):
            st.session_state.phase = "training"
            st.rerun()
    with col_validate:
        can_validate = len(st.session_state.mission_chat.get(idx, [])) >= 2
        if st.button("Valider ma mission ✓", type="primary", use_container_width=True, disabled=not can_validate):
            with st.spinner("Évaluation en cours..."):
                evaluation = evaluate_mission(profile, mission, st.session_state.mission_chat[idx])
            score = evaluation.get("score", 0)
            xp_gained = score * 5
            st.session_state.mission_status[idx] = {
                "status": "done",
                "score": score,
                "feedback": evaluation.get("feedback", ""),
                "badge": evaluation.get("badge", "Apprenti IA"),
            }
            st.session_state.total_xp += xp_gained
            st.balloons()
            st.markdown(f"""
            <div class="score-result">
                <div class="score-result-num">{score} / 100</div>
                <div class="score-result-lbl">Mission validée · +{xp_gained} XP · 🏆 {evaluation.get('badge','')}</div>
                <div class="score-result-fb">{evaluation.get('feedback','')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Votre coach d'entraînement")

    # Initialiser chat si vide
    if idx not in st.session_state.mission_chat or not st.session_state.mission_chat[idx]:
        st.session_state.mission_chat[idx] = []
        # Premier message du coach
        client = get_client()
        system = mission_coach_system(profile, result, mission)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full = ""
            with client.messages.stream(
                model=MODEL, max_tokens=500, system=system,
                messages=[{"role": "user", "content": "Présente-moi cette mission et lance-moi le premier exercice."}],
                thinking={"type": "adaptive"},
            ) as stream:
                for chunk in stream.text_stream:
                    full += chunk
                    placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        st.session_state.mission_chat[idx].append({"role": "assistant", "content": full})

    # Afficher historique
    for msg in st.session_state.mission_chat[idx]:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input utilisateur
    user_input = st.chat_input("Votre réponse ou question...")
    if user_input:
        st.session_state.mission_chat[idx].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        client = get_client()
        system = mission_coach_system(profile, result, mission)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full = ""
            with client.messages.stream(
                model=MODEL, max_tokens=700, system=system,
                messages=st.session_state.mission_chat[idx],
                thinking={"type": "adaptive"},
            ) as stream:
                for chunk in stream.text_stream:
                    full += chunk
                    placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        st.session_state.mission_chat[idx].append({"role": "assistant", "content": full})
        st.rerun()


def render_results(profile: UserProfile, result: AnalysisResult, profile_id: str):
    st.markdown("---")
    st.markdown("### Résultats de votre analyse")

    score = result.score
    grad = ("#22c55e,#16a34a") if score >= 70 else ("#f59e0b,#d97706") if score >= 40 else ("#ef4444,#dc2626")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="score-wrap" style="background:linear-gradient(135deg,{grad});">
            <div class="score-num">{score}</div>
            <div class="score-lbl">Score de compatibilité / 100</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="role-badge">→ {result.role_cible}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(result.score_justification)
        st.success(f"Salaire estimé : {result.salaire_estime}")
    with col2:
        st.markdown("**Compétences transférables**")
        st.markdown(" ".join(f'<span class="tag">{c}</span>' for c in result.competences_transferables), unsafe_allow_html=True)
        st.markdown("<br>**À acquérir**", unsafe_allow_html=True)
        st.markdown(" ".join(f'<span class="tag tag-red">{c}</span>' for c in result.competences_a_acquerir), unsafe_allow_html=True)
        st.markdown("<br>**Plan de formation**", unsafe_allow_html=True)
        for step in result.plan_formation:
            st.markdown(f"""
            <div class="res-step">
                <div class="res-step-t">{step['etape']}</div>
                <div class="res-step-m">{step['duree']} &nbsp;·&nbsp; {step['ressources']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"**Perspectives** — {result.perspectives}")
    st.markdown(f"""
    <div style="background:#f5f3ff;border-radius:10px;padding:14px 18px;margin:16px 0;font-size:14px;color:#4c4cbe;font-style:italic;">
        {result.message_encouragement}
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="saved-bar">Profil sauvegardé — ID : {profile_id}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # CTA vers l'entraînement
    st.markdown("""
    <div style="background:linear-gradient(135deg,#5b5bd6,#7c3aed);border-radius:20px;padding:36px 32px;text-align:center;color:#fff;margin:20px 0;">
        <h3 style="color:#fff;font-size:22px;font-weight:800;margin-bottom:8px;">🎮 Prêt à démarrer votre entraînement ?</h3>
        <p style="color:rgba(255,255,255,.85);margin-bottom:20px;font-size:14px;">Réalisez vos 4 missions avec Claude Opus, gagnez des XP et débloquez votre rôle IA.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 Démarrer l'entraînement", type="primary", use_container_width=True):
            st.session_state.phase = "training"
            st.rerun()
    with col_b:
        if st.button("Analyser un autre profil", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# LANDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "landing":


    st.markdown("""
    <div class="nav">
        <span class="nav-logo">✦ &nbsp;Repositionnement IA</span>
        <span class="nav-right">Propulsé par Claude Opus 4.6</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div style="display:block;margin-bottom:16px;">
            <div class="social-proof">
                <div class="avatar-stack">
                    <img src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&h=100&fit=crop&crop=faces" alt="">
                    <img src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop&crop=faces" alt="">
                    <img src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=100&h=100&fit=crop&crop=faces" alt="">
                    <img src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100&h=100&fit=crop&crop=faces" alt="">
                    <img src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=100&h=100&fit=crop&crop=faces" alt="">
                </div>
                <span class="social-proof-text"><strong>+200</strong> professionnels ont trouvé leur rôle IA</span>
            </div>
        </div>
        <h1>Votre place dans<br>l'économie <span>IA</span></h1>
        <p class="hero-sub">
            Collez votre URL LinkedIn. En 60 secondes, découvrez le rôle IA
            fait pour vous et le chemin concret pour y accéder.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # URL input directement dans le hero
    _, col_c, _ = st.columns([1, 3, 1])
    with col_c:
        url = st.text_input("url", placeholder="https://www.linkedin.com/in/votre-profil",
                            label_visibility="collapsed")
        go = st.button("Analyser mon profil →", type="primary", use_container_width=True)
        st.markdown(
            "<p class='hero-note'>Gratuit &nbsp;·&nbsp; Sans inscription &nbsp;·&nbsp; Profil LinkedIn public requis</p>",
            unsafe_allow_html=True,
        )

    if go:
        if not url.strip():
            st.error("Collez votre URL LinkedIn.")
        elif not validate_linkedin_url(url.strip()):
            st.error("URL invalide — format attendu : https://www.linkedin.com/in/votre-profil")
        elif not ANTHROPIC_API_KEY:
            st.error("Clé API manquante.")
        else:
            st.session_state.linkedin_url = url.strip()
            with st.status("Récupération du profil...", expanded=False) as s:
                data = extract_linkedin_profile(url.strip(), ANTHROPIC_API_KEY)
            if data.get("error") == "login_required":
                st.warning(
                    "⚠️ **LinkedIn bloque l'accès depuis le cloud** — c'est une limitation de "
                    "LinkedIn qui restreint les accès automatisés. Pas d'inquiétude : "
                    "saisissez vos infos manuellement en quelques secondes, le résultat sera identique."
                )
                import time
                time.sleep(2)
                st.session_state.phase = "manual_input"
                st.rerun()
            elif "error" in data:
                st.error(data["error"])
            else:
                s.update(label=f"Profil récupéré ✓", state="complete")
                st.session_state.linkedin_data = data
                st.session_state.phase = "confirming"
                st.rerun()

    # Product preview mockup
    st.markdown("""
    <div class="preview-wrap">
        <div class="preview-label">Aperçu de votre analyse</div>
        <div class="preview-card">
            <div class="preview-header">
                <span class="preview-role">→ AI Product Manager</span>
                <span class="preview-score-pill">⭐ 82 / 100</span>
            </div>
            <div style="margin-bottom:14px;">
                <div style="font-size:11px;font-weight:700;letter-spacing:.5px;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">Compétences transférables</div>
                <div class="preview-row">
                    <span class="preview-chip">Gestion de projet</span>
                    <span class="preview-chip">Communication</span>
                    <span class="preview-chip">Analyse business</span>
                    <span class="preview-chip">Leadership</span>
                </div>
            </div>
            <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:.5px;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">Plan de formation</div>
                <div class="preview-step">
                    <div style="font-weight:700;">1. Audit IA de vos 10 dossiers clients</div>
                    <div class="preview-step-meta">5 jours · Claude Opus sur claude.ai — analyse et synthèse</div>
                </div>
                <div class="preview-step">
                    <div style="font-weight:700;">2. Créer 15 prompts-types pour votre workflow</div>
                    <div class="preview-step-meta">1 semaine · Claude Projects — bibliothèque de prompts métier</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div class="feats">
        <div class="feat-card">
            <span class="feat-icon">→</span>
            <div class="feat-t">Le rôle IA le plus accessible</div>
            <div class="feat-d">L'agent croise votre parcours avec les 10 métiers IA les plus demandés pour trouver votre meilleur point d'entrée.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">→</span>
            <div class="feat-t">Un plan en 4 étapes</div>
            <div class="feat-d">Durées, plateformes, ressources précises. Pas d'orientation vague — un chemin concret adapté à votre profil actuel.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">→</span>
            <div class="feat-t">Généré en temps réel</div>
            <div class="feat-d">Chaque analyse est produite à la volée par Claude Opus 4.6, le modèle de raisonnement avancé d'Anthropic.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Comment ça marche
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("""
        <div class="s-label">Fonctionnement</div>
        <div class="s-title">Trois étapes,<br>pas une de plus</div>
        <div class="s-sub">Votre LinkedIn fait la majorité du travail. Vous confirmez et précisez votre objectif — c'est tout.</div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div class="steps">
            <div class="step-li">
                <div class="s-num">1</div>
                <div>
                    <div class="s-t">Partagez votre URL LinkedIn</div>
                    <div class="s-d">L'agent récupère votre profil public automatiquement. Aucun formulaire.</div>
                </div>
            </div>
            <div class="step-li">
                <div class="s-num">2</div>
                <div>
                    <div class="s-t">Confirmez et ajoutez votre objectif</div>
                    <div class="s-d">Vérifiez les infos extraites, précisez ce que vous cherchez dans l'IA en une phrase.</div>
                </div>
            </div>
            <div class="step-li">
                <div class="s-num">3</div>
                <div>
                    <div class="s-t">Recevez votre analyse</div>
                    <div class="s-d">Rôle recommandé, score de compatibilité, compétences transférables et plan de formation.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Split image + texte
    st.markdown("""
    <div class="split-row">
        <div>
            <img class="split-img" src="https://images.unsplash.com/photo-1552664730-d307ca884978?w=800&h=600&fit=crop" alt="">
        </div>
        <div class="split-text">
            <div class="s-label">La démarche</div>
            <div class="s-title">Vos 10 ans d'expérience<br>sont votre avantage</div>
            <div class="s-sub">
                L'économie IA a besoin de profils qui comprennent un secteur,
                pas seulement de développeurs. Votre connaissance métier — marketing,
                finance, santé, RH — est précisément ce qui manque aux équipes tech.<br><br>
                L'agent identifie comment capitaliser sur cet acquis pour accéder
                aux rôles IA sans repartir de zéro.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Personas — pour qui
    st.markdown("""
    <div style="text-align:center;margin-bottom:32px;">
        <div class="s-label">Pour qui</div>
        <div class="s-title" style="margin:0 auto;">Des parcours<br>comme le vôtre</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="personas">
        <div class="persona">
            <img class="persona-img" src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=200&h=200&fit=crop&crop=faces" alt="">
            <div class="persona-tag">Marketing</div>
            <div class="persona-name">Camille, 34 ans</div>
            <div class="persona-desc">12 ans en marketing B2B,<br>veut piloter des projets IA<br>dans son secteur.</div>
        </div>
        <div class="persona">
            <img class="persona-img" src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&h=200&fit=crop&crop=faces" alt="">
            <div class="persona-tag">Finance</div>
            <div class="persona-name">Thomas, 41 ans</div>
            <div class="persona-desc">Contrôleur de gestion,<br>cherche un rôle plus tech<br>sans reprendre d'études.</div>
        </div>
        <div class="persona">
            <img class="persona-img" src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=200&h=200&fit=crop&crop=faces" alt="">
            <div class="persona-tag">RH</div>
            <div class="persona-name">Sarah, 29 ans</div>
            <div class="persona-desc">DRH adjointe, veut devenir<br>référente IA au sein de<br>son entreprise.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Témoignages
    st.markdown("""
    <div style="text-align:center;margin-bottom:32px;">
        <div class="s-label">Témoignages</div>
        <div class="s-title" style="margin:0 auto;">Ils ont trouvé<br>leur rôle IA</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="testis">
        <div class="testi">
            <div class="testi-quote">"J'hésitais depuis des mois. En 5 minutes j'ai eu une recommandation claire et un plan concret. Je démarre ma formation la semaine prochaine."</div>
            <div class="testi-who">
                <img class="testi-img" src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200&h=200&fit=crop&crop=faces" alt="">
                <div>
                    <div class="testi-name">Marie L.</div>
                    <div class="testi-role">Chef de projet <span class="testi-arrow">→</span> AI Product Manager</div>
                </div>
            </div>
        </div>
        <div class="testi">
            <div class="testi-quote">"Le score de compatibilité m'a donné confiance. J'étais persuadé qu'il fallait tout recommencer — en fait 70% de mes compétences sont transférables."</div>
            <div class="testi-who">
                <img class="testi-img" src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=200&h=200&fit=crop&crop=faces" alt="">
                <div>
                    <div class="testi-name">Julien M.</div>
                    <div class="testi-role">Consultant <span class="testi-arrow">→</span> AI Solutions Consultant</div>
                </div>
            </div>
        </div>
        <div class="testi">
            <div class="testi-quote">"Enfin un outil qui ne dit pas juste 'apprenez Python'. Le plan est adapté à mon niveau et à mon emploi du temps. Je recommande."</div>
            <div class="testi-who">
                <img class="testi-img" src="https://images.unsplash.com/photo-1580489944761-15a19d654956?w=200&h=200&fit=crop&crop=faces" alt="">
                <div>
                    <div class="testi-name">Laure B.</div>
                    <div class="testi-role">RH <span class="testi-arrow">→</span> AI Customer Success</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Rôles
    col_l2, col_r2 = st.columns([1, 2])
    with col_l2:
        st.markdown("""
        <div class="s-label">Périmètre</div>
        <div class="s-title">Les rôles<br>analysés</div>
        <div class="s-sub">Les métiers IA les plus accessibles aux profils expérimentés sans bagage technique.</div>
        """, unsafe_allow_html=True)
    with col_r2:
        pills = " ".join(f'<span class="r-pill">{r}</span>' for r in AI_ROLES)
        st.markdown(f'<div class="roles-wrap" style="padding-top:12px;">{pills}</div>', unsafe_allow_html=True)

    # CTA band
    st.markdown("""
    <div class="cta-band">
        <h2>Prêt à savoir où vous en êtes ?</h2>
        <p>L'analyse prend moins d'une minute.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_cta, _ = st.columns([1, 2, 1])
    with col_cta:
        if st.button("Commencer maintenant →", type="primary", use_container_width=True):
            st.session_state.phase = "url_input"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — URL input (fallback si arrivée directe)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "url_input":
    st.markdown("""
    <div class="nav">
        <span class="nav-logo">✦ &nbsp;Repositionnement IA</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Votre profil LinkedIn")
    st.markdown("<p style='color:#64748b;font-size:14px;'>L'agent récupère vos informations automatiquement.</p>", unsafe_allow_html=True)

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        url = st.text_input("URL", placeholder="https://www.linkedin.com/in/votre-profil", label_visibility="collapsed")
    with col_btn:
        go_btn = st.button("Analyser →", type="primary", use_container_width=True)

    st.caption("Le profil doit être public sur LinkedIn (Paramètres → Visibilité → Public)")
    st.markdown("<br>", unsafe_allow_html=True)
    manual_btn = st.button("Saisir manuellement")

    if go_btn:
        if not url.strip():
            st.error("Entrez une URL LinkedIn.")
        elif not validate_linkedin_url(url.strip()):
            st.error("URL invalide — format : https://www.linkedin.com/in/votre-profil")
        elif not ANTHROPIC_API_KEY:
            st.error("Clé API manquante.")
        else:
            with st.status("Récupération...", expanded=False) as s:
                data = extract_linkedin_profile(url.strip(), ANTHROPIC_API_KEY)
            if data.get("error") == "login_required":
                st.warning("Profil non accessible. Rendez-le public ou saisissez manuellement.")
                st.session_state.phase = "manual_input"
                st.rerun()
            elif "error" in data:
                st.error(data["error"])
            else:
                s.update(label="Profil récupéré ✓", state="complete")
                st.session_state.linkedin_data = data
                st.session_state.phase = "confirming"
                st.rerun()

    if manual_btn:
        st.session_state.phase = "manual_input"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1b — Saisie manuelle
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "manual_input":
    st.markdown("""<div class="nav"><span class="nav-logo">✦ &nbsp;Repositionnement IA</span></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Saisissez votre profil")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        nom = c1.text_input("Prénom Nom")
        poste = c2.text_input("Poste actuel")
        c3, c4 = st.columns(2)
        secteur = c3.text_input("Secteur d'activité")
        exp = c4.number_input("Années d'expérience", min_value=0, max_value=50, value=5)
        competences = st.text_input("Compétences (séparées par des virgules)", placeholder="Ex : gestion de projet, data analysis, communication")
        formation = st.text_input("Formation", placeholder="Ex : Master Marketing, ESSEC")
        objectif = st.text_area("Votre objectif avec l'IA", placeholder="Ex : Je veux évoluer vers un rôle de consultant IA dans mon secteur...")
        submitted = st.form_submit_button("Continuer →", type="primary")

    if submitted:
        if not all([nom, poste, secteur, competences, formation, objectif]):
            st.error("Complétez tous les champs.")
        else:
            st.session_state.linkedin_data = {
                "nom": nom, "poste_actuel": poste, "annees_experience": int(exp),
                "secteur": secteur, "competences": [c.strip() for c in competences.split(",") if c.strip()],
                "formation": formation, "objectif_reconversion": objectif,
                "resume": "", "entreprise_actuelle": "", "localisation": "",
            }
            st.session_state.phase = "confirming"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Confirmation
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "confirming":
    data = st.session_state.linkedin_data

    st.markdown("""<div class="nav"><span class="nav-logo">✦ &nbsp;Repositionnement IA</span></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="li-card">
        <div class="li-name">{data.get('nom', '—')}</div>
        <div class="li-title">{data.get('poste_actuel', '') or ''}{' · ' + data.get('entreprise_actuelle','') if data.get('entreprise_actuelle') else ''}</div>
        <div class="li-meta">{data.get('secteur', '')} {'· ' + data.get('localisation','') if data.get('localisation') else ''} {'· ' + str(data.get('annees_experience','')) + ' ans' if data.get('annees_experience') else ''}</div>
    </div>
    """, unsafe_allow_html=True)

    missing = [f for f, k in [("poste actuel","poste_actuel"),("compétences","competences"),("formation","formation")] if not data.get(k)]
    if missing:
        st.info(f"LinkedIn masque certains champs — complétez : {', '.join(missing)}.")

    st.markdown("**Vérifiez et complétez vos informations**")
    with st.form("confirm_form"):
        c1, c2 = st.columns(2)
        nom = c1.text_input("Prénom Nom", value=data.get("nom", ""))
        poste = c2.text_input("Poste actuel", value=data.get("poste_actuel", ""))
        c3, c4 = st.columns(2)
        secteur = c3.text_input("Secteur", value=data.get("secteur", ""))
        exp = c4.number_input("Années d'expérience", min_value=0, max_value=50, value=int(data.get("annees_experience", 5)))
        competences_str = st.text_input("Compétences (séparées par des virgules)", value=", ".join(data.get("competences", [])))
        formation = st.text_input("Formation", value=data.get("formation", ""))
        objectif = st.text_area(
            "Votre objectif avec l'IA",
            placeholder="Ex : Je veux évoluer vers un rôle de consultant IA dans mon secteur, ou changer complètement de domaine...",
            help="Cette info n'est pas sur LinkedIn — une phrase suffit.",
        )
        confirm = st.form_submit_button("Lancer l'analyse →", type="primary", use_container_width=True)

    if confirm:
        if not objectif.strip():
            st.error("Décrivez votre objectif avec l'IA — c'est essentiel pour la recommandation.")
        else:
            st.session_state.profile = UserProfile(
                nom=nom, poste_actuel=poste, annees_experience=int(exp),
                secteur=secteur,
                competences=[c.strip() for c in competences_str.split(",") if c.strip()],
                formation=formation, objectif_reconversion=objectif,
            )
            st.session_state.phase = "analyzing"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Analyse
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "analyzing":
    profile = st.session_state.profile

    st.markdown("""<div class="nav"><span class="nav-logo">✦ &nbsp;Repositionnement IA</span></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"**Analyse de {profile.nom} en cours...**")
    st.caption("Claude Opus examine votre parcours et croise vos compétences avec les opportunités IA.")

    client = get_client()
    intro_prompt = f"""Profil de {profile.nom} : {profile.annees_experience} ans en tant que {profile.poste_actuel} dans le {profile.secteur}. Objectif : {profile.objectif_reconversion}.
Écris un message d'accueil personnalisé et chaleureux (3 phrases) : reconnais son parcours, montre de l'enthousiasme pour sa démarche, annonce que l'analyse est en cours."""

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full = ""
        with client.messages.stream(
            model=MODEL, max_tokens=400, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": intro_prompt}],
            thinking={"type": "adaptive"},
        ) as stream:
            for chunk in stream.text_stream:
                full += chunk
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)

    with st.spinner("Analyse en cours..."):
        result = run_analysis(profile)
        st.session_state.result = result

    init_database()
    profile_id = save_profile(build_profile_dict(profile, result))
    st.session_state.profile_id = profile_id
    st.session_state.phase = "done"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Résultats
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "done":
    st.markdown("""<div class="nav"><span class="nav-logo">✦ &nbsp;Repositionnement IA</span></div>""", unsafe_allow_html=True)
    render_results(st.session_state.profile, st.session_state.result, st.session_state.profile_id)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING — Dashboard gamifié
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "training":
    st.markdown("""<div class="nav"><span class="nav-logo">✦ &nbsp;Repositionnement IA</span><span class="nav-right">Mode entraînement 🎮</span></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    render_training_dashboard(st.session_state.profile, st.session_state.result)

# ══════════════════════════════════════════════════════════════════════════════
# MISSION RUNNER
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "mission":
    st.markdown("""<div class="nav"><span class="nav-logo">✦ &nbsp;Repositionnement IA</span><span class="nav-right">Mission en cours</span></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    render_mission_runner(st.session_state.profile, st.session_state.result)
