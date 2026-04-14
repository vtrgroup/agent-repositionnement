"""Interface web Streamlit — Agent de Repositionnement IA."""

import json
import re
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
    max-width:920px !important;
    padding:0 40px 80px !important;
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
.hero { padding:40px 0 16px; text-align:center !important; }
.hero * { text-align:center !important; }
.hero-eyebrow {
    display:inline-block; background:#f0eeff; color:#5b5bd6;
    border-radius:100px; padding:5px 14px; font-size:11px; font-weight:700;
    letter-spacing:.8px; text-transform:uppercase; margin-bottom:24px;
}
.hero h1 {
    font-size:54px !important; font-weight:900 !important; line-height:1.1 !important;
    letter-spacing:-2.5px !important; color:#0f172a !important; margin:0 0 20px !important;
}
.hero h1 span { color:#5b5bd6 !important; }
.hero-sub {
    font-size:17px; color:#64748b; line-height:1.7;
    max-width:480px; margin-left:auto !important; margin-right:auto !important;
    margin-bottom:12px; font-weight:400; display:block;
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

JSON :
{{
  "role_cible": "rôle parmi la liste",
  "score": entier_0_100,
  "score_justification": "2 phrases",
  "competences_transferables": ["c1","c2","c3"],
  "competences_a_acquerir": ["c1","c2","c3"],
  "plan_formation": [
    {{"etape":"1. Titre","duree":"X semaines","ressources":"Plateforme"}},
    {{"etape":"2. Titre","duree":"X semaines","ressources":"Plateforme"}},
    {{"etape":"3. Titre","duree":"X mois","ressources":"Plateforme"}},
    {{"etape":"4. Titre","duree":"X mois","ressources":"Plateforme"}}
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
    raw = resp.content[-1].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    return AnalysisResult(**json.loads(raw))

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
    if st.button("Analyser un autre profil →"):
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
        <div class="hero-eyebrow">Bilan professionnel gratuit</div>
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
