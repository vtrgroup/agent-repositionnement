"""Interface web Streamlit — Agent de Repositionnement IA (entrée LinkedIn)."""

import json
import streamlit as st
import anthropic

from config import MODEL, SYSTEM_PROMPT, ANTHROPIC_API_KEY, AI_ROLES
from agent import UserProfile, AnalysisResult, build_profile_dict
from database import save_profile, init_database, get_profile
from linkedin import extract_linkedin_profile, extract_from_text, validate_linkedin_url

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quel est votre rôle dans l'IA ?",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Global — fond blanc, texte ardoise ── */
[data-testid="stAppViewContainer"] { background:#fafafa; }
[data-testid="stSidebar"]          { background:#f1f5f9; }
body, h1,h2,h3,h4,p,li,label,div  { color:#1e293b; font-family:'Inter',sans-serif; }

/* ── Navbar fine ── */
.navbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:18px 0 24px; margin-bottom:0;
}
.navbar-brand { font-size:17px; font-weight:700; color:#1e293b; }
.navbar-sub   { font-size:13px; color:#94a3b8; font-weight:400; }

/* ── Hero ── */
.hero {
    padding:72px 24px 64px; text-align:center; max-width:720px; margin:0 auto;
}
.hero-eyebrow {
    display:inline-block; background:#ede9fe; color:#7c3aed;
    border-radius:100px; padding:5px 16px; font-size:12px; font-weight:600;
    letter-spacing:.4px; margin-bottom:24px; text-transform:uppercase;
}
.hero h1 {
    font-size:clamp(32px,4.5vw,52px); font-weight:800; line-height:1.15;
    color:#0f172a; margin:0 0 20px; letter-spacing:-1px;
}
.hero h1 em { font-style:normal; color:#7c3aed; }
.hero-sub {
    font-size:17px; color:#475569; max-width:520px;
    margin:0 auto 36px; line-height:1.75;
}
.hero-note { font-size:13px; color:#94a3b8; margin-top:12px; }

/* ── Features — 3 colonnes légères ── */
.features { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:52px 0; }
.feat {
    background:#fff; border:1px solid #e2e8f0;
    border-radius:14px; padding:24px 20px;
    transition:box-shadow .2s;
}
.feat:hover { box-shadow:0 4px 20px rgba(0,0,0,.06); }
.feat-icon  { font-size:24px; margin-bottom:12px; }
.feat-title { font-size:15px; font-weight:700; color:#0f172a; margin-bottom:6px; }
.feat-desc  { font-size:13px; color:#64748b; line-height:1.65; }

/* ── Steps ── */
.steps-wrap { max-width:680px; margin:0 auto 52px; }
.step-row {
    display:flex; align-items:flex-start; gap:20px; margin-bottom:28px;
}
.step-dot {
    flex-shrink:0; width:32px; height:32px; border-radius:50%;
    background:#7c3aed; color:#fff; font-weight:700; font-size:14px;
    display:flex; align-items:center; justify-content:center; margin-top:2px;
}
.step-body {}
.step-title2 { font-size:15px; font-weight:700; color:#0f172a; margin-bottom:3px; }
.step-desc   { font-size:14px; color:#64748b; line-height:1.6; }

/* ── Rôles pills ── */
.roles-grid { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:24px 0 52px; }
.role-pill {
    background:#f5f3ff; border:1px solid #ddd6fe;
    color:#6d28d9; border-radius:100px; padding:7px 16px; font-size:13px; font-weight:500;
}

/* ── Section title ── */
.section-title {
    text-align:center; font-size:24px; font-weight:800;
    color:#0f172a; margin-bottom:8px; letter-spacing:-.5px;
}
.section-sub { text-align:center; font-size:15px; color:#64748b; margin-bottom:32px; }

/* ── Divider doux ── */
.soft-divider { border:none; border-top:1px solid #f1f5f9; margin:48px 0; }

/* ── Bottom CTA ── */
.bottom-cta {
    background:#f5f3ff; border:1px solid #ddd6fe;
    border-radius:20px; padding:52px 40px; text-align:center; margin-bottom:32px;
}
.bottom-cta h2 { font-size:28px; font-weight:800; color:#0f172a; margin-bottom:10px; }
.bottom-cta p  { color:#64748b; font-size:15px; margin-bottom:28px; }

/* ── Results ── */
.score-wrap { text-align:center; padding:28px 20px; border-radius:16px; color:white; }
.score-num  { font-size:80px; font-weight:800; line-height:1; }
.score-lbl  { font-size:15px; opacity:.9; margin-top:6px; }
.role-badge {
    background:#f5f3ff; border:2px solid #7c3aed;
    border-radius:12px; padding:14px 20px;
    font-size:19px; font-weight:700; color:#5b21b6; text-align:center;
}
.step {
    background:#f8fafc; border-left:3px solid #7c3aed;
    border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;
}
.step-title { font-weight:700; color:#0f172a; }
.step-meta  { color:#64748b; font-size:13px; margin-top:3px; }
.tag     { display:inline-block; background:#ede9fe; color:#5b21b6;
           border-radius:20px; padding:4px 12px; margin:3px; font-size:13px; }
.tag-red { background:#fef2f2; color:#b91c1c; }
.li-card {
    background:#f0f9ff; border:1px solid #bae6fd;
    border-radius:12px; padding:18px 22px; margin-bottom:12px;
}
.li-name  { font-size:20px; font-weight:700; color:#0f172a; }
.li-title { font-size:14px; color:#0369a1; margin-top:3px; }
.li-meta  { font-size:13px; color:#64748b; margin-top:6px; }
.saved { background:#f0fdf4; border:1px solid #bbf7d0;
         border-radius:10px; padding:12px 18px; color:#166534; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase": "landing",        # landing → url_input → confirming → analyzing → done
        "linkedin_data": None,     # dict brut extrait de LinkedIn
        "profile": None,           # UserProfile validé
        "result": None,            # AnalysisResult
        "profile_id": None,
        "messages": [],            # historique chat UI
        "api_history": [],         # historique API Claude
        "api_key": ANTHROPIC_API_KEY,
        "linkedin_url": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Client ────────────────────────────────────────────────────────────────────
def get_client() -> anthropic.Anthropic:
    key = ANTHROPIC_API_KEY
    if not key:
        st.error("⚠️ Clé API Anthropic manquante — ajoutez `ANTHROPIC_API_KEY=sk-ant-...` dans le fichier `.env` du projet.")
        st.stop()
    return anthropic.Anthropic(api_key=key)

# ── LLM helpers ───────────────────────────────────────────────────────────────
def stream_chat(prompt: str, max_tokens: int = 1024) -> str:
    """Appelle l'API en streaming et affiche dans le chat."""
    client = get_client()
    st.session_state.api_history.append({"role": "user", "content": prompt})
    full = ""
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        with client.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=st.session_state.api_history,
            thinking={"type": "adaptive"},
        ) as stream:
            for chunk in stream.text_stream:
                full += chunk
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)
    st.session_state.messages.append({"role": "assistant", "content": full})
    st.session_state.api_history.append({"role": "assistant", "content": full})
    return full


def run_analysis(profile: UserProfile) -> AnalysisResult:
    client = get_client()
    roles_list = "\n".join(f"- {r}" for r in AI_ROLES)
    prompt = f"""Analyse ce profil professionnel et retourne UNIQUEMENT un JSON valide.

PROFIL :
- Nom : {profile.nom}
- Poste : {profile.poste_actuel}
- Expérience : {profile.annees_experience} ans
- Secteur : {profile.secteur}
- Compétences : {", ".join(profile.competences)}
- Formation : {profile.formation}
- Objectif : {profile.objectif_reconversion}

RÔLES IA DISPONIBLES :
{roles_list}

FORMAT JSON ATTENDU :
{{
  "role_cible": "rôle parmi la liste",
  "score": entier_0_100,
  "score_justification": "2 phrases expliquant le score",
  "competences_transferables": ["c1","c2","c3"],
  "competences_a_acquerir": ["c1","c2","c3"],
  "plan_formation": [
    {{"etape":"1. Titre","duree":"X semaines","ressources":"Plateforme / cours"}},
    {{"etape":"2. Titre","duree":"X semaines","ressources":"Plateforme / cours"}},
    {{"etape":"3. Titre","duree":"X mois","ressources":"Plateforme / cours"}},
    {{"etape":"4. Titre","duree":"X mois","ressources":"Plateforme / cours"}}
  ],
  "salaire_estime": "fourchette salariale en France",
  "perspectives": "évolutions possibles en 2-3 phrases",
  "message_encouragement": "message motivant personnalisé 2-3 phrases"
}}
UNIQUEMENT le JSON."""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system="Tu es expert RH et formation IA. Retourne uniquement du JSON valide.",
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"},
    )
    import re
    raw = resp.content[-1].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    return AnalysisResult(**json.loads(raw))

# ── Render résultats ──────────────────────────────────────────────────────────
def render_results(profile: UserProfile, result: AnalysisResult, profile_id: str):
    st.divider()
    st.markdown("## 📊 Résultats de votre analyse de repositionnement")

    # Score color
    if result.score >= 70:
        grad = "linear-gradient(135deg,#22c55e,#16a34a)"
    elif result.score >= 40:
        grad = "linear-gradient(135deg,#f59e0b,#d97706)"
    else:
        grad = "linear-gradient(135deg,#ef4444,#dc2626)"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="score-wrap" style="background:{grad};">
            <div class="score-num">{result.score}</div>
            <div class="score-lbl">Score de compatibilité / 100</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="role-badge">🎯 {result.role_cible}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(result.score_justification)
        st.success(f"💰 **Salaire estimé :** {result.salaire_estime}")

    with col2:
        st.markdown("#### ✅ Compétences transférables")
        st.markdown(
            " ".join(f'<span class="tag">{c}</span>' for c in result.competences_transferables),
            unsafe_allow_html=True,
        )

        st.markdown("#### 📚 Compétences à acquérir")
        st.markdown(
            " ".join(f'<span class="tag tag-red">{c}</span>' for c in result.competences_a_acquerir),
            unsafe_allow_html=True,
        )

        st.markdown("#### 🗺️ Plan de formation personnalisé")
        for step in result.plan_formation:
            st.markdown(f"""
            <div class="step">
                <div class="step-title">{step['etape']}</div>
                <div class="step-meta">⏱ {step['duree']} &nbsp;|&nbsp; 📖 {step['ressources']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### 🚀 Perspectives de carrière")
    st.markdown(result.perspectives)

    st.markdown(f"""
    <div style="background:#fef9c3;border:1px solid #fcd34d;border-radius:10px;padding:14px 18px;margin-top:12px;">
        💬 <em>{result.message_encouragement}</em>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="saved">✅ Profil sauvegardé — ID : <code>{profile_id}</code> &nbsp;|&nbsp; <code>data/profiles.json</code></div>',
        unsafe_allow_html=True,
    )

    with st.expander("🗂 Données JSON complètes"):
        st.json(get_profile(profile_id))

    if st.button("🔄 Analyser un autre profil", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.session_state.phase != "landing":
        st.markdown("## 🎯 Rôles IA disponibles")
        for role in AI_ROLES:
            st.markdown(f"- {role}")
        st.divider()
    if st.button("🔄 Recommencer", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════
if st.session_state.phase == "landing":

    # Navbar
    st.markdown("""
    <div class="navbar">
        <span class="navbar-brand">✦ &nbsp;Repositionnement IA</span>
        <span class="navbar-sub">Propulsé par Claude Opus 4.6</span>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Bilan professionnel gratuit</div>
        <h1>Et si l'IA était<br><em>votre</em> prochaine étape ?</h1>
        <p class="hero-sub">
            Collez votre profil LinkedIn. En moins d'une minute, découvrez le rôle IA
            qui correspond à votre parcours, et le chemin concret pour y accéder.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1.2, 2, 1.2])
    with col_c:
        if st.button("Découvrir mon rôle dans l'IA →", type="primary", use_container_width=True):
            st.session_state.phase = "url_input"
            st.rerun()
        st.markdown(
            "<p class='hero-note' style='text-align:center;'>Gratuit · Aucune inscription · 60 secondes</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div class="features">
        <div class="feat">
            <div class="feat-icon">🔍</div>
            <div class="feat-title">Votre meilleur match IA</div>
            <div class="feat-desc">L'agent analyse vos compétences actuelles et les croise avec les rôles IA les plus accessibles depuis votre profil.</div>
        </div>
        <div class="feat">
            <div class="feat-icon">🗺️</div>
            <div class="feat-title">Un plan, pas des conseils</div>
            <div class="feat-desc">4 étapes concrètes avec des durées réalistes et des ressources précises. Pas de vague orientation — un vrai chemin.</div>
        </div>
        <div class="feat">
            <div class="feat-icon">💬</div>
            <div class="feat-title">Personnalisé pour vous</div>
            <div class="feat-desc">Chaque analyse est générée en temps réel par Claude Opus, en tenant compte de votre secteur, vos années d'expérience et votre objectif.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

    # Comment ça marche
    st.markdown('<p class="section-title">Comment ça marche ?</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Trois étapes, pas une de plus.</p>', unsafe_allow_html=True)

    col_steps, _ = st.columns([2, 1])
    with col_steps:
        st.markdown("""
        <div class="steps-wrap">
            <div class="step-row">
                <div class="step-dot">1</div>
                <div class="step-body">
                    <div class="step-title2">Partagez votre URL LinkedIn</div>
                    <div class="step-desc">L'agent récupère automatiquement votre profil public — pas besoin de remplir un formulaire.</div>
                </div>
            </div>
            <div class="step-row">
                <div class="step-dot">2</div>
                <div class="step-body">
                    <div class="step-title2">Vérifiez vos informations</div>
                    <div class="step-desc">Confirmez ce qui a été extrait et précisez ce que vous cherchez dans l'IA — une phrase suffit.</div>
                </div>
            </div>
            <div class="step-row">
                <div class="step-dot">3</div>
                <div class="step-body">
                    <div class="step-title2">Recevez votre analyse complète</div>
                    <div class="step-desc">Rôle recommandé, score de compatibilité, compétences transférables et plan de formation détaillé.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

    # Rôles
    st.markdown('<p class="section-title">Les rôles que nous analysons</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Parmi les métiers IA les plus accessibles aux profils non-tech.</p>', unsafe_allow_html=True)
    roles_html = " ".join(f'<span class="role-pill">{r}</span>' for r in AI_ROLES)
    st.markdown(f'<div class="roles-grid">{roles_html}</div>', unsafe_allow_html=True)

    # Bottom CTA
    st.markdown("""
    <div class="bottom-cta">
        <h2>Prêt à voir où vous en êtes ?</h2>
        <p>L'analyse prend moins d'une minute. Votre profil LinkedIn fait le travail.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l2, col_c2, col_r2 = st.columns([1.2, 2, 1.2])
    with col_c2:
        if st.button("Commencer l'analyse →", type="primary", use_container_width=True):
            st.session_state.phase = "url_input"
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — Saisie de l'URL LinkedIn (scraping automatique)
# ═══════════════════════════════════════════════════════════════
if st.session_state.phase == "url_input":
    st.markdown("""
    <div style="max-width:600px;margin:40px auto 8px;">
        <p style="font-size:13px;color:#94a3b8;margin-bottom:8px;">Étape 1 sur 2</p>
        <h2 style="font-size:26px;font-weight:800;color:#0f172a;margin:0 0 8px;">
            Votre profil LinkedIn
        </h2>
        <p style="color:#64748b;font-size:15px;margin:0 0 28px;">
            L'agent récupère automatiquement vos informations — pas de saisie manuelle.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        url = st.text_input(
            "URL LinkedIn",
            placeholder="https://www.linkedin.com/in/votre-profil",
            label_visibility="collapsed",
        )
    with col_btn:
        go_btn = st.button("Analyser →", type="primary", use_container_width=True)

    st.markdown(
        "<p style='color:#94a3b8;font-size:13px;margin-top:6px;'>"
        "Profil doit être public sur LinkedIn (Paramètres → Visibilité → Public)"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    manual_btn = st.button("Saisir mes informations manuellement", use_container_width=False)

    if go_btn:
        if not url.strip():
            st.error("Veuillez entrer une URL LinkedIn.")
        elif not validate_linkedin_url(url.strip()):
            st.error("❌ URL invalide. Format attendu : https://www.linkedin.com/in/votre-profil")
        elif not ANTHROPIC_API_KEY:
            st.error("⚠️ Clé API Anthropic manquante — ajoutez `ANTHROPIC_API_KEY=sk-ant-...` dans le fichier `.env`.")
        else:
            st.session_state.linkedin_url = url.strip()

            with st.status("🔍 Récupération du profil LinkedIn...", expanded=True) as status_box:
                st.write("Connexion à LinkedIn...")
                data = extract_linkedin_profile(url.strip(), ANTHROPIC_API_KEY)

                if data.get("error") == "login_required":
                    status_box.update(label="⚠️ LinkedIn demande une connexion", state="error")
                    st.warning(
                        "LinkedIn protège ce profil et nécessite une authentification. "
                        "Passez en saisie manuelle ou rendez votre profil public."
                    )
                    st.session_state.phase = "manual_input"
                    st.rerun()
                elif "error" in data:
                    status_box.update(label="❌ Erreur d'extraction", state="error")
                    st.error(data["error"])
                else:
                    st.write(f"✅ Profil de **{data.get('nom', '?')}** extrait avec succès !")
                    status_box.update(label="✅ Profil récupéré !", state="complete")
                    st.session_state.linkedin_data = data
                    st.session_state.phase = "confirming"
                    st.rerun()

    if manual_btn:
        st.session_state.phase = "manual_input"
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PHASE 1b — Saisie manuelle
# ═══════════════════════════════════════════════════════════════
elif st.session_state.phase == "manual_input":
    st.markdown("### ✏️ Saisissez votre profil manuellement")
    st.markdown("[← Retour à l'import LinkedIn](#)", help="Cliquez sur 'Recommencer' dans la barre latérale")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        nom = c1.text_input("Prénom Nom *")
        poste = c2.text_input("Poste actuel *")
        c3, c4 = st.columns(2)
        secteur = c3.text_input("Secteur d'activité *")
        exp = c4.number_input("Années d'expérience *", min_value=0, max_value=50, value=5)
        competences = st.text_input(
            "Compétences principales (séparées par des virgules) *",
            placeholder="Ex : gestion de projet, data analysis, communication",
        )
        formation = st.text_input("Formation *", placeholder="Ex : Master Marketing, ESSEC")
        objectif = st.text_area(
            "Votre objectif avec l'IA *",
            placeholder="Ex : Je veux rester dans mon secteur mais avec un rôle plus centré sur l'IA...",
        )
        submitted = st.form_submit_button("Analyser mon profil →", type="primary")

    if submitted:
        if not all([nom, poste, secteur, competences, formation, objectif]):
            st.error("Veuillez remplir tous les champs obligatoires.")
        else:
            st.session_state.linkedin_data = {
                "nom": nom,
                "poste_actuel": poste,
                "annees_experience": int(exp),
                "secteur": secteur,
                "competences": [c.strip() for c in competences.split(",") if c.strip()],
                "formation": formation,
                "objectif_reconversion": objectif,
                "resume": "",
                "entreprise_actuelle": "",
                "localisation": "",
            }
            st.session_state.phase = "confirming"
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — Confirmation du profil extrait
# ═══════════════════════════════════════════════════════════════
elif st.session_state.phase == "confirming":
    data = st.session_state.linkedin_data

    st.markdown("### ✅ Profil extrait — Vérifiez et complétez")
    st.markdown(f"""
    <div class="li-card">
        <div class="li-name">👤 {data.get('nom', 'N/A')}</div>
        <div class="li-title">💼 {data.get('poste_actuel', 'N/A')} · {data.get('entreprise_actuelle', '')}</div>
        <div class="li-meta">
            🏢 {data.get('secteur', 'N/A')} &nbsp;|&nbsp;
            📍 {data.get('localisation', '')} &nbsp;|&nbsp;
            ⏳ {data.get('annees_experience', '?')} ans d'expérience
        </div>
        {"<div style='margin-top:10px;font-size:13px;color:#334155;'>" + data.get('resume','') + "</div>" if data.get('resume') else ""}
    </div>
    """, unsafe_allow_html=True)

    # Signaler les champs obfusqués par LinkedIn
    missing_fields = []
    if not data.get("poste_actuel"):
        missing_fields.append("poste actuel")
    if not data.get("competences"):
        missing_fields.append("compétences")
    if not data.get("formation"):
        missing_fields.append("formation")

    if missing_fields:
        st.warning(
            f"⚠️ **LinkedIn masque certains champs pour les non-connectés** : "
            f"{', '.join(missing_fields)}. "
            f"Complétez-les ci-dessous pour obtenir une analyse précise."
        )

    st.markdown("**Vérifiez les informations extraites — modifiez si nécessaire :**")

    with st.form("confirm_form"):
        c1, c2 = st.columns(2)
        nom = c1.text_input("Prénom Nom", value=data.get("nom", ""))
        poste = c2.text_input("Poste actuel", value=data.get("poste_actuel", ""))
        c3, c4 = st.columns(2)
        secteur = c3.text_input("Secteur", value=data.get("secteur", ""))
        exp = c4.number_input("Années d'expérience", min_value=0, max_value=50,
                               value=int(data.get("annees_experience", 5)))
        competences_str = st.text_input(
            "Compétences (séparées par des virgules)",
            value=", ".join(data.get("competences", [])),
        )
        formation = st.text_input("Formation", value=data.get("formation", ""))
        objectif = st.text_area(
            "Votre objectif avec l'IA *",
            placeholder="Ex : Je veux piloter des projets IA dans mon secteur, évoluer vers un rôle de consultant...",
            help="Cette info n'est pas sur LinkedIn — décrivez ce qui vous attire dans l'IA.",
        )

        col_a, col_b = st.columns(2)
        confirm = col_a.form_submit_button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True)
        col_b.form_submit_button("↩️ Changer d'URL", use_container_width=True)

    if confirm:
        if not objectif.strip():
            st.error("⚠️ Décrivez votre objectif avec l'IA — c'est essentiel pour la recommandation.")
        else:
            st.session_state.profile = UserProfile(
                nom=nom,
                poste_actuel=poste,
                annees_experience=int(exp),
                secteur=secteur,
                competences=[c.strip() for c in competences_str.split(",") if c.strip()],
                formation=formation,
                objectif_reconversion=objectif,
            )
            st.session_state.phase = "analyzing"
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PHASE 3 — Analyse IA
# ═══════════════════════════════════════════════════════════════
elif st.session_state.phase == "analyzing":
    profile = st.session_state.profile

    st.markdown(f"""
    <div style="background:#f0f9ff;border:1px solid #7dd3fc;border-radius:12px;padding:20px 24px;">
        <strong>⚙️ Analyse du profil de {profile.nom} en cours...</strong><br>
        <span style="color:#64748b;">Croisement des compétences avec les opportunités IA du marché</span>
    </div>
    """, unsafe_allow_html=True)

    # Enrichissement conversationnel via Claude
    intro_prompt = f"""Le profil de {profile.nom} vient d'être extrait de LinkedIn.
Voici son parcours : {profile.annees_experience} ans en tant que {profile.poste_actuel} dans le {profile.secteur}.
Son objectif : {profile.objectif_reconversion}.

Écris un court message d'accueil personnalisé (3-4 phrases) qui :
1. Reconnaît son parcours et ses compétences clés
2. Exprime de l'enthousiasme pour son projet de reconversion
3. Annonce que l'analyse est en cours"""

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        client = get_client()
        full = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": intro_prompt}],
            thinking={"type": "adaptive"},
        ) as stream:
            for chunk in stream.text_stream:
                full += chunk
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)

    with st.spinner("🧠 Analyse approfondie avec thinking adaptatif..."):
        result = run_analysis(profile)
        st.session_state.result = result

    init_database()
    profile_id = save_profile(build_profile_dict(profile, result))
    st.session_state.profile_id = profile_id
    st.session_state.phase = "done"
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# PHASE 4 — Résultats
# ═══════════════════════════════════════════════════════════════
elif st.session_state.phase == "done":
    render_results(
        st.session_state.profile,
        st.session_state.result,
        st.session_state.profile_id,
    )
