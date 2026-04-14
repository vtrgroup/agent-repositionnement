"""Agent de repositionnement professionnel vers l'IA."""

import json
from typing import Any, Dict, List, Tuple

import anthropic
from pydantic import BaseModel

from config import MODEL, SYSTEM_PROMPT, AI_ROLES, ANTHROPIC_API_KEY


class UserProfile(BaseModel):
    nom: str
    poste_actuel: str
    annees_experience: int
    secteur: str
    competences: list[str]
    formation: str
    objectif_reconversion: str


class AnalysisResult(BaseModel):
    role_cible: str
    score: int  # 0-100
    score_justification: str
    competences_transferables: list[str]
    competences_a_acquerir: list[str]
    plan_formation: list[dict[str, str]]  # [{"etape": ..., "duree": ..., "ressources": ...}]
    salaire_estime: str
    perspectives: str
    message_encouragement: str


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def stream_message(messages: List[Dict], max_tokens: int = 1024) -> str:
    """Envoie un message en streaming et retourne le texte complet."""
    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,
        thinking={"type": "adaptive"},
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
    print()
    return full_text


def collect_profile_conversationally() -> Tuple[UserProfile, List[Dict]]:
    """Collecte le profil utilisateur via une conversation naturelle."""
    conversation_history: list[dict] = []

    # Étape 1 : Message d'accueil
    welcome_prompt = """Accueille chaleureusement l'utilisateur. Présente-toi comme un expert en
repositionnement professionnel vers l'IA. Explique en 2-3 phrases ce que tu vas faire pour lui
(analyser son profil, identifier le meilleur rôle IA, créer un plan de formation personnalisé).
Puis demande-lui de commencer par se présenter : son prénom, son poste actuel et son secteur d'activité."""

    print("\n")
    conversation_history.append({"role": "user", "content": welcome_prompt})
    response = stream_message(conversation_history)
    conversation_history.append({"role": "assistant", "content": response})

    # Étape 2 : Collecte des infos de base
    user_intro = input("\nVous : ").strip()
    conversation_history.append({"role": "user", "content": user_intro})

    followup_prompt = f"""L'utilisateur vient de se présenter. Remercie-le et pose-lui maintenant
3 questions précises pour compléter son profil :
1. Combien d'années d'expérience a-t-il dans son domaine ?
2. Quelles sont ses 3-5 compétences principales (techniques et soft skills) ?
3. Quelle est sa formation initiale (diplôme + domaine) ?

Sois naturel et conversationnel."""

    response = stream_message(conversation_history + [{"role": "user", "content": followup_prompt}])
    conversation_history.append({"role": "assistant", "content": response})

    # Étape 3 : Réponse de l'utilisateur
    user_answer = input("\nVous : ").strip()
    conversation_history.append({"role": "user", "content": user_answer})

    motivation_prompt = """Parfait ! Dernière question importante : quel est son objectif avec
l'IA ? Est-ce qu'il souhaite plutôt rester dans son secteur mais avec un rôle plus tech,
changer complètement de domaine, ou autre chose ? Qu'est-ce qui le motive dans cette
reconversion vers l'IA ?"""

    response = stream_message(conversation_history + [{"role": "user", "content": motivation_prompt}])
    conversation_history.append({"role": "assistant", "content": response})

    # Étape 4 : Motivation
    user_motivation = input("\nVous : ").strip()
    conversation_history.append({"role": "user", "content": user_motivation})

    # Étape 5 : Extraction structurée du profil
    extract_prompt = f"""À partir de toute la conversation précédente, extrais les informations
du profil de l'utilisateur et retourne-les UNIQUEMENT en JSON valide avec ce format exact :

{{
  "nom": "prénom ou nom mentionné",
  "poste_actuel": "poste actuel",
  "annees_experience": nombre_entier,
  "secteur": "secteur d'activité",
  "competences": ["competence1", "competence2", "competence3"],
  "formation": "diplôme et domaine",
  "objectif_reconversion": "objectif et motivation"
}}

Si une information n'est pas disponible, utilise une valeur par défaut raisonnable.
Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""

    extract_response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="Tu es un extracteur de données. Tu retournes uniquement du JSON valide.",
        messages=conversation_history + [{"role": "user", "content": extract_prompt}],
    )

    raw_json = extract_response.content[0].text.strip()
    # Nettoyer si le modèle a quand même ajouté des backticks
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    profile_data = json.loads(raw_json)
    profile = UserProfile(**profile_data)

    return profile, conversation_history


def analyze_profile(profile: UserProfile, conversation_history: List[Dict]) -> AnalysisResult:
    """Analyse le profil et génère le rôle cible + plan de formation."""

    roles_list = "\n".join(f"- {r}" for r in AI_ROLES)

    analysis_prompt = f"""Analyse ce profil professionnel et génère une recommandation complète.

PROFIL :
- Nom : {profile.nom}
- Poste actuel : {profile.poste_actuel}
- Années d'expérience : {profile.annees_experience}
- Secteur : {profile.secteur}
- Compétences : {", ".join(profile.competences)}
- Formation : {profile.formation}
- Objectif : {profile.objectif_reconversion}

RÔLES IA DISPONIBLES :
{roles_list}

Retourne UNIQUEMENT un JSON valide avec ce format :
{{
  "role_cible": "le meilleur rôle IA parmi la liste",
  "score": nombre_entre_0_et_100,
  "score_justification": "explication du score en 2 phrases",
  "competences_transferables": ["comp1", "comp2", "comp3"],
  "competences_a_acquerir": ["comp1", "comp2", "comp3"],
  "plan_formation": [
    {{"etape": "1. Titre étape", "duree": "X semaines/mois", "ressources": "Cours/plateforme recommandée"}},
    {{"etape": "2. Titre étape", "duree": "X semaines/mois", "ressources": "Cours/plateforme recommandée"}},
    {{"etape": "3. Titre étape", "duree": "X semaines/mois", "ressources": "Cours/plateforme recommandée"}},
    {{"etape": "4. Titre étape", "duree": "X semaines/mois", "ressources": "Cours/plateforme recommandée"}}
  ],
  "salaire_estime": "fourchette salariale estimée en France",
  "perspectives": "évolutions de carrière possibles en 2-3 phrases",
  "message_encouragement": "message personnalisé et motivant de 2-3 phrases"
}}

Le score doit refléter la facilité de transition (100 = transition très naturelle, 0 = très difficile).
Réponds UNIQUEMENT avec le JSON."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system="Tu es un expert RH et formation IA. Tu retournes uniquement du JSON valide.",
        messages=[{"role": "user", "content": analysis_prompt}],
        thinking={"type": "adaptive"},
    )

    raw_json = response.content[-1].text.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    result_data = json.loads(raw_json)
    return AnalysisResult(**result_data)


def present_results(profile: UserProfile, result: AnalysisResult, conversation_history: List[Dict]) -> None:
    """Présente les résultats de manière conversationnelle et engageante."""

    plan_text = "\n".join(
        f"  - {step['etape']} ({step['duree']}) : {step['ressources']}"
        for step in result.plan_formation
    )

    presentation_prompt = f"""Présente les résultats de l'analyse à {profile.nom} de manière
enthousiaste et structurée. Voici les données à présenter :

RÉSULTATS :
- Rôle cible recommandé : {result.role_cible}
- Score de compatibilité : {result.score}/100
- Justification : {result.score_justification}
- Compétences transférables : {", ".join(result.competences_transferables)}
- Compétences à acquérir : {", ".join(result.competences_a_acquerir)}
- Plan de formation :
{plan_text}
- Salaire estimé : {result.salaire_estime}
- Perspectives : {result.perspectives}

Présente ces informations de façon claire avec des emojis pour structurer visuellement.
Commence par annoncer le rôle et le score, puis développe chaque section.
Termine avec le message d'encouragement : "{result.message_encouragement}"
Mentionne que le profil a été sauvegardé."""

    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE VOTRE ANALYSE DE REPOSITIONNEMENT")
    print("=" * 60 + "\n")

    stream_message(
        conversation_history + [{"role": "user", "content": presentation_prompt}],
        max_tokens=2048
    )


def build_profile_dict(profile: UserProfile, result: AnalysisResult) -> Dict[str, Any]:
    """Construit le dictionnaire complet pour la sauvegarde."""
    return {
        "profil": {
            "nom": profile.nom,
            "poste_actuel": profile.poste_actuel,
            "annees_experience": profile.annees_experience,
            "secteur": profile.secteur,
            "competences": profile.competences,
            "formation": profile.formation,
            "objectif_reconversion": profile.objectif_reconversion,
        },
        "analyse": {
            "role_cible": result.role_cible,
            "score": result.score,
            "score_justification": result.score_justification,
            "competences_transferables": result.competences_transferables,
            "competences_a_acquerir": result.competences_a_acquerir,
            "plan_formation": result.plan_formation,
            "salaire_estime": result.salaire_estime,
            "perspectives": result.perspectives,
        },
    }
