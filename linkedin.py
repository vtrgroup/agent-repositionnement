"""Extraction automatique de profil LinkedIn — Proxycurl + scraping direct + Claude."""

import json
import os
import re
from typing import Dict, Any

import requests
from bs4 import BeautifulSoup
import anthropic

from config import MODEL

PROXYCURL_API_KEY = os.environ.get("PROXYCURL_API_KEY", "")
PROXYCURL_URL = "https://nubela.co/proxycurl/api/v2/linkedin"

# Headers imitant un navigateur Chrome réel
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}


def fetch_linkedin_page(url: str) -> tuple[str, str]:
    """
    Télécharge la page LinkedIn publique.
    Retourne (html, status) où status = "ok" | "login_required" | "error"
    """
    try:
        session = requests.Session()
        # Premier appel pour récupérer les cookies initiaux
        resp = session.get(
            url,
            headers=BROWSER_HEADERS,
            timeout=15,
            allow_redirects=True,
        )

        final_url = resp.url
        html = resp.text

        # Redirigé vers authwall / login / signup → connexion requise
        redirected_away = (
            "authwall" in final_url
            or "login" in final_url
            or "signup" in final_url
        )

        # Si la page contient le pageKey d'un profil public → on est sur le bon profil
        is_public_profile = 'pageKey" content="public_profile_v3_desktop"' in html

        # Si on est resté sur une URL /in/ ET que la page est bien un profil public → ok
        on_profile_url = "linkedin.com/in/" in final_url

        if redirected_away or not on_profile_url:
            return html, "login_required"

        if not is_public_profile:
            # Heuristique de repli : si on est sur /in/ mais pas le pageKey public,
            # c'est probablement une page de login déguisée
            return html, "login_required"

        return html, "ok"

    except requests.RequestException as e:
        return "", f"error:{e}"


def extract_jsonld(html: str) -> list[dict]:
    """Extrait les blocs JSON-LD de la page (données structurées)."""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            results.append(data)
        except (json.JSONDecodeError, AttributeError):
            pass
    return results


def extract_meta_tags(html: str) -> dict:
    """Extrait les Open Graph et Twitter Card meta tags."""
    soup = BeautifulSoup(html, "html.parser")
    meta = {}
    for tag in soup.find_all("meta"):
        prop = tag.get("property") or tag.get("name") or ""
        content = tag.get("content") or ""
        if prop and content:
            meta[prop] = content
    return meta


def extract_visible_text(html: str) -> str:
    """Extrait le texte visible utile de la page LinkedIn."""
    soup = BeautifulSoup(html, "html.parser")

    # Supprimer scripts, styles, nav
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    # Cibler les sections pertinentes du profil LinkedIn
    sections = []
    for selector in [
        "main",
        "[data-section]",
        ".pv-profile-section",
        ".artdeco-card",
        ".scaffold-layout__main",
    ]:
        found = soup.select(selector)
        for el in found:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 50:
                sections.append(text)

    if sections:
        return "\n\n".join(sections)[:8000]

    # Fallback : tout le texte visible
    return soup.get_text(separator="\n", strip=True)[:8000]


def parse_with_claude(raw_content: str, api_key: str, source: str = "page") -> Dict[str, Any]:
    """Envoie le contenu brut à Claude pour en extraire le profil structuré."""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Voici le contenu brut extrait d'un profil LinkedIn ({source}).
Extrais les informations professionnelles et retourne UNIQUEMENT ce JSON valide :

{{
  "nom": "prénom et nom complet",
  "poste_actuel": "intitulé exact du poste actuel ou dernier poste",
  "annees_experience": estimation_entière_durée_totale_de_carrière,
  "secteur": "secteur ou industrie principale",
  "competences": ["compétence1", "compétence2", "compétence3", "compétence4", "compétence5"],
  "formation": "diplôme le plus élevé + domaine + établissement",
  "entreprise_actuelle": "nom de l'entreprise actuelle",
  "localisation": "ville ou région",
  "resume": "résumé du parcours en 2-3 phrases"
}}

Si une info est absente, déduis-la du contexte.
UNIQUEMENT le JSON, sans aucun texte avant ou après.

CONTENU :
{raw_content}"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="Tu es un extracteur de données RH. Retourne uniquement du JSON valide.",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"error": f"Parse error: {raw[:200]}"}


def fetch_via_proxycurl(url: str) -> Dict[str, Any]:
    """Utilise l'API Proxycurl pour récupérer un profil LinkedIn structuré."""
    if not PROXYCURL_API_KEY:
        return {"error": "no_proxycurl_key"}

    try:
        resp = requests.get(
            PROXYCURL_URL,
            headers={"Authorization": f"Bearer {PROXYCURL_API_KEY}"},
            params={"url": url, "use_cache": "if-present"},
            timeout=30,
        )
        if resp.status_code != 200:
            return {"error": f"proxycurl_http_{resp.status_code}: {resp.text[:200]}"}
        return resp.json()
    except requests.RequestException as e:
        return {"error": f"proxycurl_error: {e}"}


def map_proxycurl_to_profile(pc: Dict[str, Any]) -> Dict[str, Any]:
    """Transforme la réponse Proxycurl en format attendu par l'app."""
    full_name = f"{pc.get('first_name','')} {pc.get('last_name','')}".strip() or pc.get("full_name", "")

    # Expériences — estimer années
    experiences = pc.get("experiences", []) or []
    total_years = 0
    current_poste = ""
    current_company = ""
    for exp in experiences:
        starts = exp.get("starts_at") or {}
        ends = exp.get("ends_at") or {}
        if starts.get("year"):
            end_year = ends.get("year") if ends else 2026
            total_years += max(0, (end_year or 2026) - starts["year"])
        if not current_poste and (exp.get("ends_at") is None):
            current_poste = exp.get("title", "")
            current_company = exp.get("company", "")
    if not current_poste and experiences:
        current_poste = experiences[0].get("title", "")
        current_company = experiences[0].get("company", "")

    # Formations
    educations = pc.get("education", []) or []
    formation_parts = []
    for edu in educations[:2]:
        degree = edu.get("degree_name") or ""
        field = edu.get("field_of_study") or ""
        school = edu.get("school") or ""
        piece = " ".join(p for p in [degree, field, school] if p).strip()
        if piece:
            formation_parts.append(piece)
    formation = " · ".join(formation_parts)

    # Compétences
    competences = (pc.get("skills") or [])[:8]
    if not competences:
        # Fallback : déduire des titres de poste récents
        competences = [e.get("title", "") for e in experiences[:5] if e.get("title")]

    # Secteur — approximation depuis l'industrie de l'entreprise actuelle ou headline
    headline = pc.get("headline") or ""
    secteur = pc.get("industry") or current_company or ""

    location_parts = [pc.get("city"), pc.get("country_full_name") or pc.get("country")]
    localisation = ", ".join(p for p in location_parts if p)

    resume = pc.get("summary") or headline

    return {
        "nom": full_name,
        "poste_actuel": current_poste,
        "annees_experience": min(total_years, 50),
        "secteur": secteur,
        "competences": [c for c in competences if c],
        "formation": formation,
        "entreprise_actuelle": current_company,
        "localisation": localisation,
        "resume": resume[:400] if resume else "",
    }


def extract_linkedin_profile(url: str, api_key: str) -> Dict[str, Any]:
    """
    Pipeline : Proxycurl (si clé dispo) → scraping direct → Claude.
    """
    # 1. Essai via Proxycurl (marche depuis n'importe quelle IP)
    if PROXYCURL_API_KEY:
        pc_data = fetch_via_proxycurl(url)
        if "error" not in pc_data:
            return map_proxycurl_to_profile(pc_data)
        # Si Proxycurl échoue, on continue avec le fallback

    # 2. Fallback : scraping direct (ne marche pas sur IP datacenter)
    html, status = fetch_linkedin_page(url)

    if status == "login_required":
        return {"error": "login_required"}

    if status.startswith("error"):
        return {"error": f"Impossible d'accéder à la page : {status}"}

    if not html:
        return {"error": "Page vide reçue."}

    # Combiner toutes les sources disponibles : JSON-LD + meta tags + texte visible
    # LinkedIn obfusque certains champs (poste, compétences) mais pas tous (nom, entreprise, bio)
    jsonld_blocks = extract_jsonld(html)
    meta = extract_meta_tags(html)
    visible = extract_visible_text(html)

    parts = []
    if jsonld_blocks:
        parts.append(f"JSON-LD:\n{json.dumps(jsonld_blocks, ensure_ascii=False)[:4000]}")
    if meta:
        parts.append(f"META TAGS:\n{json.dumps(meta, ensure_ascii=False)[:2000]}")
    if visible:
        parts.append(f"TEXTE PAGE:\n{visible[:3000]}")

    combined = "\n\n".join(parts)

    result = parse_with_claude(combined, api_key, source="JSON-LD + meta tags + texte")

    # Marquer les champs obfusqués pour que l'UI invite l'utilisateur à les remplir
    if "error" not in result:
        obfuscated_marker = "***"
        for field in ("poste_actuel", "competences", "formation"):
            val = result.get(field)
            if isinstance(val, str) and ("*" in val or not val):
                result[field] = ""
            elif isinstance(val, list) and (not val or any("*" in str(v) for v in val)):
                result[field] = []

    return result


def validate_linkedin_url(url: str) -> bool:
    return bool(re.match(r"https?://(www\.)?linkedin\.com/in/[\w\-]+", url.strip()))


def extract_from_text(profile_text: str, api_key: str) -> Dict[str, Any]:
    """Extraction depuis texte copié-collé (fallback manuel)."""
    return parse_with_claude(profile_text[:10000], api_key, source="texte copié-collé")
