import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-opus-4-6"
DATA_DIR = Path(__file__).parent / "data"
PROFILES_FILE = DATA_DIR / "profiles.json"

# Rôles IA cibles disponibles
AI_ROLES = [
    "AI Product Manager",
    "Prompt Engineer",
    "AI Trainer / Data Labeler",
    "AI Solutions Consultant",
    "MLOps Engineer",
    "AI Ethics Officer",
    "Business Analyst IA",
    "Chief AI Officer (CAIO)",
    "AI Customer Success Manager",
    "Responsable Transformation IA",
]

SYSTEM_PROMPT = """Tu es un expert en repositionnement professionnel vers les métiers de l'IA.
Tu aides les professionnels à identifier le rôle IA le plus adapté à leur profil et à construire
un plan de formation concret et actionnable.

Ton approche est :
- Chaleureuse et encourageante
- Précise et structurée
- Basée sur les compétences transférables existantes
- Réaliste sur le temps et les efforts nécessaires

Tu t'exprimes en français, de manière professionnelle mais accessible."""
