#!/usr/bin/env python3
"""
Agent de Repositionnement Professionnel vers l'IA
Entrypoint principal — supporte le mode interactif et le mode --profile JSON
"""

import json
import sys

from agent import (
    collect_profile_conversationally,
    analyze_profile,
    present_results,
    build_profile_dict,
    UserProfile,
)
from database import save_profile, list_profiles, init_database
from config import ANTHROPIC_API_KEY


def print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║       AGENT DE REPOSITIONNEMENT PROFESSIONNEL VERS L'IA      ║
║                   Propulsé par Claude Opus 4.6               ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_with_profile(profile_data: dict) -> None:
    """Lance l'analyse à partir d'un profil pré-rempli (mode non-interactif)."""
    print_banner()
    init_database()

    profile = UserProfile(**profile_data)

    print(f"👤 Profil reçu : {profile.nom} — {profile.poste_actuel} ({profile.secteur})\n")
    print("⚙️  Analyse du profil en cours...\n")

    result = analyze_profile(profile, [])
    present_results(profile, result, [])

    profile_dict = build_profile_dict(profile, result)
    profile_id = save_profile(profile_dict)

    print(f"\n✅ Profil sauvegardé — ID : {profile_id}")
    print("   Fichier : data/profiles.json\n")


def run_interactive() -> None:
    """Lance le flux conversationnel interactif (terminal)."""
    print_banner()

    if not ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY non définie.")
        sys.exit(1)

    init_database()

    try:
        profile, history = collect_profile_conversationally()
        print("\n\n⚙️  Analyse en cours...\n")
        result = analyze_profile(profile, history)
        present_results(profile, result, history)

        profile_id = save_profile(build_profile_dict(profile, result))
        print(f"\n✅ Profil sauvegardé — ID : {profile_id}\n")

    except KeyboardInterrupt:
        print("\n\n👋 À bientôt !")
        sys.exit(0)


def show_profiles() -> None:
    profiles = list_profiles()
    if not profiles:
        print("Aucun profil sauvegardé.")
        return
    print(f"\n{'='*60}\n  {len(profiles)} PROFIL(S)\n{'='*60}")
    for p in profiles:
        pr = p.get("profil", {})
        an = p.get("analyse", {})
        print(f"\n🆔 {p['id']} | {p['created_at'][:10]}")
        print(f"   {pr.get('nom')} — {pr.get('poste_actuel')}")
        print(f"   🎯 {an.get('role_cible')} | ⭐ {an.get('score')}/100")
    print()


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "--list":
        init_database()
        show_profiles()
    elif args and args[0] == "--profile":
        if len(args) < 2:
            print("Usage : python3 main.py --profile '{\"nom\": ...}'")
            sys.exit(1)
        profile_data = json.loads(args[1])
        run_with_profile(profile_data)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
