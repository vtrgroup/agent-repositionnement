import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR, PROFILES_FILE


def init_database() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not PROFILES_FILE.exists():
        PROFILES_FILE.write_text(json.dumps({"profiles": []}, indent=2, ensure_ascii=False))


def save_profile(profile: Dict[str, Any]) -> str:
    init_database()
    data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))

    profile_id = str(uuid.uuid4())[:8]
    entry = {
        "id": profile_id,
        "created_at": datetime.now().isoformat(),
        **profile,
    }
    data["profiles"].append(entry)
    PROFILES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return profile_id


def list_profiles() -> List[Dict[str, Any]]:
    if not PROFILES_FILE.exists():
        return []
    data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    return data.get("profiles", [])


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    for p in list_profiles():
        if p.get("id") == profile_id:
            return p
    return None
