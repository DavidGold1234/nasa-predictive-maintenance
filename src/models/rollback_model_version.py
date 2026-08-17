from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

VERSIONS_BASE = Path(os.environ.get("VERSIONS_BASE", "/data/models/versions"))
TARGET_DIR = Path(os.environ.get("TARGET_DIR", "/data/models/artifacts"))
ROLLBACK_VERSION = os.environ.get("ROLLBACK_VERSION", "").strip()

REQUIRED_FILES = [
    "regime_baseline.pkl",
    "healthy_global_scaler.pkl",
    "isolation_forest.pkl",
    "lstm_autoencoder.pt",
    "cnn_autoencoder.pt",
    "selected_features.json",
    "score_calibration.json",
    "retrain_metadata.json",
]

def is_valid_version(path: Path) -> bool:
    return path.is_dir() and all((path / f).exists() for f in REQUIRED_FILES)

def list_valid_versions(base: Path):
    if not base.exists():
        raise FileNotFoundError(f"No existe VERSIONS_BASE: {base}")
    versions = [p for p in base.iterdir() if is_valid_version(p)]
    return sorted(versions, key=lambda p: p.stat().st_mtime, reverse=True)

def read_active_version(target: Path):
    active_file = target / "active_version.json"
    if not active_file.exists():
        return None
    with open(active_file, "r", encoding="utf-8") as f:
        return json.load(f)

def choose_rollback_version(base: Path, target: Path, requested: str):
    valid_versions = list_valid_versions(base)
    if not valid_versions:
        raise RuntimeError(f"No hay versiones válidas en: {base}")

    active_info = read_active_version(target)
    active_version = active_info.get("active_version") if active_info else None

    if requested:
        selected = base / requested
        if not selected.exists():
            raise FileNotFoundError(f"La versión solicitada no existe: {selected}")
        if not is_valid_version(selected):
            raise RuntimeError(f"La versión {requested} está incompleta o inválida")
        return selected, active_version

    candidates = [v for v in valid_versions if v.name != active_version]
    if not candidates:
        raise RuntimeError("No hay una versión anterior disponible para rollback")
    return candidates[0], active_version

def backup_current_target(target: Path):
    if not target.exists():
        return None
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = target.parent / f"{target.name}_rollback_backup_{ts}"
    shutil.move(str(target), str(backup_dir))
    return backup_dir

def deploy(source: Path, target: Path, previous_active: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_current_target(target)
    shutil.copytree(source, target)

    rollback_info = {
        "active_version": source.name,
        "rolled_back_from": previous_active,
        "rollback_at_utc": datetime.utcnow().isoformat(),
        "source_path": str(source),
        "target_path": str(target),
        "backup_previous": str(backup_dir) if backup_dir else None,
        "required_files": REQUIRED_FILES,
    }
    with open(target / "active_version.json", "w", encoding="utf-8") as f:
        json.dump(rollback_info, f, indent=2)
    return rollback_info

def main():
    print(f"📦 Revisando versiones en: {VERSIONS_BASE}")
    selected, previous_active = choose_rollback_version(VERSIONS_BASE, TARGET_DIR, ROLLBACK_VERSION)
    print(f"↩️ Rollback hacia: {selected.name}")
    print(f"📌 Versión activa actual: {previous_active}")
    result = deploy(selected, TARGET_DIR, previous_active)
    print("\n✅ Rollback terminado")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
