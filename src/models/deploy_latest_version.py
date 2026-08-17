from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

VERSIONS_BASE = Path(os.environ.get("VERSIONS_BASE", "/data/models/versions"))
TARGET_DIR = Path(os.environ.get("TARGET_DIR", "/data/models/artifacts"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "").strip()

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

def list_valid_versions(base: Path):
    if not base.exists():
        raise FileNotFoundError(f"No existe VERSIONS_BASE: {base}")
    versions = []
    for d in base.iterdir():
        if d.is_dir() and all((d / f).exists() for f in REQUIRED_FILES):
            versions.append(d)
    return sorted(versions, key=lambda p: p.stat().st_mtime, reverse=True)

def choose_version(base: Path, requested: str):
    valid_versions = list_valid_versions(base)
    if not valid_versions:
        raise RuntimeError(f"No hay versiones válidas en: {base}")
    if requested:
        selected = base / requested
        if not selected.exists():
            raise FileNotFoundError(f"La versión pedida no existe: {selected}")
        missing = [f for f in REQUIRED_FILES if not (selected / f).exists()]
        if missing:
            raise RuntimeError(f"La versión {requested} está incompleta. Faltan: {missing}")
        return selected
    return valid_versions[0]

def backup_target(target: Path):
    if not target.exists():
        return None
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = target.parent / f"{target.name}_backup_{ts}"
    shutil.move(str(target), str(backup_dir))
    return backup_dir

def deploy_version(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_target(target)
    shutil.copytree(source, target)

    active_info = {
        "active_version": source.name,
        "deployed_at_utc": datetime.utcnow().isoformat(),
        "source_path": str(source),
        "target_path": str(target),
        "backup_previous": str(backup_dir) if backup_dir else None,
        "required_files": REQUIRED_FILES,
    }
    with open(target / "active_version.json", "w", encoding="utf-8") as f:
        json.dump(active_info, f, indent=2)
    return active_info

def main():
    print(f"📦 Buscando versiones en: {VERSIONS_BASE}")
    source = choose_version(VERSIONS_BASE, MODEL_VERSION)
    print(f"✅ Versión seleccionada: {source.name}")
    print(f"📂 Desplegando hacia: {TARGET_DIR}")
    active_info = deploy_version(source, TARGET_DIR)
    print("\n✅ Deploy terminado")
    print(json.dumps(active_info, indent=2))

if __name__ == "__main__":
    main()
