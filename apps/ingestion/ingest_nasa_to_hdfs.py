from __future__ import annotations

import subprocess
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
HDFS_BRONZE_NASA = "/user/root/nasa/bronze/nasa_raw"

EXPECTED_FILES = [
    *(f"train_FD00{i}.txt" for i in range(1, 5)),
    *(f"test_FD00{i}.txt" for i in range(1, 5)),
    *(f"RUL_FD00{i}.txt" for i in range(1, 5)),
]


def run_cmd(args: list[str]) -> None:
    print("\n>>", " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}")


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"No existe RAW_DIR: {RAW_DIR}")

    files = [RAW_DIR / name for name in EXPECTED_FILES if (RAW_DIR / name).exists()]
    missing = [name for name in EXPECTED_FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos NASA C-MAPSS en data/raw: " + ", ".join(missing)
        )

    run_cmd(["docker", "exec", "namenode", "hdfs", "dfs", "-mkdir", "-p", HDFS_BRONZE_NASA])

    for file_path in files:
        target = f"{HDFS_BRONZE_NASA}/{file_path.name}"
        run_cmd(
            [
                "docker",
                "exec",
                "namenode",
                "hdfs",
                "dfs",
                "-put",
                "-f",
                f"/data/raw/{file_path.name}",
                target,
            ]
        )

    run_cmd(["docker", "exec", "namenode", "hdfs", "dfs", "-ls", HDFS_BRONZE_NASA])
    print("\n[OK] Ingesta NASA -> bronze completada.")


if __name__ == "__main__":
    main()
