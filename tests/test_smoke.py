from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_paths_exist() -> None:
    manifest = json.loads((ROOT / "experiment_manifest.json").read_text())
    paths = list(manifest["paper"].values())
    paths.extend(manifest["suites"]["quick"])
    paths.extend(manifest["suites"]["public"])
    paths.extend(manifest["suites"]["private_seloger"])
    paths.extend(manifest["public_data"])
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, missing


def test_private_data_not_committed() -> None:
    assert not list(ROOT.rglob("selogerdata*.csv"))


def test_personal_website_not_present() -> None:
    for path in [ROOT / "README.md"]:
        assert "redhamoulla.com" not in path.read_text(encoding="utf-8", errors="ignore")
