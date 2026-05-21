from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_write_sample_config_to_new_path(tmp_path: Path) -> None:
    config_path = tmp_path / "scan_config.yaml"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "clean_scan.py"),
            "--write-sample-config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert config_path.exists()
    text = config_path.read_text(encoding="utf-8")
    assert "input: scans/raw_object.ply" in text
    assert "fusion_reference_ply: true" in text
