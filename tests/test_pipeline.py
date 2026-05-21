from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


o3d = pytest.importorskip("open3d")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clean_scan  # noqa: E402


def make_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return pcd


def grid_points(width: int = 10, height: int = 10, spacing: float = 0.1) -> np.ndarray:
    return np.array(
        [[x * spacing, y * spacing, 0.0] for x in range(width) for y in range(height)],
        dtype=float,
    )


def write_ply(path: Path, points: np.ndarray) -> None:
    pcd = make_cloud(points)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)
    assert ok


def test_scale_correction_matches_expected_axis_dimension() -> None:
    pcd = make_cloud(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )

    scaled, details = clean_scan.apply_scale(
        pcd,
        {"known_axis": "x", "expected_mm": 20.0},
    )

    extent = np.asarray(scaled.get_axis_aligned_bounding_box().get_extent())
    assert extent[0] == pytest.approx(20.0)
    assert details["scale_factor"] == pytest.approx(2.0)


def test_crop_keeps_points_inside_bounds() -> None:
    pcd = make_cloud(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [-1.0, 0.0, 0.0],
            ]
        )
    )

    cropped, _ = clean_scan.apply_crop(
        pcd,
        {"min": [0.0, 0.0, 0.0], "max": [1.5, 1.5, 1.5]},
    )

    points = np.asarray(cropped.points)
    assert clean_scan.point_count(cropped) == 2
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.5)


def test_voxel_downsampling_reduces_point_count() -> None:
    pcd = make_cloud(grid_points(width=20, height=20, spacing=0.02))

    downsampled, details = clean_scan.apply_voxel_downsample(pcd, 0.1)

    assert details["voxel_size_mm"] == pytest.approx(0.1)
    assert 0 < clean_scan.point_count(downsampled) < clean_scan.point_count(pcd)


def test_statistical_outlier_removal_removes_obvious_outlier() -> None:
    points = np.vstack([grid_points(width=10, height=10, spacing=0.1), [[100.0, 100.0, 100.0]]])
    pcd = make_cloud(points)

    filtered, details = clean_scan.apply_statistical_outlier_removal(
        pcd,
        {"nb_neighbors": 10, "std_ratio": 1.0},
    )

    filtered_points = np.asarray(filtered.points)
    assert details["kept_indices"] == clean_scan.point_count(filtered)
    assert clean_scan.point_count(filtered) < clean_scan.point_count(pcd)
    assert not np.any(np.all(np.isclose(filtered_points, [100.0, 100.0, 100.0]), axis=1))


def test_process_generates_reports_for_synthetic_ply(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic.ply"
    output_dir = tmp_path / "processed"
    write_ply(input_path, grid_points(width=12, height=12, spacing=0.08))

    report = clean_scan.process(
        clean_scan.deep_merge(
            clean_scan.DEFAULT_CONFIG,
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "output_name": "synthetic",
                "outlier_removal": {
                    "statistical": {"enabled": False},
                    "radius": {"enabled": False},
                },
                "downsample": {"enabled": True, "voxel_size_mm": 0.2},
                "exports": {
                    "cleaned_ply": True,
                    "fusion_reference_ply": True,
                    "fusion_voxel_size_mm": 0.4,
                    "ascii": True,
                },
                "report": {"markdown": True, "json": True},
                "safety": {"allow_overwrite": True},
            },
        )
    )

    cleaned_path = Path(report.outputs["cleaned_ply"])
    fusion_path = Path(report.outputs["fusion_reference_ply"])
    json_path = Path(report.outputs["json_report"])
    markdown_path = Path(report.outputs["markdown_report"])

    assert cleaned_path.exists()
    assert fusion_path.exists()
    assert json_path.exists()
    assert markdown_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    operations = {op["name"]: op for op in data["operations"]}
    assert operations["voxel_downsample"]["before_points"] > operations["voxel_downsample"]["after_points"]
    assert data["outputs"]["json_report"] == str(json_path)
    assert data["outputs"]["markdown_report"] == str(markdown_path)
    assert "# Point Cloud Processing Report" in markdown_path.read_text(encoding="utf-8")


def test_cli_processes_synthetic_ply(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic_cli.ply"
    output_dir = tmp_path / "processed"
    write_ply(input_path, grid_points(width=8, height=8, spacing=0.05))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "clean_scan.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-name",
            "synthetic_cli",
            "--voxel-mm",
            "0.1",
            "--no-statistical-outlier",
            "--allow-overwrite",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "synthetic_cli_cleaned.ply").exists()
    assert (output_dir / "synthetic_cli_fusion_ref.ply").exists()
    assert (output_dir / "synthetic_cli_report.json").exists()
