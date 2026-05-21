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


def principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return axis / np.linalg.norm(axis)


def principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order]


def rotation_matrix_xyz(x_degrees: float, y_degrees: float, z_degrees: float) -> np.ndarray:
    x = np.deg2rad(x_degrees)
    y = np.deg2rad(y_degrees)
    z = np.deg2rad(z_degrees)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x), -np.sin(x)],
            [0.0, np.sin(x), np.cos(x)],
        ]
    )
    ry = np.array(
        [
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)],
        ]
    )
    rz = np.array(
        [
            [np.cos(z), -np.sin(z), 0.0],
            [np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rz @ ry @ rx


def rotated_elongated_points(count: int = 300) -> np.ndarray:
    t = np.linspace(-12.0, 12.0, count)
    local = np.column_stack(
        [
            t,
            0.35 * np.sin(t * 1.7),
            0.25 * np.cos(t * 1.3),
        ]
    )
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return local @ rotation.T


def rotated_t_points() -> np.ndarray:
    stem_x = np.linspace(-12.0, 12.0, 180)
    stem = np.column_stack(
        [
            stem_x,
            0.15 * np.sin(stem_x),
            0.10 * np.cos(stem_x),
        ]
    )
    arm_y = np.linspace(-4.0, 4.0, 90)
    arm = np.column_stack(
        [
            np.full_like(arm_y, 10.0),
            arm_y,
            0.10 * np.sin(arm_y),
        ]
    )
    points = np.vstack([stem, arm])
    return points @ rotation_matrix_xyz(23.0, -17.0, 41.0).T


def rotated_disk_points() -> np.ndarray:
    radii = np.linspace(0.5, 8.0, 16)
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    points = []
    for radius in radii:
        for angle in angles:
            points.append([radius * np.cos(angle), radius * np.sin(angle), 0.03 * np.sin(3.0 * angle)])
    return np.asarray(points, dtype=float) @ rotation_matrix_xyz(-31.0, 22.0, 18.0).T


def write_ply(path: Path, points: np.ndarray) -> None:
    pcd = make_cloud(points)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)
    assert ok


def write_cloud(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)
    assert ok


def sphere_like_cloud(points: int = 2200, radius: float = 10.0) -> o3d.geometry.PointCloud:
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(0)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=40)
    mesh.compute_vertex_normals()
    return mesh.sample_points_poisson_disk(number_of_points=points, init_factor=5)


def mesh_test_config(input_path: Path, output_dir: Path, output_name: str, target_triangles: int = 400) -> dict:
    return clean_scan.deep_merge(
        clean_scan.DEFAULT_CONFIG,
        {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "output_name": output_name,
            "outlier_removal": {
                "statistical": {"enabled": False},
                "radius": {"enabled": False},
            },
            "downsample": {"enabled": False},
            "exports": {
                "cleaned_ply": True,
                "fusion_reference_ply": True,
                "fusion_voxel_size_mm": 0.35,
                "ascii": True,
            },
            "mesh": {
                "enabled": True,
                "source": "fusion_reference",
                "method": "ball_pivoting",
                "export_obj": True,
                "export_stl": True,
                "normals": {
                    "radius_mm": 2.5,
                    "max_nn": 50,
                    "orient_consistent_tangent_plane_k": 30,
                },
                "ball_pivoting": {
                    "radius_multipliers": [1.5, 2.5, 3.5],
                },
                "cleanup": {
                    "simplify_enabled": True,
                    "target_triangles": target_triangles,
                },
                "validation": {
                    "min_triangles": 50,
                    "max_triangles": max(1000, target_triangles * 3),
                    "warn_if_not_watertight": True,
                },
            },
            "report": {"markdown": True, "json": True},
            "safety": {"allow_overwrite": True},
        },
    )


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


def test_major_axis_alignment_rotates_principal_axis_to_target_axis() -> None:
    pcd = make_cloud(rotated_elongated_points())

    aligned, details = clean_scan.apply_major_axis_alignment(
        pcd,
        {
            "enabled": True,
            "method": "pca_major_axis",
            "target_axis": "x",
            "center_at_origin": False,
        },
    )

    aligned_axis = principal_axis(np.asarray(aligned.points))
    assert abs(float(np.dot(aligned_axis, np.array([1.0, 0.0, 0.0])))) > 0.999
    assert details["method"] == "pca_major_axis"
    assert details["target_axis"] == "x"
    assert "rotation_matrix" in details


def test_pca_frame_alignment_rotates_t_shape_roll_to_middle_axis() -> None:
    pcd = make_cloud(rotated_t_points())

    aligned, details = clean_scan.apply_major_axis_alignment(
        pcd,
        {
            "enabled": True,
            "method": "pca_frame",
            "target_axes": {
                "major": "x",
                "middle": "y",
                "minor": "z",
            },
            "center_at_origin": False,
        },
    )

    axes = principal_axes(np.asarray(aligned.points))
    assert abs(float(np.dot(axes[:, 0], np.array([1.0, 0.0, 0.0])))) > 0.999
    assert abs(float(np.dot(axes[:, 1], np.array([0.0, 1.0, 0.0])))) > 0.999
    assert abs(float(np.dot(axes[:, 2], np.array([0.0, 0.0, 1.0])))) > 0.999
    assert details["method"] == "pca_frame"
    assert details["target_axes"] == {"major": "x", "middle": "y", "minor": "z"}


def test_pca_frame_alignment_rotates_disk_normal_to_minor_axis() -> None:
    pcd = make_cloud(rotated_disk_points())

    aligned, details = clean_scan.apply_major_axis_alignment(
        pcd,
        {
            "enabled": True,
            "method": "pca_frame",
            "target_axes": {
                "major": "x",
                "middle": "y",
                "minor": "z",
            },
            "center_at_origin": False,
        },
    )

    axes = principal_axes(np.asarray(aligned.points))
    assert abs(float(np.dot(axes[:, 2], np.array([0.0, 0.0, 1.0])))) > 0.999
    assert details["target_axes"]["minor"] == "z"
    assert details["warnings"]


def test_process_reports_major_axis_alignment(tmp_path: Path) -> None:
    input_path = tmp_path / "rotated.ply"
    output_dir = tmp_path / "processed"
    write_ply(input_path, rotated_elongated_points())

    report = clean_scan.process(
        clean_scan.deep_merge(
            clean_scan.DEFAULT_CONFIG,
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "output_name": "rotated",
                "outlier_removal": {
                    "statistical": {"enabled": False},
                    "radius": {"enabled": False},
                },
                "downsample": {"enabled": False},
                "alignment": {
                    "enabled": True,
                    "method": "pca_frame",
                    "target_axes": {
                        "major": "x",
                        "middle": "y",
                        "minor": "z",
                    },
                    "center_at_origin": False,
                },
                "exports": {
                    "cleaned_ply": True,
                    "fusion_reference_ply": False,
                    "ascii": True,
                },
                "report": {"markdown": True, "json": True},
                "safety": {"allow_overwrite": True},
            },
        )
    )

    cleaned = o3d.io.read_point_cloud(report.outputs["cleaned_ply"])
    aligned_axis = principal_axis(np.asarray(cleaned.points))
    assert abs(float(np.dot(aligned_axis, np.array([1.0, 0.0, 0.0])))) > 0.999

    json_path = Path(report.outputs["json_report"])
    data = json.loads(json_path.read_text(encoding="utf-8"))
    operations = {op["name"]: op for op in data["operations"]}
    assert operations["major_axis_alignment"]["enabled"] is True
    assert operations["major_axis_alignment"]["details"]["target_axes"]["major"] == "x"
    assert "rotation_matrix" in operations["major_axis_alignment"]["details"]


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
    assert data["mesh"]["enabled"] is False


def test_ball_pivoting_reconstructs_sphere_like_point_cloud(tmp_path: Path) -> None:
    report = clean_scan.reconstruct_and_export_mesh(
        source_pcd=sphere_like_cloud(points=1800),
        source_label="cleaned",
        cfg=clean_scan.deep_merge(
            clean_scan.DEFAULT_CONFIG["mesh"],
            {
                "enabled": True,
                "source": "cleaned",
                "method": "ball_pivoting",
                "export_obj": False,
                "export_stl": False,
                "normals": {
                    "radius_mm": 2.5,
                    "max_nn": 50,
                    "orient_consistent_tangent_plane_k": 30,
                },
                "cleanup": {
                    "simplify_enabled": False,
                },
                "validation": {
                    "min_triangles": 50,
                    "max_triangles": 100000,
                    "warn_if_not_watertight": True,
                },
            },
        ),
        output_dir=tmp_path,
        output_name="sphere_direct",
        ascii_mode=True,
        allow_overwrite=True,
    )

    assert report.enabled is True
    assert report.method == "ball_pivoting"
    assert report.triangle_count_before_cleanup is not None
    assert report.triangle_count_before_cleanup >= 50
    assert report.triangle_count_after_cleanup is not None
    assert report.triangle_count_after_cleanup >= 50
    assert report.vertex_count is not None
    assert report.vertex_count > 0
    assert "Mesh export is for Fusion reference only." in report.warnings


def test_process_mesh_export_writes_obj_stl_and_reports_simplification(tmp_path: Path) -> None:
    input_path = tmp_path / "sphere.ply"
    output_dir = tmp_path / "processed"
    write_cloud(input_path, sphere_like_cloud(points=2600))

    report = clean_scan.process(mesh_test_config(input_path, output_dir, "sphere", target_triangles=300))

    obj_path = Path(report.outputs["fusion_reference_obj"])
    stl_path = Path(report.outputs["fusion_reference_stl"])
    json_path = Path(report.outputs["json_report"])
    markdown_path = Path(report.outputs["markdown_report"])

    assert obj_path.exists()
    assert stl_path.exists()
    assert report.mesh.enabled is True
    assert report.mesh.source == "fusion_reference"
    assert report.mesh.triangle_count_before_cleanup is not None
    assert report.mesh.triangle_count_after_cleanup is not None
    assert report.mesh.triangle_count_before_cleanup > report.mesh.triangle_count_after_cleanup
    assert report.mesh.triangle_count_after_cleanup <= 300
    assert report.mesh.cleanup["simplified"] is True

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["mesh"]["output_paths"]["fusion_reference_obj"] == str(obj_path)
    assert data["mesh"]["output_paths"]["fusion_reference_stl"] == str(stl_path)
    assert "Mesh export is for Fusion reference only." in data["warnings"]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Mesh" in markdown
    assert "Verify against the cleaned PLY and physical measurements." in markdown


def test_sparse_point_cloud_mesh_export_fails_clearly(tmp_path: Path) -> None:
    input_path = tmp_path / "sparse.ply"
    output_dir = tmp_path / "processed"
    write_ply(
        input_path,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
    )

    with pytest.raises(ValueError, match="Mesh reconstruction produced"):
        clean_scan.process(mesh_test_config(input_path, output_dir, "sparse", target_triangles=300))


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
