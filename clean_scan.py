#!/usr/bin/env python3
"""
clean_scan.py

Conservative point-cloud preprocessing for Revo Metro / Revopoint-style scan exports
before use as reference geometry in Autodesk Fusion.

Primary workflow:
    Revo Metro export PLY point cloud
      -> python clean_scan.py --config scan_config.yaml
      -> cleaned PLY + optional Fusion-reference mesh/PLY + reports
      -> Insert mesh/reference geometry in Fusion and rebuild CAD parametrically

This script deliberately avoids automatic CAD reconstruction. It performs reversible,
reported preprocessing: scale verification, crop, cluster filtering, outlier removal,
voxel downsampling, optional normal estimation, optional reference mesh export, and
export.

Dependencies:
    python -m pip install open3d pyyaml numpy

Example:
    python clean_scan.py --config scan_config.yaml
    python clean_scan.py --input raw_scan.ply --output-dir processed --known-x-mm 82.5 --voxel-mm 0.25
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - exercised through CLI behavior.
    o3d = None  # type: ignore[assignment]
    OPEN3D_IMPORT_ERROR: Optional[ImportError] = exc
else:
    OPEN3D_IMPORT_ERROR = None

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised through CLI behavior.
    yaml = None  # type: ignore[assignment]
    YAML_IMPORT_ERROR: Optional[ImportError] = exc
else:
    YAML_IMPORT_ERROR = None


def require_open3d() -> None:
    if o3d is not None:
        return
    raise RuntimeError(
        "Missing dependency: open3d. Install project dependencies with "
        "`python -m pip install -r requirements.txt`. Open3D wheels are not "
        "available for every Python release; if installation says no matching "
        "distribution was found, use CPython 3.10, 3.11, or 3.12."
    ) from OPEN3D_IMPORT_ERROR


def require_yaml() -> None:
    if yaml is not None:
        return
    raise RuntimeError(
        "Missing dependency: PyYAML. Install project dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from YAML_IMPORT_ERROR


# -----------------------------
# Config defaults
# -----------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "input": None,
    "output_dir": "processed",
    "output_name": None,
    "scale": {
        "enabled": False,
        "known_axis": "x",       # x, y, z, longest
        "expected_mm": None,
    },
    "crop": {
        "enabled": False,
        "min": None,             # [x, y, z] in current units, assumed mm after scaling
        "max": None,             # [x, y, z]
    },
    "cluster": {
        "keep_largest": False,
        "eps_mm": 1.5,
        "min_points": 50,
    },
    "outlier_removal": {
        "statistical": {
            "enabled": True,
            "nb_neighbors": 30,
            "std_ratio": 1.5,
        },
        "radius": {
            "enabled": False,
            "radius_mm": 1.0,
            "min_neighbors": 8,
        },
    },
    "downsample": {
        "enabled": True,
        "voxel_size_mm": 0.25,
    },
    "normals": {
        "enabled": False,
        "radius_mm": 1.0,
        "max_nn": 30,
    },
    "alignment": {
        "enabled": False,
        "method": "pca_frame",
        "target_axis": "x",       # legacy pca_major_axis target: x, y, z, -x, -y, -z
        "target_axes": {
            "major": "x",
            "middle": "y",
            "minor": "z",
        },
        "center_at_origin": False,
    },
    "exports": {
        "cleaned_ply": True,
        "fusion_reference_ply": True,
        "fusion_voxel_size_mm": 0.5,
        "ascii": False,
    },
    "mesh": {
        "enabled": False,
        "source": "fusion_reference",
        "method": "ball_pivoting",
        "export_obj": True,
        "export_stl": True,
        "normals": {
            "radius_mm": 1.0,
            "max_nn": 30,
            "orient_consistent_tangent_plane_k": 50,
        },
        "ball_pivoting": {
            "radius_multipliers": [1.5, 2.5, 4.0],
        },
        "poisson": {
            "depth": 8,
            "width": 0,
            "scale": 1.1,
            "linear_fit": False,
            "density_trim_quantile": 0.02,
        },
        "cleanup": {
            "remove_degenerate_triangles": True,
            "remove_duplicated_triangles": True,
            "remove_duplicated_vertices": True,
            "remove_non_manifold_edges": True,
            "simplify_enabled": True,
            "target_triangles": 150000,
        },
        "validation": {
            "min_triangles": 1000,
            "max_triangles": 300000,
            "warn_if_not_watertight": True,
        },
    },
    "report": {
        "markdown": True,
        "json": True,
    },
    "safety": {
        "allow_overwrite": False,
    },
}


# -----------------------------
# Report structures
# -----------------------------

@dataclass
class CloudStats:
    label: str
    point_count: int
    bbox_min: List[float]
    bbox_max: List[float]
    bbox_dimensions: List[float]
    bbox_center: List[float]
    approximate_spacing: Optional[float] = None


@dataclass
class OperationReport:
    name: str
    enabled: bool
    before_points: int
    after_points: int
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def removed_points(self) -> int:
        return self.before_points - self.after_points


@dataclass
class MeshReport:
    enabled: bool
    source: Optional[str] = None
    method: Optional[str] = None
    normal_estimation: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    cleanup: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    triangle_count_before_cleanup: Optional[int] = None
    triangle_count_after_cleanup: Optional[int] = None
    vertex_count: Optional[int] = None
    watertight: Optional[bool] = None
    output_paths: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcessingReport:
    script: str
    timestamp_utc: str
    input_path: str
    output_dir: str
    config: Dict[str, Any]
    stats: List[CloudStats] = field(default_factory=list)
    operations: List[OperationReport] = field(default_factory=list)
    outputs: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    mesh: MeshReport = field(default_factory=lambda: MeshReport(enabled=False))


# -----------------------------
# Utility functions
# -----------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override into base recursively without mutating either."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    require_yaml()
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping/object: {path}")
    return deep_merge(DEFAULT_CONFIG, raw)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.input:
        cfg["input"] = args.input
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.output_name:
        cfg["output_name"] = args.output_name
    if args.known_x_mm is not None:
        cfg["scale"]["enabled"] = True
        cfg["scale"]["known_axis"] = "x"
        cfg["scale"]["expected_mm"] = args.known_x_mm
    if args.known_y_mm is not None:
        cfg["scale"]["enabled"] = True
        cfg["scale"]["known_axis"] = "y"
        cfg["scale"]["expected_mm"] = args.known_y_mm
    if args.known_z_mm is not None:
        cfg["scale"]["enabled"] = True
        cfg["scale"]["known_axis"] = "z"
        cfg["scale"]["expected_mm"] = args.known_z_mm
    if args.known_longest_mm is not None:
        cfg["scale"]["enabled"] = True
        cfg["scale"]["known_axis"] = "longest"
        cfg["scale"]["expected_mm"] = args.known_longest_mm
    if args.voxel_mm is not None:
        cfg["downsample"]["enabled"] = True
        cfg["downsample"]["voxel_size_mm"] = args.voxel_mm
    if args.no_statistical_outlier:
        cfg["outlier_removal"]["statistical"]["enabled"] = False
    if args.keep_largest_cluster:
        cfg["cluster"]["keep_largest"] = True
    if args.allow_overwrite:
        cfg["safety"]["allow_overwrite"] = True
    return cfg


def require_number(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required numeric value: {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value for {name}; got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Expected finite numeric value for {name}; got {value!r}")
    return result


def require_positive(value: Any, name: str) -> float:
    result = require_number(value, name)
    if result <= 0:
        raise ValueError(f"Expected positive value for {name}; got {result}")
    return result


def as_vec3(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-element list, e.g. [x, y, z]")
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numeric values")
    return arr


def point_count(pcd: o3d.geometry.PointCloud) -> int:
    return int(np.asarray(pcd.points).shape[0])


def approximate_spacing_from_bbox(pcd: o3d.geometry.PointCloud) -> Optional[float]:
    """
    Very rough spacing estimate from bounding-box volume / point count.
    This is not nearest-neighbor spacing, but it is cheap and good enough for reports.
    """
    n = point_count(pcd)
    if n <= 0:
        return None
    dims = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent(), dtype=float)
    volume = float(np.prod(np.maximum(dims, 1e-12)))
    if volume <= 0:
        return None
    return float((volume / n) ** (1.0 / 3.0))


def compute_stats(pcd: o3d.geometry.PointCloud, label: str) -> CloudStats:
    n = point_count(pcd)
    if n == 0:
        return CloudStats(label, 0, [], [], [], [], None)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.get_min_bound(), dtype=float)
    bbox_max = np.asarray(bbox.get_max_bound(), dtype=float)
    dims = np.asarray(bbox.get_extent(), dtype=float)
    center = np.asarray(bbox.get_center(), dtype=float)
    spacing = approximate_spacing_from_bbox(pcd)
    return CloudStats(
        label=label,
        point_count=n,
        bbox_min=bbox_min.tolist(),
        bbox_max=bbox_max.tolist(),
        bbox_dimensions=dims.tolist(),
        bbox_center=center.tolist(),
        approximate_spacing=spacing,
    )


def format_dims(dims: Iterable[float]) -> str:
    vals = list(dims)
    if not vals:
        return "n/a"
    return " × ".join(f"{v:.3f}" for v in vals)


def print_stats(stats: CloudStats) -> None:
    print(f"[{stats.label}]")
    print(f"  points: {stats.point_count:,}")
    print(f"  bbox dimensions: {format_dims(stats.bbox_dimensions)}")
    if stats.approximate_spacing is not None:
        print(f"  approximate spacing: {stats.approximate_spacing:.4f}")


def safe_write_path(path: Path, allow_overwrite: bool) -> Path:
    if allow_overwrite or not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(1, 10_000):
        candidate = parent / f"{stem}_{i:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find available output filename for {path}")


def write_point_cloud(path: Path, pcd: o3d.geometry.PointCloud, ascii_mode: bool, allow_overwrite: bool) -> Path:
    require_open3d()
    final_path = safe_write_path(path, allow_overwrite)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(final_path), pcd, write_ascii=ascii_mode)
    if not ok:
        raise RuntimeError(f"Open3D failed to write point cloud: {final_path}")
    return final_path


def write_triangle_mesh(path: Path, mesh: o3d.geometry.TriangleMesh, ascii_mode: bool, allow_overwrite: bool) -> Path:
    require_open3d()
    final_path = safe_write_path(path, allow_overwrite)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(final_path), mesh, write_ascii=ascii_mode)
    if not ok:
        raise RuntimeError(f"Open3D failed to write triangle mesh: {final_path}")
    return final_path


# -----------------------------
# Processing operations
# -----------------------------

def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    require_open3d()
    if not path.exists():
        raise FileNotFoundError(f"Input point cloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd is None or point_count(pcd) == 0:
        raise ValueError(f"Input did not load as a non-empty point cloud: {path}")
    return pcd


def apply_scale(pcd: o3d.geometry.PointCloud, cfg: Dict[str, Any]) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    axis = str(cfg.get("known_axis", "x")).lower()
    expected = require_positive(cfg.get("expected_mm"), "scale.expected_mm")

    bbox = pcd.get_axis_aligned_bounding_box()
    dims = np.asarray(bbox.get_extent(), dtype=float)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis)

    if axis == "longest":
        measured = float(np.max(dims))
        measured_axis = ["x", "y", "z"][int(np.argmax(dims))]
    elif axis_index is not None:
        measured = float(dims[axis_index])
        measured_axis = axis
    else:
        raise ValueError("scale.known_axis must be one of: x, y, z, longest")

    if measured <= 0:
        raise ValueError(f"Cannot scale: measured dimension on {axis!r} is {measured}")

    scale_factor = expected / measured
    scaled = copy.deepcopy(pcd)
    scaled.scale(scale_factor, center=(0.0, 0.0, 0.0))

    return scaled, {
        "known_axis": axis,
        "measured_axis": measured_axis,
        "measured_before": measured,
        "expected_mm": expected,
        "scale_factor": scale_factor,
    }


def apply_crop(pcd: o3d.geometry.PointCloud, cfg: Dict[str, Any]) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    require_open3d()
    min_bound = as_vec3(cfg.get("min"), "crop.min")
    max_bound = as_vec3(cfg.get("max"), "crop.max")
    if np.any(max_bound <= min_bound):
        raise ValueError("crop.max must be greater than crop.min on every axis")
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    cropped = pcd.crop(bbox)
    return cropped, {"min": min_bound.tolist(), "max": max_bound.tolist()}


def keep_largest_cluster(pcd: o3d.geometry.PointCloud, cfg: Dict[str, Any]) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    eps = require_positive(cfg.get("eps_mm", 1.5), "cluster.eps_mm")
    min_points = int(require_positive(cfg.get("min_points", 50), "cluster.min_points"))

    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return pcd, {"cluster_count": 0, "kept_label": None, "note": "no labels returned"}

    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        return pcd, {
            "cluster_count": 0,
            "kept_label": None,
            "note": "DBSCAN found no non-noise clusters; point cloud unchanged",
        }

    unique, counts = np.unique(valid_labels, return_counts=True)
    kept_label = int(unique[int(np.argmax(counts))])
    kept_indices = np.where(labels == kept_label)[0]
    filtered = pcd.select_by_index(kept_indices.tolist())
    return filtered, {
        "eps_mm": eps,
        "min_points": min_points,
        "cluster_count": int(unique.size),
        "kept_label": kept_label,
        "kept_cluster_points": int(kept_indices.size),
        "noise_points": int(np.sum(labels < 0)),
    }


def apply_statistical_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    nb_neighbors = int(require_positive(cfg.get("nb_neighbors", 30), "statistical.nb_neighbors"))
    std_ratio = require_positive(cfg.get("std_ratio", 1.5), "statistical.std_ratio")
    filtered, kept_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered, {
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
        "kept_indices": len(kept_indices),
    }


def apply_radius_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    radius = require_positive(cfg.get("radius_mm", 1.0), "radius.radius_mm")
    min_neighbors = int(require_positive(cfg.get("min_neighbors", 8), "radius.min_neighbors"))
    filtered, kept_indices = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    return filtered, {
        "radius_mm": radius,
        "min_neighbors": min_neighbors,
        "kept_indices": len(kept_indices),
    }


def apply_voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size_mm: float,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    voxel = require_positive(voxel_size_mm, "voxel_size_mm")
    filtered = pcd.voxel_down_sample(voxel_size=voxel)
    return filtered, {"voxel_size_mm": voxel}


def estimate_normals(pcd: o3d.geometry.PointCloud, cfg: Dict[str, Any]) -> Dict[str, Any]:
    require_open3d()
    radius = require_positive(cfg.get("radius_mm", 1.0), "normals.radius_mm")
    max_nn = int(require_positive(cfg.get("max_nn", 30), "normals.max_nn"))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # Orienting normals consistently is not always safe for partial object scans.
    return {"radius_mm": radius, "max_nn": max_nn, "note": "normals estimated but not globally reoriented"}


def axis_vector(axis_name: Any, config_name: str) -> np.ndarray:
    axis = str(axis_name).strip().lower()
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    if axis not in axes:
        raise ValueError(f"{config_name} must be one of: x, y, z, -x, -y, -z")
    return axes[axis]


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm <= 0 or target_norm <= 0:
        raise ValueError("Cannot compute rotation from a zero-length vector.")

    a = source / source_norm
    b = target / target_norm
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis = axis / np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    cross = np.cross(a, b)
    cross_matrix = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + cross_matrix + cross_matrix @ cross_matrix * (1.0 / (1.0 + dot))


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if float(eigenvalues[0]) <= 0:
        raise ValueError("PCA alignment failed: point cloud has no measurable variance.")

    if float(np.linalg.det(eigenvectors)) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    total_variance = float(np.sum(np.maximum(eigenvalues, 0.0)))
    explained = []
    if total_variance > 0:
        explained = (eigenvalues / total_variance).tolist()
    return centroid, eigenvalues, eigenvectors, explained


def default_target_axis_names(major_axis_name: Any) -> Dict[str, str]:
    major_name = str(major_axis_name).strip().lower()
    major = axis_vector(major_name, "alignment.target_axis")
    candidates = [
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ]
    middle_name = "y"
    middle = np.array([0.0, 1.0, 0.0])
    for name, vector in candidates:
        if abs(float(np.dot(major, vector))) < 0.5:
            middle_name = name
            middle = vector
            break

    minor = np.cross(major, middle)
    minor_name = vector_axis_name(minor)
    return {
        "major": major_name,
        "middle": middle_name,
        "minor": minor_name,
    }


def vector_axis_name(vector: np.ndarray) -> str:
    axes = [
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ]
    vector = vector / np.linalg.norm(vector)
    best_name = "x"
    best_dot = -1.0
    best_sign = 1.0
    for name, axis in axes:
        dot = float(np.dot(vector, axis))
        if abs(dot) > best_dot:
            best_name = name
            best_dot = abs(dot)
            best_sign = 1.0 if dot >= 0 else -1.0
    return best_name if best_sign > 0 else f"-{best_name}"


def target_frame_from_config(cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, str]]:
    raw_target_axes = cfg.get("target_axes")
    if isinstance(raw_target_axes, dict):
        target_axis_names = {
            "major": str(raw_target_axes.get("major", "x")).strip().lower(),
            "middle": str(raw_target_axes.get("middle", "y")).strip().lower(),
            "minor": str(raw_target_axes.get("minor", "z")).strip().lower(),
        }
    else:
        target_axis_names = default_target_axis_names(cfg.get("target_axis", "x"))

    frame = np.column_stack(
        [
            axis_vector(target_axis_names["major"], "alignment.target_axes.major"),
            axis_vector(target_axis_names["middle"], "alignment.target_axes.middle"),
            axis_vector(target_axis_names["minor"], "alignment.target_axes.minor"),
        ]
    )
    if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-9):
        raise ValueError("alignment.target_axes must contain three different orthogonal axes.")
    determinant = float(np.linalg.det(frame))
    if determinant < 0:
        raise ValueError("alignment.target_axes must form a right-handed frame.")
    return frame, target_axis_names


def choose_pca_axis_signs(source_frame: np.ndarray, target_frame: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    best_score = -float("inf")
    best_frame = source_frame
    best_signs = [1.0, 1.0, 1.0]
    for first in (1.0, -1.0):
        for second in (1.0, -1.0):
            for third in (1.0, -1.0):
                signs = np.array([first, second, third])
                if np.prod(signs) < 0:
                    continue
                candidate = source_frame @ np.diag(signs)
                score = float(np.trace(target_frame.T @ candidate))
                if score > best_score:
                    best_score = score
                    best_frame = candidate
                    best_signs = signs.tolist()
    return best_frame, best_signs


def pca_ambiguity_warnings(eigenvalues: np.ndarray) -> List[str]:
    warnings: List[str] = []
    if float(eigenvalues[0]) <= 0:
        return warnings
    major_middle_gap = float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])
    middle_minor_gap = float((eigenvalues[1] - eigenvalues[2]) / eigenvalues[0])
    if major_middle_gap < 0.05:
        warnings.append(
            "PCA major and middle variances are close; in-plane rotation may be unstable for round or disk-like scans."
        )
    if middle_minor_gap < 0.05:
        warnings.append(
            "PCA middle and minor variances are close; roll alignment may be unstable for nearly symmetric scans."
        )
    return warnings


def apply_major_axis_alignment(
    pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    method = str(cfg.get("method", "pca_major_axis")).lower()
    if method not in {"pca_major_axis", "pca_frame"}:
        raise ValueError("alignment.method must be one of: pca_major_axis, pca_frame")
    if point_count(pcd) < 3:
        raise ValueError("PCA alignment requires at least 3 points.")

    points = np.asarray(pcd.points, dtype=float)
    centroid, eigenvalues, eigenvectors, explained = pca_axes(points)
    source_frame = eigenvectors
    target_axis_names: Dict[str, str]
    source_signs: List[float]

    if method == "pca_major_axis":
        major_axis = np.asarray(source_frame[:, 0], dtype=float)
        target_axis_name = str(cfg.get("target_axis", "x")).strip().lower()
        target = axis_vector(target_axis_name, "alignment.target_axis")
        if float(np.dot(major_axis, target)) < 0:
            major_axis = -major_axis
        rotation = rotation_matrix_from_vectors(major_axis, target)
        target_axis_names = {"major": target_axis_name}
        source_signs = [1.0, 1.0, 1.0]
    else:
        target_frame, target_axis_names = target_frame_from_config(cfg)
        source_frame, source_signs = choose_pca_axis_signs(source_frame, target_frame)
        rotation = target_frame @ source_frame.T

    aligned = copy.deepcopy(pcd)
    aligned.rotate(rotation, center=centroid)

    translation = [0.0, 0.0, 0.0]
    center_at_origin = bool(cfg.get("center_at_origin", False))
    if center_at_origin:
        translation_vector = -np.asarray(aligned.get_center(), dtype=float)
        aligned.translate(translation_vector)
        translation = translation_vector.tolist()

    return aligned, {
        "method": method,
        "target_axis": target_axis_names.get("major"),
        "target_axes": target_axis_names,
        "center_at_origin": center_at_origin,
        "centroid_before": centroid.tolist(),
        "principal_axes_before": eigenvectors.T.tolist(),
        "principal_variances": eigenvalues.tolist(),
        "explained_variance_ratio": explained,
        "source_axis_signs": source_signs,
        "rotation_matrix": rotation.tolist(),
        "translation_after_rotation": translation,
        "warnings": pca_ambiguity_warnings(eigenvalues),
    }


def triangle_count(mesh: o3d.geometry.TriangleMesh) -> int:
    return int(np.asarray(mesh.triangles).shape[0])


def vertex_count(mesh: o3d.geometry.TriangleMesh) -> int:
    return int(np.asarray(mesh.vertices).shape[0])


def estimate_nearest_neighbor_spacing(pcd: o3d.geometry.PointCloud) -> Optional[float]:
    if point_count(pcd) < 2:
        return None
    distances = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size == 0:
        return None
    return float(np.median(distances))


def estimate_mesh_normals(
    pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    require_open3d()
    radius = require_positive(cfg.get("radius_mm", 1.0), "mesh.normals.radius_mm")
    max_nn = int(require_positive(cfg.get("max_nn", 30), "mesh.normals.max_nn"))
    orient_k = int(require_number(
        cfg.get("orient_consistent_tangent_plane_k", 50),
        "mesh.normals.orient_consistent_tangent_plane_k",
    ))
    if orient_k < 0:
        raise ValueError("mesh.normals.orient_consistent_tangent_plane_k must be >= 0")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    oriented = False
    if orient_k > 0:
        if point_count(pcd) <= orient_k:
            warnings.append(
                "Skipped consistent normal orientation because the source cloud has "
                f"{point_count(pcd)} points and k={orient_k}."
            )
        else:
            try:
                pcd.orient_normals_consistent_tangent_plane(orient_k)
                oriented = True
            except RuntimeError as exc:
                warnings.append(f"Could not consistently orient mesh normals: {exc}")

    return {
        "radius_mm": radius,
        "max_nn": max_nn,
        "orient_consistent_tangent_plane_k": orient_k,
        "oriented_consistently": oriented,
    }


def build_ball_pivoting_mesh(
    pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, Any]]:
    require_open3d()
    spacing = estimate_nearest_neighbor_spacing(pcd)
    if spacing is None or spacing <= 0:
        raise ValueError("Cannot run Ball Pivoting: failed to estimate point spacing from the mesh source.")

    raw_multipliers = cfg.get("radius_multipliers", [1.5, 2.5, 4.0])
    if not isinstance(raw_multipliers, list) or not raw_multipliers:
        raise ValueError("mesh.ball_pivoting.radius_multipliers must be a non-empty list of positive numbers.")
    multipliers = [require_positive(value, "mesh.ball_pivoting.radius_multipliers[]") for value in raw_multipliers]
    radii = [spacing * multiplier for multiplier in multipliers]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),
    )
    return mesh, {
        "estimated_spacing_mm": spacing,
        "radius_multipliers": multipliers,
        "pivot_radii_mm": radii,
    }


def build_poisson_mesh(
    pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, Any]]:
    require_open3d()
    depth = int(require_positive(cfg.get("depth", 8), "mesh.poisson.depth"))
    width = int(require_number(cfg.get("width", 0), "mesh.poisson.width"))
    scale = require_positive(cfg.get("scale", 1.1), "mesh.poisson.scale")
    linear_fit = bool(cfg.get("linear_fit", False))
    density_trim_quantile = require_number(
        cfg.get("density_trim_quantile", 0.02),
        "mesh.poisson.density_trim_quantile",
    )
    if width < 0:
        raise ValueError("mesh.poisson.width must be >= 0")
    if not 0.0 <= density_trim_quantile < 1.0:
        raise ValueError("mesh.poisson.density_trim_quantile must be >= 0 and < 1")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit,
    )
    densities_array = np.asarray(densities, dtype=float)
    trim_threshold: Optional[float] = None
    removed_by_density = 0
    if densities_array.size > 0 and density_trim_quantile > 0:
        trim_threshold = float(np.quantile(densities_array, density_trim_quantile))
        remove_mask = densities_array < trim_threshold
        removed_by_density = int(np.sum(remove_mask))
        mesh.remove_vertices_by_mask(remove_mask.tolist())

    source_bbox = pcd.get_axis_aligned_bounding_box()
    spacing = estimate_nearest_neighbor_spacing(pcd)
    extent = np.asarray(source_bbox.get_extent(), dtype=float)
    margin = spacing if spacing is not None else float(np.max(extent)) * 0.005
    margin = max(float(margin), 1e-6)
    crop_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.asarray(source_bbox.get_min_bound(), dtype=float) - margin,
        max_bound=np.asarray(source_bbox.get_max_bound(), dtype=float) + margin,
    )
    mesh = mesh.crop(crop_bbox)

    return mesh, {
        "depth": depth,
        "width": width,
        "scale": scale,
        "linear_fit": linear_fit,
        "density_trim_quantile": density_trim_quantile,
        "density_trim_threshold": trim_threshold,
        "vertices_removed_by_density_trim": removed_by_density,
        "crop_margin_mm": margin,
    }


def cleanup_mesh(mesh: o3d.geometry.TriangleMesh, cfg: Dict[str, Any]) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, Any]]:
    cleaned = copy.deepcopy(mesh)
    details: Dict[str, Any] = {
        "triangles_before_cleanup": triangle_count(cleaned),
        "vertices_before_cleanup": vertex_count(cleaned),
        "steps": [],
    }

    if cfg.get("remove_degenerate_triangles", True):
        cleaned.remove_degenerate_triangles()
        details["steps"].append("remove_degenerate_triangles")
    if cfg.get("remove_duplicated_triangles", True):
        cleaned.remove_duplicated_triangles()
        details["steps"].append("remove_duplicated_triangles")
    if cfg.get("remove_duplicated_vertices", True):
        cleaned.remove_duplicated_vertices()
        details["steps"].append("remove_duplicated_vertices")
    if cfg.get("remove_non_manifold_edges", True):
        if hasattr(cleaned, "remove_non_manifold_edges"):
            cleaned.remove_non_manifold_edges()
            details["steps"].append("remove_non_manifold_edges")
        else:
            details["remove_non_manifold_edges"] = "not supported by this Open3D version"

    details["triangles_after_basic_cleanup"] = triangle_count(cleaned)
    details["vertices_after_basic_cleanup"] = vertex_count(cleaned)

    target_triangles = int(require_positive(cfg.get("target_triangles", 150000), "mesh.cleanup.target_triangles"))
    simplify_enabled = bool(cfg.get("simplify_enabled", True))
    details["simplify_enabled"] = simplify_enabled
    details["target_triangles"] = target_triangles
    details["simplified"] = False
    if simplify_enabled and triangle_count(cleaned) > target_triangles:
        cleaned = cleaned.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        cleaned.remove_degenerate_triangles()
        cleaned.remove_duplicated_triangles()
        cleaned.remove_duplicated_vertices()
        details["simplified"] = True
        details["steps"].append("simplify_quadric_decimation")

    details["triangles_after_cleanup"] = triangle_count(cleaned)
    details["vertices_after_cleanup"] = vertex_count(cleaned)
    return cleaned, details


def mesh_watertight_status(mesh: o3d.geometry.TriangleMesh) -> Optional[bool]:
    if not hasattr(mesh, "is_watertight"):
        return None
    try:
        return bool(mesh.is_watertight())
    except RuntimeError:
        return None


def reconstruct_and_export_mesh(
    source_pcd: o3d.geometry.PointCloud,
    source_label: str,
    cfg: Dict[str, Any],
    output_dir: Path,
    output_name: str,
    ascii_mode: bool,
    allow_overwrite: bool,
) -> MeshReport:
    require_open3d()
    method = str(cfg.get("method", "ball_pivoting")).lower()
    if method not in {"ball_pivoting", "poisson"}:
        raise ValueError("mesh.method must be one of: ball_pivoting, poisson")
    if point_count(source_pcd) < 4:
        raise ValueError("Mesh reconstruction requires at least 4 source points.")

    warnings = [
        "Mesh export is for Fusion reference only.",
        "Verify against the cleaned PLY and physical measurements.",
        "Poisson may invent surfaces or cap holes.",
        "Ball Pivoting may leave holes if scan density is uneven.",
    ]
    mesh_source = copy.deepcopy(source_pcd)
    normal_details = estimate_mesh_normals(mesh_source, cfg.get("normals", {}), warnings)

    if method == "ball_pivoting":
        raw_mesh, reconstruction_details = build_ball_pivoting_mesh(
            mesh_source,
            cfg.get("ball_pivoting", {}),
        )
    else:
        raw_mesh, reconstruction_details = build_poisson_mesh(
            mesh_source,
            cfg.get("poisson", {}),
        )

    before_cleanup = triangle_count(raw_mesh)
    if before_cleanup == 0:
        raise ValueError(
            f"Mesh reconstruction produced 0 triangles using {method}; "
            "try a denser source cloud, different normal radius, or a different mesh method."
        )

    cleaned_mesh, cleanup_details = cleanup_mesh(raw_mesh, cfg.get("cleanup", {}))
    after_cleanup = triangle_count(cleaned_mesh)
    vertices = vertex_count(cleaned_mesh)
    validation_cfg = cfg.get("validation", {})
    min_triangles = int(require_positive(validation_cfg.get("min_triangles", 1000), "mesh.validation.min_triangles"))
    max_triangles = int(require_positive(validation_cfg.get("max_triangles", 300000), "mesh.validation.max_triangles"))
    if max_triangles < min_triangles:
        raise ValueError("mesh.validation.max_triangles must be greater than or equal to min_triangles")
    if after_cleanup < min_triangles:
        raise ValueError(
            f"Mesh reconstruction produced only {after_cleanup} triangles after cleanup; "
            f"minimum useful mesh threshold is {min_triangles}."
        )
    if after_cleanup > max_triangles:
        raise ValueError(
            f"Mesh reconstruction produced {after_cleanup} triangles after cleanup; "
            f"maximum configured threshold is {max_triangles}. Enable simplification or raise the limit."
        )
    if vertices == 0:
        raise ValueError("Mesh cleanup removed all vertices.")

    cleaned_mesh.compute_vertex_normals()
    watertight = mesh_watertight_status(cleaned_mesh)
    warn_if_not_watertight = bool(validation_cfg.get("warn_if_not_watertight", True))
    if warn_if_not_watertight and watertight is False:
        warnings.append("Mesh is not watertight; use it as reference geometry, not final CAD.")

    output_paths: Dict[str, str] = {}
    if cfg.get("export_obj", True):
        obj_path = output_dir / f"{output_name}_fusion_ref.obj"
        written = write_triangle_mesh(obj_path, cleaned_mesh, ascii_mode=True, allow_overwrite=allow_overwrite)
        output_paths["fusion_reference_obj"] = str(written)
    if cfg.get("export_stl", True):
        stl_path = output_dir / f"{output_name}_fusion_ref.stl"
        if ascii_mode:
            warnings.append("Open3D does not support ASCII STL export; wrote binary STL instead.")
        written = write_triangle_mesh(stl_path, cleaned_mesh, ascii_mode=False, allow_overwrite=allow_overwrite)
        output_paths["fusion_reference_stl"] = str(written)
    if not output_paths:
        warnings.append("Mesh reconstruction ran, but OBJ and STL export are both disabled.")

    validation_details = {
        "min_triangles": min_triangles,
        "max_triangles": max_triangles,
        "warn_if_not_watertight": warn_if_not_watertight,
    }
    return MeshReport(
        enabled=True,
        source=source_label,
        method=method,
        normal_estimation=normal_details,
        reconstruction=reconstruction_details,
        cleanup=cleanup_details,
        validation=validation_details,
        triangle_count_before_cleanup=before_cleanup,
        triangle_count_after_cleanup=after_cleanup,
        vertex_count=vertices,
        watertight=watertight,
        output_paths=output_paths,
        warnings=warnings,
    )


def record_operation(
    report: ProcessingReport,
    name: str,
    enabled: bool,
    before: int,
    after: int,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    op = OperationReport(
        name=name,
        enabled=enabled,
        before_points=before,
        after_points=after,
        details=details or {},
    )
    report.operations.append(op)
    if enabled:
        print(f"{name}: {before:,} -> {after:,} points ({before - after:,} removed)")


# -----------------------------
# Reports
# -----------------------------

def write_json_report(path: Path, report: ProcessingReport, allow_overwrite: bool) -> Path:
    final_path = safe_write_path(path, allow_overwrite)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    return final_path


def write_markdown_report(path: Path, report: ProcessingReport, allow_overwrite: bool) -> Path:
    require_yaml()
    final_path = safe_write_path(path, allow_overwrite)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Point Cloud Processing Report")
    lines.append("")
    lines.append(f"- Script: `{report.script}`")
    lines.append(f"- Timestamp UTC: `{report.timestamp_utc}`")
    lines.append(f"- Input: `{report.input_path}`")
    lines.append(f"- Output directory: `{report.output_dir}`")
    lines.append("")

    lines.append("## Statistics")
    lines.append("")
    lines.append("| Stage | Points | Bounding box dimensions | Approx. spacing |")
    lines.append("|---|---:|---:|---:|")
    for s in report.stats:
        spacing = "" if s.approximate_spacing is None else f"{s.approximate_spacing:.4f}"
        lines.append(f"| {s.label} | {s.point_count:,} | {format_dims(s.bbox_dimensions)} | {spacing} |")
    lines.append("")

    lines.append("## Operations")
    lines.append("")
    lines.append("| Operation | Enabled | Before | After | Removed | Details |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for op in report.operations:
        details = json.dumps(op.details, ensure_ascii=False)
        lines.append(
            f"| {op.name} | {op.enabled} | {op.before_points:,} | {op.after_points:,} "
            f"| {op.removed_points:,} | `{details}` |"
        )
    lines.append("")

    lines.append("## Mesh")
    lines.append("")
    mesh = report.mesh
    lines.append(f"- Enabled: `{mesh.enabled}`")
    if mesh.enabled:
        lines.append(f"- Source: `{mesh.source}`")
        lines.append(f"- Method: `{mesh.method}`")
        lines.append(f"- Triangle count before cleanup: `{mesh.triangle_count_before_cleanup}`")
        lines.append(f"- Triangle count after cleanup: `{mesh.triangle_count_after_cleanup}`")
        lines.append(f"- Vertex count: `{mesh.vertex_count}`")
        lines.append(f"- Watertight: `{mesh.watertight}`")
        lines.append("")
        lines.append("### Mesh Details")
        lines.append("")
        lines.append(f"- Normal estimation: `{json.dumps(mesh.normal_estimation, ensure_ascii=False)}`")
        lines.append(f"- Reconstruction: `{json.dumps(mesh.reconstruction, ensure_ascii=False)}`")
        lines.append(f"- Cleanup: `{json.dumps(mesh.cleanup, ensure_ascii=False)}`")
        lines.append(f"- Validation: `{json.dumps(mesh.validation, ensure_ascii=False)}`")
        if mesh.output_paths:
            lines.append("")
            lines.append("### Mesh Outputs")
            lines.append("")
            for key, value in mesh.output_paths.items():
                lines.append(f"- {key}: `{value}`")
        if mesh.warnings:
            lines.append("")
            lines.append("### Mesh Warnings")
            lines.append("")
            for warning in mesh.warnings:
                lines.append(f"- {warning}")
    lines.append("")

    if report.outputs:
        lines.append("## Outputs")
        lines.append("")
        for key, value in report.outputs.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Effective Config")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(report.config, sort_keys=False).rstrip())
    lines.append("```")
    lines.append("")

    final_path.write_text("\n".join(lines), encoding="utf-8")
    return final_path


# -----------------------------
# Main pipeline
# -----------------------------

def process(config: Dict[str, Any]) -> ProcessingReport:
    require_open3d()
    input_value = config.get("input")
    if not input_value:
        raise ValueError("No input file specified. Use --input or set input: in YAML config.")

    input_path = Path(input_value).expanduser().resolve()
    output_dir = Path(config.get("output_dir", "processed")).expanduser().resolve()
    output_name = config.get("output_name") or input_path.stem
    allow_overwrite = bool(config.get("safety", {}).get("allow_overwrite", False))
    ascii_mode = bool(config.get("exports", {}).get("ascii", False))

    report = ProcessingReport(
        script=Path(__file__).name,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        input_path=str(input_path),
        output_dir=str(output_dir),
        config=config,
    )

    print(f"Loading: {input_path}")
    pcd = load_point_cloud(input_path)
    initial_stats = compute_stats(pcd, "input")
    report.stats.append(initial_stats)
    print_stats(initial_stats)

    # Scale
    before = point_count(pcd)
    scale_cfg = config.get("scale", {})
    if scale_cfg.get("enabled"):
        pcd, details = apply_scale(pcd, scale_cfg)
        after = point_count(pcd)
        record_operation(report, "scale", True, before, after, details)
        stats = compute_stats(pcd, "after_scale")
        report.stats.append(stats)
        print_stats(stats)
    else:
        record_operation(report, "scale", False, before, before, {})

    # Crop
    before = point_count(pcd)
    crop_cfg = config.get("crop", {})
    if crop_cfg.get("enabled"):
        pcd, details = apply_crop(pcd, crop_cfg)
        after = point_count(pcd)
        record_operation(report, "crop", True, before, after, details)
        if after == 0:
            raise ValueError("Crop removed all points. Check crop.min/crop.max.")
        report.stats.append(compute_stats(pcd, "after_crop"))
    else:
        record_operation(report, "crop", False, before, before, {})

    # Keep largest cluster
    before = point_count(pcd)
    cluster_cfg = config.get("cluster", {})
    if cluster_cfg.get("keep_largest"):
        pcd, details = keep_largest_cluster(pcd, cluster_cfg)
        after = point_count(pcd)
        record_operation(report, "keep_largest_cluster", True, before, after, details)
        if after == 0:
            raise ValueError("Cluster filtering removed all points. Use larger eps_mm or disable cluster filtering.")
        report.stats.append(compute_stats(pcd, "after_largest_cluster"))
    else:
        record_operation(report, "keep_largest_cluster", False, before, before, {})

    # Statistical outlier removal
    before = point_count(pcd)
    stat_cfg = config.get("outlier_removal", {}).get("statistical", {})
    if stat_cfg.get("enabled"):
        pcd, details = apply_statistical_outlier_removal(pcd, stat_cfg)
        after = point_count(pcd)
        record_operation(report, "statistical_outlier_removal", True, before, after, details)
        if after == 0:
            raise ValueError("Statistical outlier removal removed all points. Use less aggressive parameters.")
        report.stats.append(compute_stats(pcd, "after_statistical_outlier_removal"))
    else:
        record_operation(report, "statistical_outlier_removal", False, before, before, {})

    # Radius outlier removal
    before = point_count(pcd)
    radius_cfg = config.get("outlier_removal", {}).get("radius", {})
    if radius_cfg.get("enabled"):
        pcd, details = apply_radius_outlier_removal(pcd, radius_cfg)
        after = point_count(pcd)
        record_operation(report, "radius_outlier_removal", True, before, after, details)
        if after == 0:
            raise ValueError("Radius outlier removal removed all points. Use less aggressive parameters.")
        report.stats.append(compute_stats(pcd, "after_radius_outlier_removal"))
    else:
        record_operation(report, "radius_outlier_removal", False, before, before, {})

    # Main downsample
    before = point_count(pcd)
    down_cfg = config.get("downsample", {})
    if down_cfg.get("enabled"):
        pcd, details = apply_voxel_downsample(pcd, down_cfg.get("voxel_size_mm"))
        after = point_count(pcd)
        record_operation(report, "voxel_downsample", True, before, after, details)
        if after == 0:
            raise ValueError("Voxel downsampling removed all points. Use a smaller voxel_size_mm.")
        report.stats.append(compute_stats(pcd, "after_voxel_downsample"))
    else:
        record_operation(report, "voxel_downsample", False, before, before, {})

    # Normals
    before = point_count(pcd)
    normals_cfg = config.get("normals", {})
    if normals_cfg.get("enabled"):
        details = estimate_normals(pcd, normals_cfg)
        after = point_count(pcd)
        record_operation(report, "estimate_normals", True, before, after, details)
    else:
        record_operation(report, "estimate_normals", False, before, before, {})

    # Optional rigid alignment for downstream reference use.
    before = point_count(pcd)
    alignment_cfg = config.get("alignment", {})
    if alignment_cfg.get("enabled"):
        pcd, details = apply_major_axis_alignment(pcd, alignment_cfg)
        after = point_count(pcd)
        record_operation(report, "major_axis_alignment", True, before, after, details)
        for warning in details.get("warnings", []):
            report.warnings.append(f"PCA alignment: {warning}")
        report.stats.append(compute_stats(pcd, "after_major_axis_alignment"))
    else:
        record_operation(report, "major_axis_alignment", False, before, before, {})

    final_stats = compute_stats(pcd, "final_cleaned")
    report.stats.append(final_stats)
    print_stats(final_stats)

    # Exports
    exports = config.get("exports", {})
    mesh_cfg = config.get("mesh", {})
    mesh_enabled = bool(mesh_cfg.get("enabled", False))
    mesh_source = str(mesh_cfg.get("source", "fusion_reference")).lower()
    if mesh_enabled and mesh_source not in {"fusion_reference", "cleaned"}:
        raise ValueError("mesh.source must be one of: fusion_reference, cleaned")

    if exports.get("cleaned_ply", True):
        cleaned_path = output_dir / f"{output_name}_cleaned.ply"
        written = write_point_cloud(cleaned_path, pcd, ascii_mode, allow_overwrite)
        report.outputs["cleaned_ply"] = str(written)
        print(f"Wrote cleaned PLY: {written}")

    fusion_pcd: Optional[o3d.geometry.PointCloud] = None
    needs_fusion_reference = bool(exports.get("fusion_reference_ply", True)) or (
        mesh_enabled and mesh_source == "fusion_reference"
    )
    if needs_fusion_reference:
        fusion_voxel = require_positive(exports.get("fusion_voxel_size_mm", 0.5), "exports.fusion_voxel_size_mm")
        before = point_count(pcd)
        fusion_pcd, details = apply_voxel_downsample(pcd, fusion_voxel)
        after = point_count(fusion_pcd)
        record_operation(report, "fusion_reference_downsample", True, before, after, details)
        if after == 0:
            raise ValueError("Fusion-reference downsampling removed all points. Use a smaller fusion_voxel_size_mm.")
        report.stats.append(compute_stats(fusion_pcd, "fusion_reference"))
        if exports.get("fusion_reference_ply", True):
            fusion_path = output_dir / f"{output_name}_fusion_ref.ply"
            written = write_point_cloud(fusion_path, fusion_pcd, ascii_mode, allow_overwrite)
            report.outputs["fusion_reference_ply"] = str(written)
            print(f"Wrote Fusion reference PLY: {written}")

    if mesh_enabled:
        if mesh_source == "fusion_reference":
            if fusion_pcd is None:
                raise RuntimeError("Internal error: fusion reference source was not prepared for mesh export.")
            source_pcd = fusion_pcd
        else:
            source_pcd = pcd

        report.mesh = reconstruct_and_export_mesh(
            source_pcd=source_pcd,
            source_label=mesh_source,
            cfg=mesh_cfg,
            output_dir=output_dir,
            output_name=output_name,
            ascii_mode=ascii_mode,
            allow_overwrite=allow_overwrite,
        )
        report.outputs.update(report.mesh.output_paths)
        report.warnings.extend(report.mesh.warnings)
        print(
            "Mesh export: "
            f"{report.mesh.triangle_count_before_cleanup:,} -> "
            f"{report.mesh.triangle_count_after_cleanup:,} triangles"
        )
        for key, value in report.mesh.output_paths.items():
            print(f"Wrote {key}: {value}")

    # Reports
    report_base = output_dir / f"{output_name}_report"
    report_cfg = config.get("report", {})
    json_report_path: Optional[Path] = None
    markdown_report_path: Optional[Path] = None

    if report_cfg.get("json", True):
        json_report_path = safe_write_path(report_base.with_suffix(".json"), allow_overwrite)
        report.outputs["json_report"] = str(json_report_path)
    if report_cfg.get("markdown", True):
        markdown_report_path = safe_write_path(report_base.with_suffix(".md"), allow_overwrite)
        report.outputs["markdown_report"] = str(markdown_report_path)

    if json_report_path is not None:
        written = write_json_report(json_report_path, report, allow_overwrite=True)
        print(f"Wrote JSON report: {written}")
    if markdown_report_path is not None:
        written = write_markdown_report(markdown_report_path, report, allow_overwrite=True)
        print(f"Wrote Markdown report: {written}")

    return report


# -----------------------------
# CLI and sample config
# -----------------------------

SAMPLE_CONFIG_TEXT = """# scan_config.yaml
# Conservative v1 config for Revo Metro PLY point-cloud preprocessing.

input: scans/raw_object.ply
output_dir: processed
output_name: object_cleaned

scale:
  enabled: false
  known_axis: x        # x, y, z, or longest
  expected_mm: 82.5

crop:
  enabled: false
  min: [-50, -50, -10]
  max: [50, 50, 90]

cluster:
  keep_largest: false
  eps_mm: 1.5
  min_points: 50

outlier_removal:
  statistical:
    enabled: true
    nb_neighbors: 30
    std_ratio: 1.5
  radius:
    enabled: false
    radius_mm: 1.0
    min_neighbors: 8

downsample:
  enabled: true
  voxel_size_mm: 0.25

normals:
  enabled: false
  radius_mm: 1.0
  max_nn: 30

alignment:
  enabled: false
  method: pca_frame     # pca_frame or pca_major_axis
  target_axis: x        # legacy pca_major_axis target: x, y, z, -x, -y, or -z
  target_axes:
    major: x            # Longest PCA axis.
    middle: y           # Second PCA axis, useful for T arms and disk in-plane roll.
    minor: z            # Shortest PCA axis, useful as disk/plate normal.
  center_at_origin: false

exports:
  cleaned_ply: true
  fusion_reference_ply: true
  fusion_voxel_size_mm: 0.5
  ascii: false

mesh:
  enabled: false
  source: fusion_reference        # fusion_reference or cleaned
  method: ball_pivoting           # ball_pivoting or poisson
  export_obj: true
  export_stl: true

  normals:
    radius_mm: 1.0
    max_nn: 30
    orient_consistent_tangent_plane_k: 50

  ball_pivoting:
    radius_multipliers: [1.5, 2.5, 4.0]

  poisson:
    depth: 8
    width: 0
    scale: 1.1
    linear_fit: false
    density_trim_quantile: 0.02

  cleanup:
    remove_degenerate_triangles: true
    remove_duplicated_triangles: true
    remove_duplicated_vertices: true
    remove_non_manifold_edges: true
    simplify_enabled: true
    target_triangles: 150000

  validation:
    min_triangles: 1000
    max_triangles: 300000
    warn_if_not_watertight: true

report:
  markdown: true
  json: true

safety:
  allow_overwrite: false
"""


def write_sample_config(path: Path) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_CONFIG_TEXT, encoding="utf-8")
    print(f"Wrote sample config: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conservative point-cloud cleanup/downsampling for Revo Metro PLY exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="YAML config file.")
    parser.add_argument("--write-sample-config", type=str, help="Write a sample YAML config and exit.")
    parser.add_argument("--input", type=str, help="Input PLY point cloud.")
    parser.add_argument("--output-dir", type=str, help="Directory for output files.")
    parser.add_argument("--output-name", type=str, help="Base name for output files.")
    parser.add_argument("--known-x-mm", type=float, help="Scale so X dimension matches this measurement in mm.")
    parser.add_argument("--known-y-mm", type=float, help="Scale so Y dimension matches this measurement in mm.")
    parser.add_argument("--known-z-mm", type=float, help="Scale so Z dimension matches this measurement in mm.")
    parser.add_argument("--known-longest-mm", type=float, help="Scale so longest bbox dimension matches this measurement in mm.")
    parser.add_argument("--voxel-mm", type=float, help="Main voxel downsample size in mm.")
    parser.add_argument("--no-statistical-outlier", action="store_true", help="Disable statistical outlier removal.")
    parser.add_argument("--keep-largest-cluster", action="store_true", help="Enable DBSCAN largest-cluster filter.")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow overwriting outputs.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.write_sample_config:
            write_sample_config(Path(args.write_sample_config))
            return 0

        config = load_config(Path(args.config).expanduser().resolve() if args.config else None)
        config = apply_cli_overrides(config, args)
        process(config)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
