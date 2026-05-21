#!/usr/bin/env python3
"""
clean_scan.py

Conservative point-cloud preprocessing for Revo Metro / Revopoint-style scan exports
before use as reference geometry in Autodesk Fusion.

Primary workflow:
    Revo Metro export PLY point cloud
      -> python clean_scan.py --config scan_config.yaml
      -> cleaned PLY + optional Fusion-reference PLY + reports
      -> Insert mesh/point reference in Fusion and rebuild CAD parametrically

This script deliberately avoids automatic CAD reconstruction. It performs reversible,
reported preprocessing: scale verification, crop, cluster filtering, outlier removal,
voxel downsampling, optional normal estimation, and export.

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
    "exports": {
        "cleaned_ply": True,
        "fusion_reference_ply": True,
        "fusion_voxel_size_mm": 0.5,
        "ascii": False,
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

    final_stats = compute_stats(pcd, "final_cleaned")
    report.stats.append(final_stats)
    print_stats(final_stats)

    # Exports
    exports = config.get("exports", {})

    if exports.get("cleaned_ply", True):
        cleaned_path = output_dir / f"{output_name}_cleaned.ply"
        written = write_point_cloud(cleaned_path, pcd, ascii_mode, allow_overwrite)
        report.outputs["cleaned_ply"] = str(written)
        print(f"Wrote cleaned PLY: {written}")

    if exports.get("fusion_reference_ply", True):
        fusion_voxel = require_positive(exports.get("fusion_voxel_size_mm", 0.5), "exports.fusion_voxel_size_mm")
        before = point_count(pcd)
        fusion_pcd, details = apply_voxel_downsample(pcd, fusion_voxel)
        after = point_count(fusion_pcd)
        record_operation(report, "fusion_reference_downsample", True, before, after, details)
        fusion_path = output_dir / f"{output_name}_fusion_ref.ply"
        written = write_point_cloud(fusion_path, fusion_pcd, ascii_mode, allow_overwrite)
        report.outputs["fusion_reference_ply"] = str(written)
        report.stats.append(compute_stats(fusion_pcd, "fusion_reference"))
        print(f"Wrote Fusion reference PLY: {written}")

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

exports:
  cleaned_ply: true
  fusion_reference_ply: true
  fusion_voxel_size_mm: 0.5
  ascii: false

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
