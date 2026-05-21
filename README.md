# Revo Metro Point-Cloud Preprocessor

Conservative Python/Open3D preprocessing for PLY point clouds exported from Revo Metro before using them as reference geometry in Autodesk Fusion.

This is intentionally not a scan-to-CAD or automatic reconstruction tool. It keeps the workflow low-step and reportable:

```text
Revo Metro PLY -> clean_scan.py -> cleaned PLY + optional Fusion reference OBJ/STL + reports -> Fusion reference workflow
```

## Setup

Use a Python version supported by Open3D wheels. CPython 3.10, 3.11, and 3.12 are the safest choices across Windows, macOS, and Linux.

```bash
python -m pip install -r requirements.txt
```

If `open3d` reports "No matching distribution found", create the environment with Python 3.12 and run the install command again.

## Smoke Test

```bash
python clean_scan.py --write-sample-config scratch/scan_config.yaml
pytest
```

The sample-config writer refuses to overwrite existing files. To regenerate `examples/scan_config.yaml`, remove the old file first or write to a new path.

## Revo Metro Export Guidance

Export the scan as a PLY point cloud. Keep the raw export unchanged and treat it as source data.

Recommended local file set for a scan:

```text
scans/raw_object.ply                 # primary input for this script
scans/raw_object_visual_mesh.obj     # optional visual reference only
scans/raw_object_revo_project/       # optional native/project backup
```

Before export, prefer:

- Units in millimeters when available.
- Enough scan overlap to preserve edges and thin features.
- No aggressive smoothing, hole filling, or mesh decimation as the primary source.
- A known caliper measurement on X, Y, Z, or the longest axis for scale verification.

## Basic Usage

Create a config, edit paths and conservative cleanup values, then run:

```bash
python clean_scan.py --config examples/scan_config.yaml
```

Common CLI overrides:

```bash
python clean_scan.py --input scans/raw_object.ply --output-dir processed --known-x-mm 82.5 --voxel-mm 0.25
```

Outputs are written to `processed/` by default:

- `<name>_cleaned.ply`
- `<name>_fusion_ref.ply`
- `<name>_fusion_ref.obj` when mesh export is enabled
- `<name>_fusion_ref.stl` when mesh export is enabled
- `<name>_report.json`
- `<name>_report.md`

Every destructive cleanup step records before/after point counts in the reports.
Mesh export is disabled by default and must be enabled explicitly in YAML.

Optional PCA alignment can rotate the processed output files so the cleaned
point cloud's principal axes land on a chosen X/Y/Z frame. This is a rigid
reference transform, not CAD feature recognition.

## Fusion Workflow Guidance

Keep `<name>_cleaned.ply` as the authoritative cleaned scan intermediate. Fusion does not reliably open point-cloud PLY files, so the optional mesh export exists only to get reference geometry into Fusion.

When mesh export is enabled, prefer:

- OBJ as the primary Fusion mesh import.
- STL as a fallback if OBJ import is awkward in a specific Fusion workflow.
- The cleaned PLY and physical measurements as the authority for scale and dimensions.

Mesh output is not dimensionally authoritative CAD. Ball Pivoting can leave holes where scan density is uneven, and Poisson reconstruction can invent surfaces or cap holes.
STL output is written as binary STL because Open3D 0.19 does not support ASCII STL export.

Suggested flow:

1. Import or insert `<name>_fusion_ref.obj` into Fusion as reference geometry.
2. Verify scale against the same known measurement used in the config.
3. Create sketches, construction planes, axes, and solid features from measured references.
4. Keep the scan/reference body separate from the rebuilt CAD body.
5. Avoid treating the script output as a finished CAD model; do not rely on automatic mesh-to-BRep conversion for final geometry unless you intentionally validate it.

The `fusion_reference_ply` export is usually more downsampled than the cleaned PLY so downstream reference workflows stay responsive. It is also the default source for optional mesh reconstruction.

## PCA Alignment

PCA alignment is disabled by default. When enabled, it runs after cleanup and
before output export, so the cleaned PLY, Fusion-reference PLY, OBJ/STL mesh,
and reports all use the aligned coordinates.

The recommended `pca_frame` method aligns all three principal axes. This removes
the unconstrained roll left by single-axis alignment, which matters for T-shaped
objects and disk-like objects.

```yaml
alignment:
  enabled: true
  method: pca_frame     # pca_frame or pca_major_axis
  target_axis: x        # legacy pca_major_axis target: x, y, z, -x, -y, or -z
  target_axes:
    major: x            # Longest PCA axis.
    middle: y           # Second PCA axis, useful for T arms and disk in-plane roll.
    minor: z            # Shortest PCA axis, useful as disk/plate normal.
  center_at_origin: false
```

The report records the PCA centroid, principal axes, variances, rotation matrix,
and optional translation. Keep verifying scale and orientation against the
cleaned PLY and physical measurements.

For a long object where the length should run up Fusion's Y axis, use:

```yaml
alignment:
  enabled: true
  method: pca_frame
  target_axes:
    major: y
    middle: x
    minor: -z
```

For disk-like scans, mapping `minor: z` usually aligns the disk with the XY
plane. If the disk is close to round, the in-plane angle can still be arbitrary
because PCA cannot infer a meaningful clocking direction from a symmetric disk.

## Mesh Reconstruction Options

Mesh reconstruction is conservative and opt-in:

- `ball_pivoting` uses local point spacing and configured radius multipliers. It tends to preserve scan gaps but may leave holes on uneven or sparse captures.
- `poisson` uses Screened Poisson reconstruction. It can produce smoother, more continuous reference meshes, but it may invent surfaces or cap holes.

Example mesh-enabled config section:

```yaml
mesh:
  enabled: true
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
```

## Configuration

See [examples/scan_config.yaml](examples/scan_config.yaml). Defaults are conservative:

- Scale correction is off until a known measurement is supplied.
- Cropping is off until explicit bounds are supplied.
- Largest-cluster filtering is off by default.
- Statistical outlier removal is on with mild settings.
- Radius outlier removal and normal estimation are off by default.
- PCA alignment is off by default.
- Mesh reconstruction is not performed.

## Tests

The test suite creates synthetic point clouds and does not require real scan files:

```bash
pytest
```

The Open3D-dependent tests cover scale correction, crop behavior, voxel downsampling, statistical outlier removal, full report generation, CLI processing of a synthetic PLY, and optional OBJ/STL mesh export from synthetic point clouds.
