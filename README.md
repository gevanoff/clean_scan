# Revo Metro Point-Cloud Preprocessor

Conservative Python/Open3D preprocessing for PLY point clouds exported from Revo Metro before using them as reference geometry in Autodesk Fusion.

This is intentionally not a scan-to-CAD or automatic reconstruction tool. It keeps the workflow low-step and reportable:

```text
Revo Metro PLY -> clean_scan.py -> cleaned PLY + Fusion reference PLY + reports -> Fusion reference workflow
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
- `<name>_report.json`
- `<name>_report.md`

Every destructive cleanup step records before/after point counts in the reports.

## Fusion Workflow Guidance

Use the Fusion reference PLY as visual/reference geometry, then rebuild the part parametrically.

Suggested flow:

1. Import or insert the cleaned/Fusion reference PLY into Fusion as reference geometry.
2. Verify scale against the same known measurement used in the config.
3. Create sketches, construction planes, axes, and solid features from measured references.
4. Keep the scan/reference body separate from the rebuilt CAD body.
5. Avoid treating the script output as a finished CAD model; do not rely on automatic mesh-to-BRep conversion for final geometry unless you intentionally validate it.

The `fusion_reference_ply` export is usually more downsampled than the cleaned PLY so Fusion stays responsive.

## Configuration

See [examples/scan_config.yaml](examples/scan_config.yaml). Defaults are conservative:

- Scale correction is off until a known measurement is supplied.
- Cropping is off until explicit bounds are supplied.
- Largest-cluster filtering is off by default.
- Statistical outlier removal is on with mild settings.
- Radius outlier removal and normal estimation are off by default.
- Mesh reconstruction is not performed.

## Tests

The test suite creates synthetic point clouds and does not require real scan files:

```bash
pytest
```

The Open3D-dependent tests cover scale correction, crop behavior, voxel downsampling, statistical outlier removal, full report generation, and CLI processing of a synthetic PLY.
