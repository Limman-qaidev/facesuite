# FaceSuite — Face-based Photo Organizer

**FaceSuite** scans a photo directory (images-first, then subfolders), detects faces with InsightFace, clusters common faces, exports crops for manual labeling, and then organizes your entire library into per-person folders (and a `group/` folder with multi-person photos renamed to include all detected names).

## Features
- Ordered traversal: process images **before** descending into subfolders.
- Face detection & embeddings via **InsightFace (buffalo_l)**.
- Unsupervised clustering with **DBSCAN (cosine)** to find common faces.
- Manual labeling step: rename/merge cluster folders to human names.
- Full-library organization into `dest/<PersonName>/` or `dest/group/` with filename renaming.
- Persisted DB (`faces_db.parquet` + `faces_db.json`), CSV report, and configurable thresholds.

## Installation
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart
### Phase 1 — Scan, Cluster & Export Faces
```bash
python -m facesuite.cli scan --src "/path/photos" --work "/path/work"
python -m facesuite.cli cluster --work "/path/work" --eps 0.35 --min-samples 2
python -m facesuite.cli export-faces --work "/path/work"
```
👉 Then go to `work/faces_to_label/` and rename/merge folders to real person names.

### Phase 2 — Organize
```bash
python -m facesuite.cli organize   --src "/path/photos"   --work "/path/work"   --dest "/path/output"   --assign-th 0.35   --mode copy   # or move | symlink
```

## Threshold tips
- Start with `--eps 0.35` (clustering) and `--assign-th 0.35` (assignment).
- If different people end up in one cluster, **decrease** `--eps` (e.g., 0.30).
- If Phase 2 misses matches, **decrease** `--assign-th` slightly (e.g., 0.32).

## Roadmap
- Optional FAISS index for large datasets.
- Simple Streamlit UI for cluster review & naming.
- GPU support notes and benchmarks.

## License
MIT License © 2025 Jonathan
