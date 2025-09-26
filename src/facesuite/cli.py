"""
facesuite.py — Two‑phase project to identify faces, prepare manual labeling
 folders,
and organize your photo library by person, exactly following your requested
 flow.

Phase 1
-------
1) Walk a directory tree **processing current‑level images first, then
 subfolders**.
2) Detect faces and compute embeddings with InsightFace (buffalo_l via
 onnxruntime).
3) Cluster faces with DBSCAN (cosine metric) to discover "common faces"
 (same person).
4) Export **crops** of each face into `faces_to_label/<cluster_id>/...`.
 Here you will:
   - Rename each folder to the **person's name** (e.g., `Luna`,
     `Jonathan`, `Dad`).
   - Merge folders if two clusters represent the **same person**.

Phase 2
-------
1) Read `faces_to_label/` after your manual changes. Each subfolder becomes a
 **person**
   (name = subfolder name). We compute person centroids.
2) Classify **all original photos**:
   - If an image has exactly 1 person ⇒ copy/move to `dest/<PersonName>/`.
   - If an image has >1 person ⇒ copy/move to `dest/group/` and rename the
     file to
     include all names separated by hyphens, e.g. `Luna-Jonathan-IMG_0001.jpg`.

Extras
------
- Persist every detection/embedding in `faces_db.parquet` + `faces_db.json`
 (metadata) for debugging.
- Write a CSV report of assignments to `organize_report.csv`.
- Output modes: copy | move | symlink (default: copy).
- Tunable thresholds; sensible defaults for buffalo_l (cosine similarity
 space).

Requirements
------------
- Python 3.9+
- pip install: insightface onnxruntime opencv-python pillow numpy scikit-learn
 pandas pyarrow tqdm

Quick start
-----------
Phase 1 (scan + cluster + export crops):
    python facesuite.py scan --src "/path/photos" --work "/path/work"
    python facesuite.py cluster --work "/path/work" --eps 0.35 --min-samples 2
    python facesuite.py export-faces --work "/path/work"

(Now **you** rename/merge folders inside `work/faces_to_label/` to set names
 and merge duplicates.)

Phase 2 (organize):
    python facesuite.py organize --src "/path/photos" --work "/path/work"
      --dest "/path/output" \
        --assign-th 0.35 --mode copy
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Optional: silence noisy backends
os.environ.setdefault("INSIGHTFACE_TORCH_MPS", "0")

try:
    from insightface.app import FaceAnalysis
except Exception as e:  # pragma: no cover (installation)
    print(
        "[ERROR] You must install insightface and onnxruntime:"
        "  pip install insightface onnxruntime"
        "Original error: " + str(e)
    )
    raise

try:
    import cv2  # opencv-python
except Exception as e:  # pragma: no cover
    print("[ERROR] You must install opencv-python: pip install"
          " opencv-python" + str(e))
    raise


# --------------------------- FS & Logging utilities --------------------------

logger = logging.getLogger("facesuite")


def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


# --------------------------- Detection & embeddings --------------------------

@dataclass
class FaceRecord:
    image_path: str
    rel_dir: str  # relative folder wrt the src root
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    embedding: List[float]
    det_score: float


class FaceEncoder:
    """Small wrapper over InsightFace (buffalo_l)."""

    def __init__(self, providers: Optional[List[str]] = None) -> None:
        self.app = FaceAnalysis(name="buffalo_l")
        providers = providers or ["CPUExecutionProvider"]
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace loaded (buffalo_l)")

    def extract(
            self,
            img_bgr: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
        faces = self.app.get(img_bgr)
        results = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox.astype(int))
            emb = f.normed_embedding  # already L2-normalized
            results.append(((x1, y1, x2, y2), float(f.det_score), emb))
        return results


# --------------------------- Ordered recursive walk -------------------------

def iter_dir_images_first(root: Path) -> Iterable[Path]:
    """
    Yield images first in the current level, then recurse into subfolders.
    """
    items = list(root.iterdir())
    for p in sorted([x for x in items if x.is_file() and is_image(x)]):
        yield p
    for d in sorted([x for x in items if x.is_dir()]):
        yield from iter_dir_images_first(d)


# --------------------------- Face DB ---------------------------

@dataclass
class DBMeta:
    src_root: str
    work_dir: str
    embeddings_parquet: str
    meta_json: str
    faces_to_label_dir: str


class FaceDB:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = work_dir / "faces_db.parquet"
        self.meta_path = work_dir / "faces_db.json"
        self.faces_to_label = work_dir / "faces_to_label"
        self.faces_to_label.mkdir(exist_ok=True)

    def save(self, df: pd.DataFrame, meta: Dict) -> None:
        df.to_parquet(self.emb_path, index=False)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.emb_path)

    def load_meta(self) -> Dict:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def summary(self) -> str:
        parts = []
        parts.append(f"DB: {self.emb_path}")
        if self.emb_path.exists():
            df = self.load_df()
            parts.append(
                f"  records: {len(df)} |"
                f" unique images: {df['image_path'].nunique()}"
                )
        else:
            parts.append("  [empty]")
        return "".join(parts)


# --------------------------- Pipeline: SCAN ---------------------------

def cmd_scan(
        src: Path,
        work: Path,
        min_size: int = 60,
        verbosity: int = 1
) -> None:
    setup_logging(verbosity)
    logger.info("Scanning…")

    enc = FaceEncoder()
    records: List[FaceRecord] = []

    src = src.resolve()
    work = work.resolve()
    db = FaceDB(work)

    for img_path in tqdm(list(iter_dir_images_first(src)), desc="Images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Cannot read: {img_path}")
                continue
            detections = enc.extract(img)
            rel_dir = str(
                img_path.parent.relative_to(src)
                ) if img_path.parent != src else ""
            for (x1, y1, x2, y2), score, emb in detections:
                w, h = x2 - x1, y2 - y1
                if min(w, h) < min_size:
                    continue
                rec = FaceRecord(
                    image_path=str(img_path),
                    rel_dir=rel_dir,
                    bbox=(x1, y1, x2, y2),
                    embedding=emb.tolist(),
                    det_score=score,
                )
                records.append(rec)
        except Exception as e:  # pragma: no cover
            logger.exception(f"Error processing {img_path}: {e}")

    if not records:
        logger.warning("No faces detected.")

    df = pd.DataFrame([asdict(r) for r in records])
    meta = asdict(DBMeta(
        src_root=str(src),
        work_dir=str(work),
        embeddings_parquet=str(db.emb_path),
        meta_json=str(db.meta_path),
        faces_to_label_dir=str(db.faces_to_label),
    ))
    db.save(df, meta)
    print(db.summary())


# --------------------------- Pipeline: CLUSTER ---------------------------

def cmd_cluster(
        work: Path,
        eps: float = 0.35,
        min_samples: int = 2,
        verbosity: int = 1
) -> None:
    setup_logging(verbosity)
    db = FaceDB(work)
    df = db.load_df()
    if df.empty:
        print("[WARN] Empty DB. Run `scan` first.")
        return

    X = np.vstack(df["embedding"].values).astype(np.float32)
    X = normalize(X)  # ensure L2-normalized (InsightFace already does,
    # but just in case)

    # DBSCAN in cosine space => distance = 1 - cosine_similarity
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(X)
    df["cluster_id"] = labels

    df.to_parquet(db.emb_path, index=False)

    n_noise = int((labels == -1).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))

    print(
        f"Formed clusters: {n_clusters} |"
        f" Noise: {n_noise} | Total faces: {len(df)}"
        f"(Hint: tune --eps if you see over/under-clustering)"
    )


# --------------------------- Pipeline: EXPORT-FACES (Phase 1) -------------

def _safe_name(s: str) -> str:
    s = s.strip().replace(os.sep, "-")
    return "".join(c for c in s if c.isalnum() or c in "-_. ").strip()


def crop_and_save(
        img_path: Path,
        bbox: Tuple[int, int, int, int],
        out_path: Path,
        margin: float = 0.25
) -> None:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    crop = img.crop((x1, y1, x2, y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)


def cmd_export_faces(work: Path, top_k_per_cluster: int = 40,
                     verbosity: int = 1) -> None:
    setup_logging(verbosity)
    db = FaceDB(work)
    df = db.load_df()
    if "cluster_id" not in df.columns:
        print("[WARN] No clusters found. Run `cluster` before `export-faces`.")
        return

    df_sorted = df.sort_values(["cluster_id", "det_score"],
                               ascending=[True, False])

    counts: Dict[int, int] = {}
    skipped = 0
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted),
                       desc="Exporting crops"):
        cid = int(row["cluster_id"])  # -1 = noise
        if cid == -1:
            continue
        cnt = counts.get(cid, 0)
        if cnt >= top_k_per_cluster:
            continue
        img_path = Path(row["image_path"])  # absolute
        bbox = tuple(int(x) for x in row["bbox"])  # type: ignore
        out_dir = db.faces_to_label / f"cluster_{cid:04d}"
        out_name = _safe_name(img_path.stem) + f"_{cnt:03d}.jpg"
        out_path = out_dir / out_name
        try:
            crop_and_save(img_path, bbox, out_path)
            counts[cid] = cnt + 1
        except Exception as e:  # pragma: no cover
            logger.warning(f"Error exporting {img_path}: {e}")
            skipped += 1

    print(
        f"Done. Review and RENAME/MERGE folders in: {db.faces_to_label}"
        "- Each subfolder represents a person. Give it a human name (e.g.,"
        " 'Luna')."
        "- If two folders are the same person, merge their content into one."
        f"(Skipped due to errors: {skipped})"
    )


# --------------------------- Pipeline: ORGANIZE (Phase 2) ------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # a,b are L2-normalized


def build_label_centroids(label_dir: Path) -> Dict[str, np.ndarray]:
    """Each subfolder => person. The person's embedding = mean of its crop
      embeddings.
    We recompute embeddings from the crops using InsightFace (simple & robust).
    """
    encoder = FaceEncoder()

    def collect_embeds_from_crops(person_dir: Path) -> List[np.ndarray]:
        embs: List[np.ndarray] = []
        for p in person_dir.glob("**/*"):
            if p.is_file() and is_image(p):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                faces = encoder.extract(img)
                if not faces:
                    continue
                faces.sort(key=lambda r: (r[0][2] - r[0][0]) * (
                    r[0][3] - r[0][1]), reverse=True)
                embs.append(faces[0][2])
        return embs

    centroids: Dict[str, np.ndarray] = {}
    for sub in sorted([d for d in label_dir.iterdir() if d.is_dir()]):
        name = sub.name
        embs = collect_embeds_from_crops(sub)
        if not embs:
            logger.warning(f"[organize] No embeddings for: {sub}")
            continue
        X = normalize(np.vstack(embs))
        centroid = np.mean(X, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[name] = centroid.astype(np.float32)

    if not centroids:
        raise RuntimeError(
            "No labeled person folders found in faces_to_label/."
            )

    return centroids


def assign_face_labels(emb: np.ndarray, centroids: Dict[str, np.ndarray],
                       th: float) -> Optional[str]:
    best_name = None
    best_sim = -1.0
    for name, c in centroids.items():
        s = cosine_sim(emb, c)
        if s > best_sim:
            best_sim = s
            best_name = name
    if best_sim >= th:
        return best_name
    return None


def cmd_organize(src: Path, work: Path, dest: Path, assign_th: float = 0.35,
                 mode: str = "copy", verbosity: int = 1) -> None:
    setup_logging(verbosity)
    db = FaceDB(work)
    df = db.load_df()
    if df.empty:
        print("[WARN] Empty DB. Run `scan` and `cluster` first.")
        return

    centroids = build_label_centroids(db.faces_to_label)
    logger.info(f"Labeled persons (Phase 2): {list(centroids.keys())}")

    dest.mkdir(parents=True, exist_ok=True)
    group_dir = dest / "group"
    group_dir.mkdir(exist_ok=True)

    # Index faces by image
    by_img: Dict[str, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = {}
    for _, row in df.iterrows():
        img_path = row["image_path"]
        emb = np.array(row["embedding"], dtype=np.float32)
        emb = emb / max(np.linalg.norm(emb), 1e-8)
        bbox = tuple(int(x) for x in row["bbox"])  # type: ignore
        by_img.setdefault(img_path, []).append((emb, bbox))

    report_rows = []
    for img_path, faces in tqdm(by_img.items(), desc="Organizing"):
        labels: List[str] = []
        for emb, _ in faces:
            name = assign_face_labels(emb, centroids, th=assign_th)
            if name is not None:
                labels.append(name)
        labels = sorted(set(labels))
        src_path = Path(img_path)
        if not labels:
            # No assigned persons: skip (or redirect to an "unidentified"
            #  folder if desired)
            continue
        elif len(labels) == 1:
            person_dir = dest / _safe_name(labels[0])
            person_dir.mkdir(exist_ok=True)
            out_path = person_dir / src_path.name
        else:
            out_name = f"{'-'.join(_safe_name(x) for x in labels)}-{
                src_path.name}"
            out_path = group_dir / out_name

        _apply_file_op(src_path, out_path, mode)
        report_rows.append({
            "src": str(src_path),
            "dest": str(out_path),
            "labels": ",".join(labels),
        })

    rep_path = dest / "organize_report.csv"
    pd.DataFrame(report_rows).to_csv(rep_path, index=False)
    print(f"Organization finished. Report: {rep_path}")


def _apply_file_op(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        if not dest.exists():
            shutil.copy2(src, dest)
    elif mode == "move":
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))
    elif mode == "symlink":
        if dest.exists():
            return
        try:
            dest.symlink_to(src)
        except OSError:
            # Windows without symlink privileges => fall back to copy
            shutil.copy2(src, dest)
    else:
        raise ValueError("mode must be one of: copy|move|symlink")


# --------------------------- CLI ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Photo classification by people (Phase 1 & Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("scan", help="Phase 1: scan and extract embeddings")
    s1.add_argument("--src", type=Path, required=True,
                    help="Photo root folder")
    s1.add_argument("--work", type=Path, required=True,
                    help="Working directory")
    s1.add_argument("--min-size", type=int, default=60,
                    help="Minimum face size (px)")
    s1.add_argument("-v", "--verbose", action="count", default=1)

    s2 = sub.add_parser(
        "cluster", help="Phase 1: cluster with DBSCAN (cosine metric)"
        )
    s2.add_argument("--work", type=Path, required=True)
    s2.add_argument("--eps", type=float, default=0.35,
                    help="DBSCAN eps (cosine distance)")
    s2.add_argument("--min-samples", type=int, default=2)
    s2.add_argument("-v", "--verbose", action="count", default=1)

    s3 = sub.add_parser("export-faces",
                        help="Phase 1: export face crops per cluster")
    s3.add_argument("--work", type=Path, required=True)
    s3.add_argument("--top-k-per-cluster", type=int, default=40)
    s3.add_argument("-v", "--verbose", action="count", default=1)

    s4 = sub.add_parser("organize",
                        help="Phase 2: organize photos by labeled persons")
    s4.add_argument("--src", type=Path, required=True)
    s4.add_argument("--work", type=Path, required=True)
    s4.add_argument("--dest", type=Path, required=True)
    s4.add_argument("--assign-th", type=float, default=0.35,
                    help="Min cosine similarity for assignment")
    s4.add_argument("--mode", choices=["copy", "move", "symlink"],
                    default="copy")
    s4.add_argument("-v", "--verbose", action="count", default=1)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.cmd == "scan":
        cmd_scan(src=args.src, work=args.work,
                 min_size=args.min_size, verbosity=args.verbose)
    elif args.cmd == "cluster":
        cmd_cluster(work=args.work, eps=args.eps,
                    min_samples=args.min_samples, verbosity=args.verbose)
    elif args.cmd == "export-faces":
        cmd_export_faces(work=args.work,
                         top_k_per_cluster=args.top_k_per_cluster,
                         verbosity=args.verbose)
    elif args.cmd == "organize":
        cmd_organize(src=args.src, work=args.work, dest=args.dest,
                     assign_th=args.assign_th, mode=args.mode,
                     verbosity=args.verbose)
    else:  # pragma: no cover
        raise SystemExit(2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
