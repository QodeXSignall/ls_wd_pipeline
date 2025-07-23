"""
Build YOLO‑v5/v8 **image‑classification** dataset from Label Studio JSON export.

Key points
===========
* Designed for **choice‑based** tasks (each image assigned a single class via `choices`).
* Folder layout expected by Ultralytics classification:

    DATASET_ROOT/
        ├── train/
        │   ├── class_A/  *.jpg
        │   └── class_B/  *.jpg
        ├── val/   ...
        └── test/  ...

* Generates **labels.txt** (one class per line) – Ultralytics ≥ v8 can infer classes from folder names, but labels.txt is handy for quick reference.
* Provides quick audit: **`analyze_dataset_cls()`** or CLI flag `--analyze`.
* Robust path resolution: avoids the common *“/mnt/webdav_frames/webdav_frames/…”* duplication when `--mount` already points to the parent folder.

Usage examples
--------------
```bash
# Build dataset (images live under /mnt/webdav_frames/*)
python build_dataset_cls.py \
       --json response.json \
       --dataset ./cls_dataset \
       --mount /mnt/webdav_frames   #  or /mnt if you want /mnt/webdav_frames/...  👈

# Analyse existing dataset
python build_dataset_cls.py --dataset ./cls_dataset --analyze

# Save disk space by linking instead of copying
python build_dataset_cls.py ... --symlink
```
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence
from ls_wb_pipeline import settings

from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 🔧 helpers
# ---------------------------------------------------------------------------

def _get_choice_label(task: Dict[str, Any]) -> str | None:
    """Return the first value from `choices` in any annotation.result."""
    for ann in task.get("annotations", []):
        for res in ann.get("result", []):
            if res.get("type") == "choices":
                choices = res["value"].get("choices")
                if choices:
                    return choices[0]
    return None


def _decode_image_path(url: str) -> Path:
    """Label Studio local‑files URL → *relative* path w.r.t. mounted storage."""
    # e.g. "/data/local-files/?d=webdav_frames/some.jpg" → "webdav_frames/some.jpg"
    if "?d=" in url:
        url = url.split("?d=")[-1]
    return Path(url.lstrip("/"))


def _resolve_src(rel_path: Path, mounted_root: Path | None) -> Path:
    """Join mounted_root + rel_path but avoid duplicate folder names.

    Typical pattern:
        mounted_root = "/mnt/webdav_frames"
        rel_path     = "webdav_frames/img.jpg"
        Desired      = "/mnt/webdav_frames/img.jpg"
    """
    if not mounted_root:
        return rel_path

    # If the first part of rel_path equals mounted_root.name, drop it.
    rel_parts = rel_path.parts
    if rel_parts and rel_parts[0] == mounted_root.name:
        rel_path = Path(*rel_parts[1:])
    return mounted_root / rel_path

# ---------------------------------------------------------------------------
# 🚀 build routine
# ---------------------------------------------------------------------------

def build_dataset_cls_from_tasks(
    tasks: Sequence[dict],
    dataset_root: Path | str = settings.DATASET_PATH,
    mounted_root: Path | str = settings.MOUNTED_PATH,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    symlink: bool = False,
) -> None:
    """Convert tasks → folder dataset suitable for Ultralytics classification."""

    dataset_root = Path(dataset_root)
    mounted_root = Path(mounted_root) if mounted_root else None

    # 1️⃣ Gather (src_path, class_name)
    entries: list[dict] = []
    for t in tasks:
        cls = _get_choice_label(t)
        if not cls:
            continue
        rel_img_path = _decode_image_path(t["data"]["image"])
        src = _resolve_src(rel_img_path, mounted_root)
        entries.append({"class": cls, "src": src, "name": src.name})

    if not entries:
        raise RuntimeError("Нет валидных choice‑аннотаций в tasks.")

    classes = sorted({e["class"] for e in entries})

    # 2️⃣ Train/val/test split  (stratify only if every class has ≥2 imgs)
    y = [e["class"] for e in entries]
    min_cls_count = min(Counter(y).values())
    strat = y if min_cls_count >= 2 else None

    if len(entries) < 3:
        split = {"train": entries, "val": [], "test": []}
    else:
        tv, test = train_test_split(entries, test_size=test_ratio, random_state=42, stratify=strat)
        y_tv = [e["class"] for e in tv]
        strat_tv = y_tv if min(Counter(y_tv).values()) >= 2 else None
        val_ratio_adj = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(tv, test_size=val_ratio_adj, random_state=42, stratify=strat_tv)
        split = {"train": train, "val": val, "test": test}

    # 3️⃣ Create directories
    for sp in ("train", "val", "test"):
        for cls in classes:
            (dataset_root / sp / cls).mkdir(parents=True, exist_ok=True)

    # 4️⃣ Copy / link images
    copied, skipped = 0, 0
    for sp, items in split.items():
        for it in items:
            dst = dataset_root / sp / it["class"] / it["name"]
            if dst.exists():
                continue
            if not it["src"].exists():
                skipped += 1
                continue
            if symlink:
                os.symlink(os.path.abspath(it["src"]), dst)
            else:
                shutil.copy2(it["src"], dst)
            copied += 1

    # 5️⃣ labels.txt
    with open(dataset_root / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    # 6️⃣ Console summary
    print("\n✅  Dataset created at:", dataset_root)
    stats = analyze_dataset_cls(dataset_root)
    if skipped:
        print(f"⚠️  {skipped} images referenced in JSON not found at resolved path.\n    → Проверьте правильность --mount: {mounted_root or '<not set>'}\n      Пример проблемного пути: {entries[0]['src'] if entries else '<n/a>'}")

    return stats

# ---------------------------------------------------------------------------
# 📊 analyze routine
# ---------------------------------------------------------------------------

def analyze_dataset_cls(dataset_root: Path | str = settings.DATASET_PATH) -> dict[str, dict[str, int]]:
    """Return and print number of images per class per split."""
    dataset_root = Path(dataset_root)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                count = len([p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
                stats[cls_dir.name][split] = count

    # total
    for cls, d in stats.items():
        d["total"] = d["train"] + d["val"] + d["test"]

    # pretty print
    if stats:
        print("\n📊  Images per class:")
        hdr = f"{'Class':25} | train | val | test | total"
        print(hdr)
        print("-" * len(hdr))
        for cls, d in stats.items():
            print(f"{cls:25} | {d['train']:5} | {d['val']:3} | {d['test']:4} | {d['total']:5}")
    else:
        print("No images found in dataset.")

    return stats

# ---------------------------------------------------------------------------
# 📜 CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    p = argparse.ArgumentParser("Build or analyze YOLO classification dataset from Label Studio export")

    p.add_argument("--dataset", required=True, help="Dataset root (for build or analyze)")
    p.add_argument("--json", default="", help="Label Studio JSON export (provide to build)")
    p.add_argument("--mount", default="", help="Filesystem root where images reside (e.g. /mnt or /mnt/webdav_frames)")
    p.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying images")
    p.add_argument("--analyze", action="store_true", help="Only analyze existing dataset; ignore --json")
    args = p.parse_args()

    if args.analyze:
        analyze_dataset_cls(Path(args.dataset))
        return

    if not args.json:
        raise SystemExit("--json is required when not using --analyze")

    with open(args.json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    build_dataset_cls_from_tasks(
        tasks,
        dataset_root=Path(args.dataset),
        mounted_root=Path(args.mount) if args.mount else None,
        symlink=args.symlink,
    )


if __name__ == "__main__":
    _cli()
