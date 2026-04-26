"""Create face chips and the LocalDatasetLoader directory layout from an annotations CSV.

Reads a single CSV (one row per image/annotation) and writes one or more of:
- ``<output>/mask/``            (requires ``rle_mask``)
- ``<output>/landmark_affine/`` (requires landmarks)
- ``<output>/landmark_crop/``   (requires landmarks)

Each output directory contains the chip images plus ``image_metadata.csv`` and
``dataset_metadata.json`` consumed by ``LocalDatasetLoader``.

The chip extraction algorithms are byte-compatible with the historical
extraction job that produced the published Kaggle dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from pycocotools import mask as mask_utils
from tqdm import tqdm

LANDMARK_COLS = (
    "right_eye_x",
    "right_eye_y",
    "left_eye_x",
    "left_eye_y",
    "nose_x",
    "nose_y",
)
VALID_QUALITY = ("excellent", "good", "ok", "poor")

AFFINE_RIGHT_EYE_REL = (0.1405, 0.3590)
AFFINE_LEFT_EYE_REL = (1 - 0.1405, 0.3590)
AFFINE_NOSE_REL = (0.50, 0.736539)


def _decode_rle(counts: str, height: int, width: int) -> np.ndarray:
    rle = {"counts": counts, "size": [int(height), int(width)]}
    return mask_utils.decode(rle)  # type: ignore[arg-type]


def _mask_polygon(binary_mask: np.ndarray) -> list[float]:
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=lambda c: float(cv2.contourArea(c)))
    return contour.reshape(-1, 2).astype(float).flatten().tolist()


def _mask_bbox_xywh(binary_mask: np.ndarray) -> list[float]:
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [0.0, 0.0, 0.0, 0.0]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [
        float(xmin),
        float(ymin),
        float(xmax - xmin + 1),
        float(ymax - ymin + 1),
    ]


def _make_chip_affine(
    image: np.ndarray,
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
    target_size: int,
    binary_mask: np.ndarray | None,
) -> tuple[np.ndarray, list[float], list[float] | None]:
    origin_kps = np.array([right_eye, left_eye, nose], dtype=np.float32)
    chip_kps = np.array(
        [
            [target_size * AFFINE_RIGHT_EYE_REL[0], target_size * AFFINE_RIGHT_EYE_REL[1]],
            [target_size * AFFINE_LEFT_EYE_REL[0], target_size * AFFINE_LEFT_EYE_REL[1]],
            [target_size * AFFINE_NOSE_REL[0], target_size * AFFINE_NOSE_REL[1]],
        ],
        dtype=np.float32,
    )
    transform = cv2.getAffineTransform(origin_kps, chip_kps)
    chip = cv2.warpAffine(image, transform, (target_size, target_size))
    keypoints = chip_kps.flatten().astype(float).tolist()

    segmentation: list[float] | None = None
    if binary_mask is not None:
        chip_mask = cv2.warpAffine(
            binary_mask, transform, (target_size, target_size), flags=cv2.INTER_NEAREST
        )
        segmentation = _mask_polygon(chip_mask)
    return chip, keypoints, segmentation


def _make_chip_landmark_crop(
    image: np.ndarray,
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
    target_size: int,
    padding_ratio: float,
    binary_mask: np.ndarray | None,
) -> tuple[np.ndarray, list[float], list[float], list[float] | None]:
    img_h, img_w = image.shape[:2]
    xs = [right_eye[0], left_eye[0], nose[0]]
    ys = [right_eye[1], left_eye[1], nose[1]]
    pad = padding_ratio * max(max(xs) - min(xs), max(ys) - min(ys))

    xmin = int(max(0, min(xs) - pad))
    ymin = int(max(0, min(ys) - pad))
    xmax = int(min(img_w, max(xs) + pad))
    ymax = int(min(img_h, max(ys) + pad))

    crop_w = xmax - xmin
    crop_h = ymax - ymin
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("Landmark bbox crop is empty")

    square_size = max(crop_w, crop_h)
    square_image = np.zeros((square_size, square_size, 3), dtype=image.dtype)
    y_off = (square_size - crop_h) // 2
    x_off = (square_size - crop_w) // 2
    square_image[y_off : y_off + crop_h, x_off : x_off + crop_w] = image[
        ymin:ymax, xmin:xmax
    ]
    chip = cv2.resize(
        square_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )

    scale = target_size / square_size

    def _proj(pt: tuple[float, float]) -> tuple[float, float]:
        return (
            float((pt[0] - xmin + x_off) * scale),
            float((pt[1] - ymin + y_off) * scale),
        )

    re_c, le_c, no_c = _proj(right_eye), _proj(left_eye), _proj(nose)
    keypoints = [re_c[0], re_c[1], le_c[0], le_c[1], no_c[0], no_c[1]]
    bbox = [
        float(x_off * scale),
        float(y_off * scale),
        float(crop_w * scale),
        float(crop_h * scale),
    ]

    segmentation: list[float] | None = None
    if binary_mask is not None:
        square_mask = np.zeros((square_size, square_size), dtype=np.uint8)
        square_mask[y_off : y_off + crop_h, x_off : x_off + crop_w] = binary_mask[
            ymin:ymax, xmin:xmax
        ]
        chip_mask = cv2.resize(
            square_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
        )
        segmentation = _mask_polygon(chip_mask)

    return chip, bbox, keypoints, segmentation


def _make_chip_mask(
    image: np.ndarray,
    binary_mask: np.ndarray,
    target_size: int,
    landmarks: list[tuple[float, float]] | None,
) -> tuple[np.ndarray, list[float], list[float], list[float] | None]:
    """Tight-bbox crop around mask, mask out background, pad to square, resize."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        raise ValueError("Mask is empty - no face pixels found")
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    bbox_w = xmax - xmin + 1
    bbox_h = ymax - ymin + 1

    image_crop = image[ymin : ymax + 1, xmin : xmax + 1].copy()
    mask_crop = binary_mask[ymin : ymax + 1, xmin : xmax + 1]
    mask_3ch = np.stack([mask_crop] * 3, axis=-1)
    image_crop[mask_3ch == 0] = 0

    square_size = max(bbox_w, bbox_h)
    square_image = np.zeros((square_size, square_size, 3), dtype=image.dtype)
    square_mask = np.zeros((square_size, square_size), dtype=np.uint8)
    y_off = (square_size - bbox_h) // 2
    x_off = (square_size - bbox_w) // 2
    square_image[y_off : y_off + bbox_h, x_off : x_off + bbox_w] = image_crop
    square_mask[y_off : y_off + bbox_h, x_off : x_off + bbox_w] = mask_crop

    chip = cv2.resize(
        square_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    chip_mask = cv2.resize(
        square_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )
    bbox = _mask_bbox_xywh(chip_mask)
    segmentation = _mask_polygon(chip_mask)

    keypoints: list[float] | None = None
    if landmarks is not None:
        scale = target_size / square_size
        keypoints = []
        for x, y in landmarks:
            keypoints.extend(
                [
                    float((x - xmin + x_off) * scale),
                    float((y - ymin + y_off) * scale),
                ]
            )
    return chip, bbox, segmentation, keypoints


def _row_has_landmarks(row: pd.Series) -> bool:
    return all(c in row.index and pd.notna(row[c]) for c in LANDMARK_COLS)


def _row_has_mask(row: pd.Series) -> bool:
    return (
        "rle_mask" in row.index
        and pd.notna(row["rle_mask"])
        and str(row["rle_mask"]).strip() != ""
    )


def _serialize(value: list[float] | None) -> str | None:
    return json.dumps(value) if value is not None else None


def _detect_methods(df: pd.DataFrame, requested: list[str]) -> list[str]:
    has_any_landmarks = all(c in df.columns for c in LANDMARK_COLS) and df[
        list(LANDMARK_COLS)
    ].notna().all(axis=1).any()
    has_any_mask = "rle_mask" in df.columns and df["rle_mask"].notna().any()

    if requested == ["auto"]:
        methods: list[str] = []
        if has_any_landmarks:
            methods.extend(["landmark_affine", "landmark_crop"])
        if has_any_mask:
            methods.append("mask")
        if not methods:
            raise SystemExit(
                "No usable annotations found: need either all landmark columns "
                f"({', '.join(LANDMARK_COLS)}) or an 'rle_mask' column."
            )
        return methods

    for m in requested:
        if m == "mask" and not has_any_mask:
            raise SystemExit("Method 'mask' requested but no 'rle_mask' column present.")
        if m in ("landmark_affine", "landmark_crop") and not has_any_landmarks:
            raise SystemExit(
                f"Method '{m}' requested but landmark columns are missing/empty."
            )
    return requested


def _validate_quality(df: pd.DataFrame) -> pd.DataFrame:
    if "image_quality" not in df.columns:
        logger.warning("No 'image_quality' column found; defaulting all rows to 'good'.")
        df = df.copy()
        df["image_quality"] = "good"
    bad = ~df["image_quality"].isin(VALID_QUALITY)
    if bad.any():
        offenders = df.loc[bad, "image_quality"].unique().tolist()
        raise SystemExit(
            f"Invalid image_quality values: {offenders}. "
            f"Allowed: {list(VALID_QUALITY)}."
        )
    return df


def _process(
    df: pd.DataFrame,
    methods: list[str],
    images_root: Path,
    output_dir: Path,
    target_size: int,
    padding_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    method_dirs = {m: output_dir / m for m in methods}
    for d in method_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    rows_by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}

    skipped = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="extracting"):
        image_rel = str(row["image_path"])
        image_path = images_root / image_rel if not Path(image_rel).is_absolute() else Path(image_rel)
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"row {idx}: failed to read {image_path}; skipping")
            skipped += 1
            continue

        has_lm = _row_has_landmarks(row)
        has_mask = _row_has_mask(row)
        if not (has_lm or has_mask):
            logger.warning(f"row {idx}: no landmarks and no rle_mask; skipping")
            skipped += 1
            continue

        landmarks: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ] | None = None
        if has_lm:
            landmarks = (
                (float(row["right_eye_x"]), float(row["right_eye_y"])),
                (float(row["left_eye_x"]), float(row["left_eye_y"])),
                (float(row["nose_x"]), float(row["nose_y"])),
            )

        binary_mask: np.ndarray | None = None
        if has_mask:
            try:
                binary_mask = _decode_rle(
                    str(row["rle_mask"]), img.shape[0], img.shape[1]
                )
            except Exception as e:
                logger.warning(f"row {idx}: failed to decode rle_mask ({e}); skipping mask")
                binary_mask = None

        identity = str(row["identity"])
        image_quality = str(row["image_quality"])
        image_id = str(row["image_id"]) if "image_id" in row.index and pd.notna(row.get("image_id")) else f"row_{idx:08d}"

        for method in methods:
            try:
                if method == "mask":
                    if binary_mask is None:
                        continue
                    chip, bbox, segmentation, keypoints = _make_chip_mask(
                        img,
                        binary_mask,
                        target_size,
                        list(landmarks) if landmarks is not None else None,
                    )
                elif method == "landmark_affine":
                    if landmarks is None:
                        continue
                    right_eye, left_eye, nose = landmarks
                    chip, keypoints, segmentation = _make_chip_affine(
                        img, right_eye, left_eye, nose, target_size, binary_mask
                    )
                    bbox = [0.0, 0.0, float(target_size), float(target_size)]
                elif method == "landmark_crop":
                    if landmarks is None:
                        continue
                    right_eye, left_eye, nose = landmarks
                    chip, bbox, keypoints, segmentation = _make_chip_landmark_crop(
                        img,
                        right_eye,
                        left_eye,
                        nose,
                        target_size,
                        padding_ratio,
                        binary_mask,
                    )
                else:
                    raise ValueError(f"Unknown method {method}")
            except ValueError as e:
                logger.warning(f"row {idx} method={method}: {e}; skipping")
                continue

            target_dir = method_dirs[method]
            file_name = f"{image_id}.png"
            target_path = target_dir / file_name
            i = 0
            while target_path.exists():
                i += 1
                file_name = f"{image_id}_{i:02}.png"
                target_path = target_dir / file_name
            cv2.imwrite(str(target_path), chip)

            rows_by_method[method].append(
                {
                    "image_id": image_id,
                    "path": file_name,
                    "identity": identity,
                    "image_quality": image_quality,
                    "bbox": _serialize(bbox),
                    "keypoints": _serialize(keypoints),
                    "segmentation": _serialize(segmentation),
                }
            )

    if skipped:
        logger.info(f"skipped {skipped} rows")
    return rows_by_method


def _write_outputs(
    rows_by_method: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    dataset_name: str,
) -> None:
    created = datetime.now().astimezone().isoformat()
    for method, rows in rows_by_method.items():
        method_dir = output_dir / method
        df_out = pd.DataFrame(rows)
        df_out.to_csv(method_dir / "image_metadata.csv", index=False)

        meta = {
            "name": dataset_name,
            "created": created,
            "animals_simple": "red deer",
            "face_chip_method": method,
            "reported_n_total": len(df_out),
            "reported_n_individuals": int(df_out["identity"].nunique()) if len(df_out) else 0,
        }
        (method_dir / "dataset_metadata.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"{method}: wrote {len(df_out)} chips to {method_dir}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a LocalDatasetLoader-compatible directory from an annotations CSV."
    )
    p.add_argument("--annotations", type=Path, required=True, help="Path to annotations CSV.")
    p.add_argument("--images-root", type=Path, required=True, help="Base directory for image_path values.")
    p.add_argument("--output-dir", type=Path, default=Path("data/extracted"))
    p.add_argument("--face-chip-size", type=int, default=224)
    p.add_argument("--landmark-crop-padding", type=float, default=0.3)
    p.add_argument("--dataset-name", type=str, default="CustomDeerFaces")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["auto"],
        choices=["auto", "mask", "landmark_affine", "landmark_crop"],
        help="Which face-chip methods to produce. 'auto' picks based on available columns.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not args.annotations.exists():
        raise SystemExit(f"Annotations CSV not found: {args.annotations}")
    if not args.images_root.exists():
        raise SystemExit(f"Images root not found: {args.images_root}")

    df = pd.read_csv(args.annotations)
    for required in ("image_path", "identity"):
        if required not in df.columns:
            raise SystemExit(f"Annotations CSV missing required column: {required}")
    df = _validate_quality(df)

    methods = _detect_methods(df, args.methods)
    logger.info(f"producing methods: {methods}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_method = _process(
        df,
        methods,
        args.images_root,
        args.output_dir,
        args.face_chip_size,
        args.landmark_crop_padding,
    )
    _write_outputs(rows_by_method, args.output_dir, args.dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
