# Train on a custom dataset

This guide takes you from raw deer face images to a working training run, using
`LocalDatasetLoader`. The default `KaggleDatasetLoader` only knows how to fetch
the published Rotwild-ID Kaggle dataset; for any other dataset you produce the
extracted directory yourself with the helper script in [scripts/](../scripts/).

## 1. What you need

- **Raw images.** One or more JPEG/PNG files showing a single deer face. Multiple
  images per individual are required for metric learning to work.
- **Annotations**, one per image, with at least one of:
  - **3 landmarks**: pixel coordinates for `right_eye`, `left_eye`, `nose`.
  - **Segmentation mask**: a COCO-RLE `counts` string for the face region.

The script supports three face-chip methods, with these prerequisites:

| Method            | Requires        | Output dir                            |
|-------------------|-----------------|---------------------------------------|
| `landmark_affine` | landmarks       | `data/extracted/landmark_affine/`     |
| `landmark_crop`   | landmarks       | `data/extracted/landmark_crop/`       |
| `mask`            | `rle_mask`      | `data/extracted/mask/`                |

If a method's prerequisite is missing, that subdir is simply not produced. Per
row, methods whose prerequisites are missing are skipped.

## 2. Annotation CSV schema

The script expects **one CSV with one row per image**. Columns:

| Column              | Required             | Notes |
|---------------------|----------------------|-------|
| `image_path`        | yes                  | Path relative to `--images-root` (or absolute). |
| `identity`          | yes                  | Individual deer ID. Strings; whatever you use as label. |
| `image_quality`     | recommended          | One of `excellent`, `good`, `ok`, `poor`. Defaults to `good` if absent. |
| `image_id`          | optional             | Used as the chip filename. Defaults to `row_<idx>`. |
| `right_eye_x`, `right_eye_y` | conditional | Pixel coords in the original image. |
| `left_eye_x`, `left_eye_y`   | conditional | Pixel coords in the original image. |
| `nose_x`, `nose_y`           | conditional | Pixel coords in the original image. |
| `rle_mask`          | conditional          | Uncompressed COCO-RLE `counts` string. |

A row must have **either** all 6 landmark columns **or** a non-empty `rle_mask`.

### Example

```csv
image_path,identity,image_quality,right_eye_x,right_eye_y,left_eye_x,left_eye_y,nose_x,nose_y,rle_mask
deer_a/img001.jpg,deer_a,good,412,288,540,290,476,365,
deer_a/img002.jpg,deer_a,excellent,418,295,548,294,482,372,PXc1k0Q3...
deer_b/img001.jpg,deer_b,ok,,,,,,PXc1k0Q3...
```

The first row (landmarks only) yields chips in `landmark_affine/` and
`landmark_crop/`. The second (both) yields chips in all three. The third
(mask only) yields a chip in `mask/`.

## 3. Producing landmarks and masks

This is out of scope for the script — bring your own annotations. A few options:

- **Landmarks**: hand-annotate ([Label Studio](https://github.com/HumanSignal/label-studio/), [CVAT](https://github.com/cvat-ai/cvat/)) or train a lightweight
  keypoint detector. The 3-point convention here is right eye, left eye, nose
  tip in the original-image pixel coordinate frame.
- **Masks**: Models from the [SAM Family](https://sam3ai.com/model/) produce COCO-RLE-friendly masks. Encode each
  binary mask via `pycocotools.mask.encode` and store the resulting `counts`
  string (decoded to UTF-8) in the `rle_mask` column.
- **Image quality**: the four categories (`excellent`, `good`, `ok`, `poor`) are
  there so you can later filter the training set without re-extracting. We used
  `poor` when the face occupied only a small portion of the frame (low-resolution
  chip after cropping), `ok`/`good` for the typical case, and `excellent` for
  sharp, well-lit, frontal shots. The labels are subjective — pick a convention
  and apply it consistently. If you don't care about filtering, set every row to
  the same value (e.g. `good`) and move on.

### Our experience producing annotations

- **One annotation per image, single deer in frame.** Pre-crop each source
  image so that only the target individual's face is visible. Multi-deer
  frames make both annotation and downstream data handling significantly
  harder, and this script assumes one annotation per row.
- **Masks**: we generated face masks with [SAM3](https://github.com/facebookresearch/sam3) using the text prompt `"face"`,
  then refined them in Label Studio. Pure SAM output was usually close but not
  good enough to skip the review step.
- **Landmarks**: we hand-annotated all keypoints. In a small PoC we got
  reasonable results by prompting SAM3 with `"eye"` / `"nose"` and taking the
  centroid of each returned mask as the keypoint — viable as a bootstrap if you
  have a lot of images and limited annotation budget, but verify on a sample
  before trusting it end-to-end.

## 4. Run the extractor

All dependencies are part of the main project environment:

```bash
uv sync --extra cu128  # or --extra cpu
source .venv/bin/activate
python scripts/create_face_chips.py \
    --annotations path/to/annotations.csv \
    --images-root path/to/images/ \
    --output-dir ./data/extracted \
    --dataset-name MyDeerDataset
```

Useful flags:

- `--methods landmark_affine landmark_crop` — restrict to specific methods
  (default `auto` picks based on which columns are present).
- `--face-chip-size 224` — output chip resolution (default `224`).
- `--landmark-crop-padding 0.3` — bbox padding ratio for `landmark_crop`.

The script writes, per produced method:

```
data/extracted/<method>/
├── image_metadata.csv
├── dataset_metadata.json
└── <image_id>.png
```

## 5. Train

Point the trainer at the local directory and pick a method that exists in your
output:

```bash
deer-face-embed \
    dataset_loader.KIND=LocalDatasetLoader \
    dataset_loader.dataset_location=./data/extracted \
    dataset_loader.face_chip_method=mask
```

Or set the same values in [config/job/training.yaml](../config/job/training.yaml):

```yaml
dataset_loader:
  KIND: LocalDatasetLoader
  dataset_location: ./data/extracted
  face_chip_method: mask
```

`LocalDatasetLoader` validates that every `path` in `image_metadata.csv` exists
on disk before training starts.

## 6. Troubleshooting

- **`Image file missing at: ...`** — `image_metadata.csv` references a chip
  filename that's not on disk. Re-run the extractor; do not edit the CSV by hand.
- **`No usable annotations found`** — your CSV has neither landmark columns nor
  a non-empty `rle_mask` column. Check column names exactly.
- **`Invalid image_quality values: [...]`** — only `excellent`, `good`, `ok`,
  `poor` are accepted. Anything else aborts the run.
- **`failed to decode rle_mask`** — the `counts` string is malformed or doesn't
  match the image's `(height, width)`. The script falls back to producing only
  the landmark methods for that row.
- **`Landmark bbox crop is empty`** — landmarks fall outside the image (or
  collapse to a single point). The row is skipped for `landmark_crop`.
