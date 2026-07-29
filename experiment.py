"""
Shared offline evaluation runner for the trainable anomalib detectors.

Fits a detector on captured normal images, scores the held-out test split,
and reports the gap between the worst normal score and the best defect
score. That gap is the number that actually matters: a positive gap means
the detector separates this defect class, and its size is how much margin a
threshold would have.

Both patchcore_experiment.py and efficientad_experiment.py call this so
their output is byte-for-byte comparable in format - comparing detectors is
the entire purpose of this tool, and that only works if the reports line up.

NOTE: the anomalib Folder/Engine calls (via detector_data.py) and the
prediction-result attribute names below are a best-effort reading of
anomalib 2.x's API. Validated for PatchCore; expect possible debugging for
other models.
"""

import os

from anomalib.engine import Engine

from detector_data import build_datamodule, defect_subdirs, normal_path
from train_service import count_images


def run_experiment(*, label, model_cls, min_normal, datamodule_kwargs=None, engine_kwargs=None):
    """Returns True if the run produced a usable result, False if it bailed."""
    datamodule_kwargs = datamodule_kwargs or {}
    engine_kwargs = engine_kwargs or {}

    normal_count = count_images(normal_path())
    defect_dirs = defect_subdirs()
    defect_count = sum(count_images(os.path.join("inspection_data/images", d)) for d in defect_dirs)

    if normal_count < min_normal:
        print(f"Only {normal_count} normal images captured - {label} needs at least {min_normal}.")
        print("Capture more via the Live Capture tab first.")
        print(f"\nRESULT: INCONCLUSIVE ({label}) - not enough normal images to train on.")
        return False
    if defect_count == 0:
        print("No defect images found - capture at least one to evaluate against.")
        print(f"\nRESULT: INCONCLUSIVE ({label}) - nothing to compare normal images against.")
        return False

    print(
        f"Training {label} on {normal_count} normal images, evaluating against "
        f"{defect_count} defect images from {', '.join(defect_dirs)}..."
    )

    datamodule = build_datamodule(f"{label.lower()}_experiment", **datamodule_kwargs)
    model = model_cls()
    engine = Engine(**engine_kwargs)
    engine.fit(datamodule=datamodule, model=model)

    results = engine.predict(datamodule=datamodule, model=model)

    normal_scores, defect_scores = [], []
    for batch in results or []:
        for gt_label, score in zip(batch.gt_label, batch.pred_score):
            (defect_scores if gt_label else normal_scores).append(float(score))

    if not normal_scores or not defect_scores:
        print("No scored predictions came back - check the anomalib output above for errors.")
        print(f"\nRESULT: INCONCLUSIVE ({label}) - evaluation produced no scores.")
        return False

    max_normal = max(normal_scores)
    min_defect = min(defect_scores)
    gap = min_defect - max_normal

    print(f"\n=== {label} ===")
    print(f"Normal scores: max={max_normal:.3f} (n={len(normal_scores)})")
    print(f"Defect scores: min={min_defect:.3f} (n={len(defect_scores)})")
    print(f"Gap: {gap:+.3f}")

    if gap > 0:
        print(f"\nRESULT: CLEAN SEPARATION ({label}) - gap of {gap:.3f}.")
        print(f"Set the threshold between {max_normal:.3f} and {min_defect:.3f}.")
    else:
        print(f"\nRESULT: OVERLAP ({label}) - scores do not separate this defect class.")
        # Distinguishing "wrong tool" from "not enough data" matters: the first
        # means try another approach, the second means go capture more images.
        # Only the second is fixable by doing more of what you were doing.
        if normal_scores and len(normal_scores) < 10:
            print(
                f"Only {len(normal_scores)} normal images reached evaluation, so this may mean "
                "'not enough data to tell yet' rather than 'wrong approach'. Capture more "
                "and re-run before concluding."
            )
    return True
