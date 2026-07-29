"""
Shared dataset layout and anomalib datamodule construction.

Both the live servers (detector_server.py) and the offline evaluation
scripts (experiment.py) read the same capture directory that app.py's Live
Capture tab writes to, and must build the datamodule the same way - if they
diverge, offline results stop predicting live behaviour, which would defeat
the point of a comparison platform.
"""

import os

from anomalib.data import Folder

from train_service import count_images

# Disabling the validation split needs the MODE, not just the ratio - setting
# val_split_ratio=0.0 while the mode stays FROM_TEST still halves the test set.
# That was silently costing us half of an already-tiny evaluation set.
try:
    from anomalib.data.utils import ValSplitMode
    VAL_SPLIT_NONE = ValSplitMode.NONE
except ImportError:  # pragma: no cover - enum location varies across versions
    VAL_SPLIT_NONE = "none"

DATA_ROOT = "inspection_data/images"
NORMAL_DIR = "normal"
DEFECT_PARENT = "defect"

# Fixed split seed. Without this, Folder reshuffles the train/test split on
# every call, so two evaluations of the same model on the same images score
# a different subset and can disagree completely - we measured PatchCore and
# EfficientAD swapping best/worst places between consecutive runs on
# identical data. Comparisons have to be reproducible to mean anything.
SPLIT_SEED = 42


def normal_path():
    return os.path.join(DATA_ROOT, NORMAL_DIR)


def defect_subdirs():
    """All defect subtype dirs that actually contain images, as paths relative
    to DATA_ROOT (the form anomalib's Folder wants)."""
    parent = os.path.join(DATA_ROOT, DEFECT_PARENT)
    if not os.path.isdir(parent):
        return []
    return [
        os.path.join(DEFECT_PARENT, sub)
        for sub in sorted(os.listdir(parent))
        if count_images(os.path.join(parent, sub)) > 0
    ]


def first_defect_dir():
    """anomalib's Folder takes a single abnormal_dir. Training needs only
    normal images, but Folder builds a test split too, so hand it a defect
    folder when one exists. None if there aren't any."""
    dirs = defect_subdirs()
    return dirs[0] if dirs else None


def build_datamodule(name, abnormal_dir=None, **extra):
    """abnormal_dir defaults to the first non-empty defect subtype."""
    kwargs = {
        "name": name,
        "root": DATA_ROOT,
        "normal_dir": NORMAL_DIR,
        # Turn the validation split OFF by mode, not by ratio. With an
        # abnormal_dir present the mode defaults to FROM_TEST, which takes
        # half the test images regardless of val_split_ratio - that is how
        # 30 normal images produced only 3 scored ones (30 * 0.2 = 6, halved).
        "val_split_mode": VAL_SPLIT_NONE,
        "val_split_ratio": 0.0,
        # Reproducibility - see SPLIT_SEED above.
        "seed": SPLIT_SEED,
        # With abnormal_dir set, anomalib is in FROM_DIR mode, where
        # normal_split_ratio (NOT test_split_ratio) decides how many normal
        # images are held out for evaluation. 0.35 of 30 gives ~10 scored.
        "normal_split_ratio": 0.35,
    }
    chosen = abnormal_dir or first_defect_dir()
    if chosen:
        kwargs["abnormal_dir"] = chosen
    kwargs.update(extra)
    datamodule = Folder(**kwargs)
    datamodule.setup()
    return datamodule
