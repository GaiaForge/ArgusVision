# Findings

What this platform has actually measured, what was decided and why, and what is
still open. Kept separate from `README.md`, which describes how to *use* the
tool rather than what it has *learned*.

Entries are dated. Numbers here are only meaningful with their conditions
attached — that is the whole reason the tool records setup metadata.

---

## Purpose

ArgusVision is a **bench platform for pre-validating manufacturing vision
approaches**. It is not deployed as an inspection system. It exists to answer,
before a production system is specified: *which detection approach suits this
process and this defect class?*

Its deliverable is a recommendation other people act on, which makes
**reproducibility and provenance part of the product**, not polish. A tool that
produced a different winner on each run — as this one briefly did — would be
worse than no tool, because the numbers look authoritative either way.

---

## The core principle

Every result so far points the same way: **match the approach to how precisely
the check can be specified.**

| If the check is... | Use | Why |
|---|---|---|
| Fully specifiable at a fixed position ("channel 3 holds the blue wire") | Classical CV | Fastest, fully explainable, names *which* region failed. No model, no GPU, no training data. |
| A known property you have good examples of, but no rule for | PatchCore / EfficientAD | Learns normal from your own parts. Needs captured images and stable conditions. |
| Genuinely open-ended — defects you cannot enumerate in advance | LiZAD (zero-shot) | The only one that can flag a defect nobody defined. Needs no data at all. |
| A known object class worth labelling | YOLO | Gives class *and* location. Costs a bounding-box dataset and retraining per change. Untested here. |

The corollary matters as much: **if you can state the rule exactly, don't reach
for anomaly detection.** It will be slower, less explainable, and worse at it.

---

## Measured results

### 2026-07-28 — missing MOSFETs, handheld, no fixture

Conditions: 7 unpopulated MOSFET footprints with bare solder pads. Camera on a
desk, boards propped up, both moved between shots. 30 normal / 7 defect images.

| Detector | Gap | Verdict |
|---|---|---|
| LiZAD | +0.023 | marginal |
| PatchCore | −0.056 | overlap |
| EfficientAD | −0.172 | overlap |

**Nothing separated.** Three architecturally different detectors agreeing is a
statement about the dataset, not about the detectors.

**The strongest single piece of evidence:** PatchCore scored a clean **+0.521**
gap on an earlier **18**-image set, and got *worse* at 30 images. More data made
it worse. That only happens when the added images contributed variation rather
than information — which is what hand-positioning produces.

Reference-based methods (PatchCore, EfficientAD) are structurally the most
damaged by pose variation: they learn what normal looks like, so a normal part
at an unfamiliar angle either reads as anomalous, or forces the model to widen
"normal" until it stops noticing real defects. LiZAD, having no reference to
violate, was unaffected — its +0.023 was identical across every run.

**Conclusion: the fixture is the blocking item, not the model choice.** Roughly
two hours went into tuning split parameters before that was clear. Do not repeat
that.

### Earlier, same defect, 18 images

LiZAD ~0.25 vs ~0.0 — a positive but fragile margin that flipped verdicts under
normal camera noise. PatchCore 0.000 vs 0.521 — clean. This is the result that
prompted adding PatchCore, and later EfficientAD.

---

## Decisions and why

**Four approaches, not one.** The architecture is a fossil record of failures,
not foresight. LiZAD came first; it underperformed on missing components, so
PatchCore was added; PatchCore's success pointed at logical anomalies as the
real category, which is why EfficientAD went in. Channel Check exists because
some checks are fully specifiable and shouldn't go near a model at all.

**PatchCore and EfficientAD share their implementation** (`detector_server.py`,
`detector_client.py`, `detector_data.py`, `experiment.py`, `train_service.py`).
Two reasons: one live path to debug instead of two, and a guarantee the
detectors are compared through identical plumbing rather than subtly different
code. A fairness property, not just DRY.

**Training is a UI button.** A comparison platform that requires SSH and a conda
activate to retrain is not a platform — it is a script with a web page attached.

**Scores are never compared across detectors.** Each normalises differently.
Only the *gap* within a single run is meaningful. The UI and the report both say
so, because it is the most likely way these numbers get misread.

**Clearing history archives rather than deletes.** The table is meant to be an
auditable record and the button is one press away during a demo.

---

## Things that cost real time

**Non-reproducible evaluation.** anomalib's `Folder` reshuffles its train/test
split on every call. Two evaluations of the same model on the same images had
PatchCore and EfficientAD swapping best and worst place. LiZAD — deterministic
and zero-shot — was identical to three decimals across both, and that is what
identified the cause. Fixed with a `seed`. **A boring, deterministic baseline is
worth keeping around purely as an instrument.**

**anomalib's split is a trap.** `val_split_ratio=0.0` does nothing; the *mode*
must be `ValSplitMode.NONE` or `FROM_TEST` still takes half the test set. With
`abnormal_dir` set it is in `FROM_DIR` mode, where `normal_split_ratio` — not
`test_split_ratio` — governs the normal count. 30 normal images were yielding 3
scored ones.

**torch 2.6 changed `torch.load`'s `weights_only` default to True**, which
rejects the enums inside Lightning checkpoints. Forcing it False is required —
`setdefault` silently loses, because Lightning passes the argument explicitly.

**Environment and camera-network gotchas** are in `README.md` under *Environment
gotchas*; they are reproduction steps rather than findings.

---

## Open questions

1. **LiZAD has never been tested on the defects it is designed for.** Every test
   so far has been absence-type — its known structural weakness. Scratches,
   discoloration, contamination are untested. This is the cheapest high-value
   experiment available and needs only a physically damaged sample.
2. **EfficientAD is effectively unmeasured.** It has only ever run on handheld
   data, and it is the most data-hungry of the three. `MIN_TRAIN_IMAGES=20` and
   `MAX_EPOCHS=20` are starting points, not tuned values.
3. **YOLO is untested.** It is the only approach that gives class *and* location
   ("MOSFET 3 of 7 missing"). Gated on whether that is worth building a
   bounding-box labelling workflow for.
4. **Target hardware should probably be an output.** "This defect needs
   Orin-class compute" and "this one runs on a €200 board" are both things a
   project team needs before committing. Currently the report says which
   detector won, not what hardware that implies.
5. **Evaluation splits remain thin.** ~20% of normal images are scored. Capturing
   more directly widens it, since it is a ratio.
