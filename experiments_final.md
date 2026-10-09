# stage 1

| ID | Parent | Exact change | Why this candidate / selection | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---:|---:|---:|
| P0 | R1 | Existing `--hu_min -1000 --hu_max 300` | Wide-window control; no new run. Retains low-HU airway context. | 0.7765 | 14.884 | 3.407 |
| P1 | R1 | `--hu_min -310 --hu_max 400` | Higher soft-tissue contrast per PNG level for mediastinal organs; check trachea performance after low-HU clipping. Core. | 0.7895 | 14.102 | 2.909 |
| P2 | R1 | `--clahe --hu_min -310 --hu_max 400` | Compare to P1 to isolate adaptive contrast. Existing implementation fixes clip limit 0.01; do not invent a CLI parameter sweep. Core. | 0.7955 | 13.270 | 2.840 |
| P3 | R1 | `--hu_windows -1000 300 -310 400` | Complementary wide/narrow information. Five slices x two windows = 10 channels, supported by U-Net. Wide window first also preserves DINO's established input. Core. | 0.7891 | 18.984 | 3.224 |


Values are the unweighted mean of the four organ means in `results_new/*.csv`: `macro = (esophagus + heart + trachea + aorta) / 4`; background excluded. Each organ mean averages validation patients. 

**P2 wins stage 1** on macro metrics

# stage 2


| ID | Parent | Exact change | Why / stop rule | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| A0 | P1 | Existing rotation/noise `--augment`, scale 0 | Existing control. Rotation is +/-5 degrees, probability 0.5; Gaussian noise probability 0.25. | Reuse | 0.7895 | 14.102 | 2.909 |
| A1 | A0 (P1) | `--augment --augment_scale 0.15` | Modest shared zoom 0.85-1.15, probability 0.5. Check clipping after zoom; a scale sweep is unjustified. | Completed | 0.8086 | 14.476 | 2.731 |
| A2 | A1 (P1) | Preprocessing `--crop_body` | Crop improves all three macro metrics; retain inverse geometry. | Completed; selected stage 2 | 0.8343 | 9.461 | 2.116 |
| A2-P2 | A2 | Add preprocessing `--clahe`; retain crop/window/scale | Test P2 contrast on the best complete recipe. | Completed; retain A2 | 0.8308 | 10.447 | 2.243 |
| A3 | A2 | Set `--augment_scale 0`, keep old augmentation and crop | Check whether scale still helps after cropping. | Completed; retain A2 | 0.8102 | 12.132 | 2.659 |
| G1-resolution | Best available A configuration | Preprocessing `--shape 512 512`, no other change | Run only if esophagus/contour errors suggest information loss at 256 and memory profiling fits batch 8. This adds real image detail, unlike enlarging DINO's already-downsampled image. | Conditional: 1 | Pending | - | - |



# stage 3

| ID | Parent | Change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| C0 | A2 (P1, crop, scale 0.15) | Existing `--context_slices 2` (five slices) | Incoming winner. Physical support is patient-dependent because z spacing is unchanged. | Reuse | 0.8343 | 9.461 | 2.116 |
| C1 | C0 | `--context_slices 0` | Single-slice comparison on cropped A2. | Completed; retain C0 | 0.8288 | 10.638 | 2.388 |
| C2 | C0 | `--context_slices 1` (three slices) | Intermediate context; compare directly with C0 and C1 on A2. | Completed; retain C0 | 0.8273 | 9.686 | 2.279 |



# stage 4

| ID | Parent | Exact change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| L0 | A2 = C0 | Binary Balance, normalization none, alpha 0.5, t 0.9, fallback 12 | Lowest small-model HD95/ASD; carry Balance forward. | Selected loss | 0.8343 | 9.461 | 2.116 |
| L2 | L0 | `--loss_fn dicece --dicece_lambda 0.5`, no CE weights | Higher Dice; worse HD95/ASD than L0. | Completed; trade-off | 0.8554 | 12.266 | 2.194 |
| L3 | L2 | `--ce_weights invfreq --ce_weights_alpha 0.5` | Versus L2: lower Dice, slightly lower HD95, higher ASD; retain Balance. | Completed; not selected | 0.8503 | 11.965 | 2.379 |


# stage 5

| ID | Parent | Change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| M0 | A2 = L0 | Existing `--model unet-small`, Balance loss | Matched control for M1. | Reuse | 0.8343 | 9.461 | 2.116 |
| M1 | M0 | `--model unet-large` | Improves all metrics versus M0; best surface distances, selected compromise versus M2. | Completed; selected | 0.8689 | 7.020 | 1.653 |
| M2 | M1 | `--loss_fn dicece --dicece_lambda 0.5`, no CE weights | Highest Dice; versus M1: +0.0069 Dice, +2.855 mm HD95, +0.438 mm ASD. | Completed; Dice alternative | 0.8758 | 9.875 | 2.092 |


Best config so far: linear HU [-310, 400], body cropping, rotation/noise + scale 0.15, five-slice context, large U-Net, Balance loss; no foundation (M1). M2 remains the Dice-leading alternative.


# stage 6

| ID | Parent | Exact change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| F0 | M1 | No foundation; large U-Net + Balance | Selected stage-5 control; reuse existing metrics. | Reuse | 0.8689 | 7.020 | 1.653 |
| F1 | F0 | `--foundation_model meddinov3-vitb16 --foundation_fusion encoder --foundation_upsample 1` | Worse than M1 on all macro metrics; heart improves, other organs worsen. | Completed; not selected | 0.8644 | 9.864 | 1.808 |
| F2 | F1, if beneficial | Change `--foundation_upsample 2`; Tests a finer feature/fusion config. At 256 input: DINO input 512 and patch grid 32 x 32, fusion level 3 instead of 4 | Conditional on beneficial F1. | Skip; F1 not beneficial | - | - | - |
| F3 | F1 settings; M1 promotion control | Change only `--foundation_fusion decoder`; retain upsample 1 | One fusion-placement test; adopt only if better than M1. | Completed; not selected | 0.8595 | 7.987 | 1.729 |


Stage 6 selects M1 (no foundation model)

# stage 7

| ID | Parent | One controlled change | Trigger | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---:|---:|---:|
| X1 | M1 | Alternative linear-HU representation | Skip: no foundation selected | - | - | - |
| X2 | M1 | Alternative loss | Covered by M2: retain Balance for lower surface errors; no duplicate run. | - | - | - |
| X3 | M1 | `--context_slices 0`; all else fixed | Completed; M1 wins all three macro metrics. | 0.8664 | 7.826 | 1.755 |


# stage 8

M1 remains selected: large U-Net, Balance, five slices, linear HU [-310, 400], crop, scale 0.15, no foundation

| ID | Input | Candidate settings | Status / purpose | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---:|---:|---:|
| Q0 | M1 raw | No filtering | Completed; matches M1 exactly. | 0.8689 | 7.020 | 1.653 |
| Q1 | Q0 | Largest component per class; k=1, connectivity 26, classes 2/3/4 | Completed; helps aorta, harms trachea. | 0.8691 | 7.153 | 1.652 |
| Q2 | Q0 | Anatomy-aware config: relative minimum 0.05; esophagus may retain fragments; classes 2/3/4 capped at 1 | Completed; best macro Dice/ASD, but trachea HD95 worsens. | 0.8693 | 7.100 | 1.636 |
| Q3 | Q0 | Q2 with esophagus relative minimum 0.01 | Deferred; current esophagus filtering improves all metrics. | - | - | - |
| Q4 | Q0 | Fill holes; heart only, connectivity 6 | Completed; tiny heart improvement on all metrics. | 0.8690 | 7.004 | 1.652 |
| Q5 | Q0 | Closing; heart only, 1 iteration, connectivity 6 | Completed; heart HD95/ASD worsen. | 0.8690 | 7.022 | 1.658 |
| Q6 | Q0 | Opening; heart only, 1 iteration, connectivity 6 | Completed; heart HD95/ASD worsen. | 0.8690 | 7.026 | 1.660 |
| Q6b | Q0 | Salt-and-pepper median; heart only, kernel 3 | Completed; heart ASD worsens. | 0.8691 | 7.016 | 1.659 |
| Q7 | M1 probabilities + CT | Dense CRF; configured defaults, 5 iterations, spatial/bilateral weights 3/5 | Completed; harms overall result, helps aorta only. | 0.8396 | 12.810 | 2.319 |
| Q8 | Q7 inputs | Spatial/bilateral weights 1.5/2.5 | Deferred; test saved aorta-only result first. | - | - | - |
| Q9 | Best individual methods | Combine best 2-3; CRF first, then component filtering, then accepted morphology | See Q9a/Q9b; no full-organ CRF combination. | - | - | - |
| Q10 | Q0 | Combined-foreground components; k=4, connectivity 26 | Completed; negligible improvement. | 0.8689 | 7.020 | 1.653 |
| Q11 | Q0 | Remove per-class components smaller than 100 voxels, connectivity 26 | Completed; trachea surface errors worsen. | 0.8691 | 7.531 | 1.675 |
| Q2a | Q0 | Q2 anatomy rules without class 3 | Ready; isolate removal of harmful trachea filtering. | Pending | - | - |
| Q9a | Q0 | Q2a then Q4 heart hole-filling | Ready; combine improvements on different organs. | Pending | - | - |
| Q7a | Q0 + saved Q7 | Transfer Q7 aorta only into raw background/aorta voxels; preserve classes 1/2/3 | Ready; test aorta benefit without other-organ damage. | Pending | - | - |
| Q9b | Q0 + saved Q7 | Q7a, then Q2a, then Q4 | Ready; three-method interaction test; do not assume additive gains. | Pending | - | - |



# stage 9

| ID | Configuration / runs | Purpose and decision |
|---|---|---|
| V1 | Frozen best two complete pipelines; existing training seed 43 plus seeds **44 and 45** on the **same fold-0 manifest**. Up to 4 new training runs at the selected full budget. Apply each pipeline's fixed postprocessing. | Compare mean/sample SD and paired patient effects. Do not choose a favorable seed as the final “method.” A marginal expensive feature that fails repeats is removed. |
| V2 | Same two frozen recipes on **folds 1,2,3**, preprocessing seed 43, retains 10, training seed 43. Up to 6 new runs. Fit CE weights on each fold's training patients only. | With verified 40 patients, this completes four 30/10 folds. Compare paired out-of-fold predictions across all patients; report fold variation separately from training-seed variation. Keep postprocessing fixed. |
| V3 | R0 baseline-method recipe on the same remaining folds (3 runs), if no valid matched runs exist | Supports a robust baseline-improvement claim with the same native-GT metric route. If omitted for budget reasons, explicitly limit the baseline claim to fold 0. |
| E1 | Optional equal-probability average of the accepted configuration's three same-split seed models; no new training | Gated G3 and a later ensemble wrapper (not currently a CLI feature). Evaluate on their common unseen validation patients; apply fixed postprocessing **after averaging**. Keep only if confirmed improvement warrants 3x inference/storage. |
| Z1 | Freeze winning preprocessing, context, model/foundation, loss, training duration, and postprocessing | Record exact resolved configuration. Generate final test predictions only after freezing. Never score an “ensemble validation” prediction using a model that trained on that patient. |
