# Experiment plan and results

Completed experiments use SegTHOR fold 0 (15 training patients, 5 validation
patients) and deterministic seed 43. Reported results are unweighted macro
averages over foreground classes 1-4. Higher Dice is better; lower HD95 and
ASD are better.

| Stage | ID | Change from current winner | Macro Dice | Macro HD95 (mm) | Macro ASD (mm) | Status and future use |
|---|---|---|---:|---:|---:|---|
| Legacy reference | B0 | Original per-volume min-max and uncorrected merged ground truth | - | - | - | Completed, but four-organ results were not recorded here and are not directly comparable to fixed-GT runs |
| Fixed-GT reference | R0 | Corrected labels, original per-volume min-max, 2D, CE, Adam, LR `5e-4`, no augmentation/sampling/scheduler/post-processing | 0.515 | 29.330 | 6.225 | Completed; scientific fixed-GT reference |
| Preprocessing | P1 | Fixed HU window `[-1000, 300]` | **0.595** | **18.454** | 4.906 | **Selected preprocessing**; used by D1-D3 and all later stages |
| Preprocessing | P2 | CLAHE after clipping to `[-1000, 300]` | 0.591 | 23.018 | **4.823** | Completed; not selected because P1 has better Dice and HD95 |
| Data handling | D1 | P1 plus `--augment` | **0.631** | 25.242 | **4.669** | **Selected data method**; carry P1 plus augmentation into the context stage |
| Data handling | D2 | P1 plus `--drop_empty_slices 0.9` | 0.590 | 21.931 | 5.263 | Completed; rejected because it does not improve P1 overall |
| Data handling | D3 | P1 plus `--oversample_foreground --oversample_foreground_percent 0.5` | 0.541 | 29.290* | 10.489* | Completed; rejected after near-total esophagus failure |
| Data handling | D4 | P1 plus augmentation and `--drop_empty_slices 0.9` | 0.515 | 23.406 | 6.151 | Completed; rejected because combining augmentation and slice dropping substantially harms three of four classes |
| Context | C1 | P1 plus D1 augmentation and `--context_slices 1` | 0.589 | 26.047 | 5.367 | Completed; rejected because three-slice context is worse than D1 on every macro metric |
| Context | C2 | P1 plus D1 augmentation and `--context_slices 2` | **0.640** | **17.419** | **4.315** | **Selected context method**; five-slice 2.5D input is carried into the loss stage |
| Loss reference | L0 | Selected P1 + D1 + C2 pipeline with plain CE | 0.640 | 17.419 | 4.315 | Completed by C2; no duplicate L0 run is needed |
| Loss | L1 | L0 with CE and `--ce_weights invfreq --ce_weights_alpha 0.5` | 0.656 | 26.908 | 5.013 | Completed; modest Dice gain, substantially worse surface distances |
| Loss | L2 | L0 with Dice | 0.501 | 140.135 | 47.757 | Completed; rejected due to severe segmentation and boundary degradation |
| Loss | L3 | L0 with DiceCE and `--dicece_lambda 0.5` | 0.688 | 20.471 | 4.116 | Completed; better Dice than L4, but not selected after L6 |
| Loss | L4 | L0 with Balance, `alpha=0.5`, `t=0.9`, and automatic halfway fallback | 0.683 | 15.631 | 3.788 | Completed; retained as the unnormalized Balance reference |
| Loss follow-up | L5 | Weighted DiceCE | - | - | - | Deferred; weighted CE did not improve the overall trade-off enough to justify combining it with DiceCE yet |
| Loss follow-up | L6 | L4 with `--balance_normalized` | **0.713** | 16.324 | **3.484** | **Selected loss**; best Dice and ASD, with HD95 `0.693 mm` above L4 |
| Learning rate | T1 | L6 pipeline with Adam and LR `1e-4` | 0.668 | 18.255 | 4.270 | Completed; rejected, worse than L6 on all three metrics |
| Learning rate | T2 | L6 pipeline with Adam and LR `1e-3` | 0.701 | **14.563** | 3.629 | Completed; best HD95, but lower Dice and worse ASD than L6 |
| Optimizer | T3 | L6 pipeline at LR `5e-4` with AdamW instead of Adam | 0.706 | 17.531 | 3.566 | Completed; worse than L6 on all three metrics |
| Scheduler | T4 | L6 pipeline with StepLR (`step_size=10`, `gamma=0.1`) | 0.691 | 17.806 | 3.925 | Completed; worse than L6 on all three metrics |
| Scheduler | T5 | L6 pipeline with ReduceLROnPlateau (`patience=5`, `gamma=0.1`) | 0.696 | 14.967 | 3.717 | Completed; improves HD95 over L6, but T2 is better on all three metrics |
| Scheduler | T6 | L6 pipeline with CosineAnnealingLR (`T_max=25`) | 0.700 | 16.521 | 3.751 | Completed; worse than L6 on all three metrics |
| Early stopping | T7 | L6 with `--early_stopping_patience 10` | - | - | - | Next; efficiency check against the same 25-epoch reference |
| Stability | V1 | L6 on the same fold-0 patient split with training seed 44 | - | - | - | Next; independent training-seed repeat, not a new validation fold |
| Stability | V2 | L6 on the same fold-0 patient split with training seed 45 | - | - | - | Next; independent training-seed repeat, not a new validation fold |

\* D3 failed to predict the esophagus for Patient 03. Its class-1 HD95 and ASD
were `NaN`, and the current evaluator omitted those values when computing the
class means. D3's reported distance averages are therefore optimistic.

## Stage decisions

- Preprocessing winner: P1 fixed HU windowing.
- Data-handling winner: D1 augmentation. It has the best macro Dice and ASD,
  but its trachea HD95 regression must remain visible in later comparisons.
- D4 confirms that adding aggressive empty-slice dropping to augmentation is
  not beneficial. D1 remains the selected data-handling method.
- Context winner: C2 with two neighboring slices on each side. Relative to D1,
  it improves macro Dice by `0.009`, HD95 by `7.823 mm`, and ASD by `0.354 mm`.
- Loss-stage reference: P1 HU windowing, D1 augmentation, C2 five-slice 2.5D
  input, Adam, LR `5e-4`, no scheduler, and seed 43. C2 is plain-CE L0.
- Loss-stage choice: L6 normalized Balance improves macro Dice by `0.030` and
  ASD by `0.303 mm` over L4, while HD95 worsens by `0.693 mm`. L4 remains the
  best-HD95 alternative. Pure Dice (L2) is clearly unsuitable in this setup.
- Learning-rate choice: keep `5e-4` from L6. T1 (`1e-4`) is worse on all three
  metrics. T2 (`1e-3`) lowers HD95 by `1.761 mm` relative to L6, but lowers
  Dice by `0.013` and raises ASD by `0.144 mm`. Retain T2 as an HD95-focused
  alternative rather than combining it with other changes now.
- Current reference: P1 HU windowing, D1 augmentation, C2 five-slice 2.5D,
  L6 normalized Balance, Adam, LR `5e-4`, no scheduler, deterministic seed 43.
- Optimizer and scheduler choice: retain Adam and no scheduler. T3, T4, and T6
  are worse than L6 on all three metrics. T5 reduces HD95 by `1.358 mm` but
  worsens Dice and ASD; it is also dominated by T2, so no combination run is
  justified by these results.
- Next: T7 checks whether patience 10 saves epochs without losing the best
  checkpoint. V1 and V2 repeat L6 with training seeds 44 and 45 on the same
  P1 fold-0 split. These three jobs are independent and can run concurrently.
- Selection is provisional: these comparisons use one seed and only five
  validation patients. V1/V2 assess training-seed variability, not split
  variability; repeat across folds before treating small differences as robust.
