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
| Loss | L3 | L0 with DiceCE and `--dicece_lambda 0.5` | **0.688** | 20.471 | 4.116 | Completed; best macro Dice, retained as a strong alternative |
| Loss | L4 | L0 with Balance, `alpha=0.5`, `t=0.9`, and automatic halfway fallback | 0.683 | **15.631** | **3.788** | **Selected loss** for the next stage; best HD95 and ASD with near-best Dice |
| Loss follow-up | L5 | Weighted DiceCE | - | - | - | Deferred; weighted CE did not improve the overall trade-off enough to justify combining it with DiceCE yet |
| Loss follow-up | L6 | L4 with `--balance_normalized` | - | - | - | Next; isolated normalization check at LR `5e-4` |
| Learning rate | T1 | L4 pipeline with Adam and LR `1e-4` | - | - | - | Ready if L4 remains selected after L6; lower-LR comparison against L4 at `5e-4` |
| Learning rate | T2 | L4 pipeline with Adam and LR `1e-3` | - | - | - | Ready if L4 remains selected after L6; higher-LR comparison against L4 at `5e-4` |
| Optimizer | T3 | AdamW at the selected LR | - | - | - | Planned; compares AdamW plus its default weight decay |
| Scheduler | T4 | Step scheduler | - | - | - | Planned; fixed scheduled decay |
| Scheduler | T5 | Plateau scheduler | - | - | - | Planned; validation-responsive decay |
| Scheduler | T6 | Cosine scheduler | - | - | - | Planned; smooth decay |
| Early stopping | T7 | Best setup with patience around 10 | - | - | - | Planned as an efficiency experiment |

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
- Loss-stage choice: L4 Balance is the best multi-metric trade-off. L3 DiceCE
  has `0.004` higher macro Dice, but L4 lowers macro HD95 by `4.839 mm` and
  ASD by `0.328 mm`. Pure Dice (L2) is clearly unsuitable in this setup.
- Next comparisons: run L6 first; it changes only Balance normalization. If
  L4 remains selected, run T1 and T2, changing only LR relative to L4 (`5e-4`).
  If L6 wins, update the T1/T2 jobs to use normalized Balance before submitting
  them. Choose the LR before running T3 optimizer and T4-T6 scheduler
  comparisons. T7 early stopping comes after the training setup is selected.
