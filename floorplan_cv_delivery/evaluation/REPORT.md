# Evaluation report status

The selected model card reports a held-out test split of 1,682 plans with per-class instance counts, precision, recall, F1, mask mAP50, and mask mAP50:95. The values are mirrored in `metrics.json` and are **upstream-reported results**. They do not measure this wrapper, PDF pipeline, tiled inference, OCR, or the planner's own input distribution.

The model card reports its test split grouped by near-duplicate source drawings and a partial-label scoring protocol (only score categories fully annotated by each source). This package does not redistribute that split. Obtain the source data under its original license to reproduce it.

Escalator has only four test instances and gate has none, so both remain marked unvalidated. Column mask mAP50 is 0.300 with recall 0.247; wall mask mAP50:95 is 0.230. These weak areas require review and likely target-domain validation.

No held-out planner-specific set was supplied. No OCR CER/WER, pixel IoU/Dice, confusion matrix, project-specific hard examples, false-positive/false-negative images, or target-hardware CPU/GPU latency and memory have been measured. The current environment also lacked free disk space to install PyTorch, so the inference wrapper itself has not been executed here. `evaluation_report.pdf` summarizes these results and limitations.
