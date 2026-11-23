# Evaluation status

This delivery has no project-specific evaluation results. A held-out dataset, annotations, and evaluation protocol were not supplied, so precision, recall, F1, mAP, IoU, Dice, OCR error rates, latency, memory, confusion matrix, and failure-case artifacts are deliberately marked unavailable in `metrics.json`.

The upstream model card reports wall mask AP 0.263 at IoU 0.50:0.95, AP 0.620 at IoU 0.50, and AP 0.179 at IoU 0.75 on its own validation split. These numbers are reference context only and must not be represented as results on Private 5G Deployment Planner data.

To complete acceptance evaluation, supply held-out floor plans with per-instance wall annotations and (for expanded classes) corresponding labels, then report class-wise metrics and operational measurements on the deployment hardware.
