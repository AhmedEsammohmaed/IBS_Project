"""Regenerate evaluation_report.pdf from the checked-in metrics.json."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
doc = SimpleDocTemplate(str(ROOT / "evaluation_report.pdf"), pagesize=landscape(letter))
styles = getSampleStyleSheet()
story = [Paragraph("Floor Plan CV Model Evaluation", styles["Title"]), Spacer(1, 8)]
story.append(Paragraph(
    "Upstream-reported results only: RAQEEB-CCTV v1.0.0 test split, "
    "1,682 held-out plans. This package's preprocessing, OCR, tiled inference, "
    "and planner-domain performance were not independently evaluated.", styles["BodyText"]
))
story.append(Spacer(1, 12))
table_data = [["Class", "Instances", "Images", "Precision", "Recall", "F1*", "Mask AP50", "Mask AP50:95"]]
for name, row in metrics["per_class"].items():
    table_data.append([name, row["instances"], row["images"], row["precision"], row["recall"], row["f1"], row["mask_map50"], row["mask_map50_95"]])
table_data.append(["Overall", "48,255", "1,682", "—", "—", "—", metrics["aggregate"]["mask_map50"], metrics["aggregate"]["mask_map50_95"]])
table = Table(table_data, repeatRows=1, hAlign="LEFT")
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCC7D1")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F8")]),
    ("FONTSIZE", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
]))
story.extend([table, Spacer(1, 10), Paragraph(
    "*F1 is the upstream per-class best-threshold F1. Gate has no held-out examples; "
    "escalator has only four. Column recall and wall mask AP are weak. See metrics.json "
    "for per-class counts and source. OCR error rates, confusion matrix, deployment "
    "failure cases, CPU/GPU latency, and memory are not available. The CC BY-NC 4.0 "
    "license restricts commercial use; separate permission is required.", styles["BodyText"]
)])
doc.build(story)
