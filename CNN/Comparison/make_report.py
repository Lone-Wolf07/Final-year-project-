import csv
import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

COMP_DIR = Path(__file__).resolve().parent
RES = COMP_DIR / "Results"
REPORT = COMP_DIR.parents[1] / "run_reports" / "Feature_Extractor_x_Classifier_Report.docx"
EXTRACTORS = ("ConvNeXt-ECA", "ResNet50")
HEAD, QUANTUM, LINEAR = "Classical head (end-to-end)", "Quantum fidelity", "Linear probe"


def read(name):
    with open(RES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(doc, header, rows, bold_first=True):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, (cell, text) in enumerate(zip(cells, r)):
            cell.text = str(text)
            if bold_first and i == 0:
                cell.paragraphs[0].runs[0].bold = True
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9.5)
    doc.add_paragraph()
    return t


def main():
    cells = {(r["extractor"], r["classifier"]): r for r in read("cells.csv")}
    bench = {r["extractor"]: r for r in read("epoch_benchmark.csv")}
    notes = json.loads((COMP_DIR / "report_notes.json").read_text(encoding="utf-8"))

    def c(e, k, key):
        return cells[(e, k)][key]

    def ms(e, k):
        r = cells[(e, k)]
        return f"{r['accuracy_mean']} ± {r['accuracy_std']}" if r.get("accuracy_mean") else "single model"

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading("Feature Extractor × Classifier Comparison", 0)
    p = doc.add_paragraph(f"IQ-OTH/NCCD lung CT, 80:20 split (211 test images) · {date.today():%d %B %Y}")
    p.runs[0].font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    doc.add_heading("Purpose", 1)
    doc.add_paragraph(notes["purpose"])

    doc.add_heading("Setup", 1)
    for line in notes["setup"]:
        doc.add_paragraph(line, style="List Bullet")

    doc.add_heading("Result: accuracy (2 × 2)", 1)
    doc.add_paragraph("Test accuracy (%) of the single trained model (seed 0). Brackets: benign recall (%), "
                      "the hardest class with 20 test images.")
    table(doc, ["Feature extractor", "Classical head", "Quantum fidelity classifier"],
          [[e, f"{c(e, HEAD, 'accuracy')}  ({c(e, HEAD, 'recall_benign')})",
            f"{c(e, QUANTUM, 'accuracy')}  ({c(e, QUANTUM, 'recall_benign')})"] for e in EXTRACTORS])

    doc.add_heading("Full test-set scores", 2)
    keys = ["accuracy", "precision", "recall", "f1", "mcc", "auc", "recall_benign", "recall_malignant", "recall_normal"]
    rows = [[f"{e} + {k.split(' (')[0]}"] + [c(e, k, key) for key in keys] + [ms(e, k)]
            for e in EXTRACTORS for k in (HEAD, QUANTUM)]
    table(doc, ["Model", "Acc", "Prec", "Rec", "F1", "MCC", "AUC", "Benign", "Malig.", "Normal", "Acc over 5 seeds"],
          rows)

    doc.add_heading("Result: time (2 × 2)", 1)
    doc.add_paragraph("Model time per test image when the 211 test images are classified in batches "
                      "(feature extraction + classifier, preprocessing excluded), in milliseconds.")
    table(doc, ["Feature extractor", "Classical head", "Quantum fidelity classifier"],
          [[e, c(e, HEAD, "batch_model_ms"), c(e, QUANTUM, "batch_model_ms")] for e in EXTRACTORS])

    doc.add_heading("Time breakdown", 2)
    rows = []
    for e in EXTRACTORS:
        for k in (HEAD, QUANTUM):
            r = cells[(e, k)]
            rows.append([f"{e} + {k.split(' (')[0]}", r["batch_preprocess_ms"], r["batch_extract_ms"],
                         r["batch_classify_ms"], r["batch_total_ms"], r["single_extract_ms"],
                         r["single_classify_ms"], r["single_total_ms"]])
    table(doc, ["Model", "Batch: preprocess", "Batch: extractor", "Batch: classifier", "Batch: total",
                "1 image: extractor", "1 image: classifier", "1 image: total"], rows)
    doc.add_paragraph("All values in ms per image. '1 image' is the mean of 20 test images classified one at a time "
                      "from the raw file, including preprocessing.")

    doc.add_heading("Training time", 2)
    rows = []
    for e in EXTRACTORS:
        b = bench[e]
        q = float(c(e, QUANTUM, "classifier_train_s"))
        est = float(b["estimate_50_epochs_s"])
        rows.append([e, b["seconds_per_epoch"], f"{est / 60:.1f}", f"{float(b['measured_50_epochs_s']) / 60:.1f}",
                     "0 (trained with the network)", f"{q / 60:.1f}"])
    table(doc, ["Feature extractor", "s / epoch (benchmark)", "50 epochs, est. (min)", "50 epochs, as run (min)",
                "Classical head (min)", "Quantum classifier, 50 epochs (min)"], rows)
    doc.add_paragraph(notes["training_note"])

    doc.add_heading("Extra check: linear classifier on the same frozen features", 1)
    doc.add_paragraph(notes["linear_intro"])
    table(doc, ["Feature extractor", "Classical head", "Quantum fidelity", "Linear probe"],
          [[e] + [f"{ms(e, k) if k != HEAD else c(e, k, 'accuracy')}" for k in (HEAD, QUANTUM, LINEAR)]
           for e in EXTRACTORS])
    doc.add_paragraph("Accuracy (%); quantum and linear values are mean ± std over 5 seeds.")

    doc.add_heading("Confusion matrices", 1)
    doc.add_picture(str(RES / "confusion_matrices.png"), width=Inches(5.6))

    doc.add_heading("Findings", 1)
    for line in notes["findings"]:
        doc.add_paragraph(line, style="List Bullet")

    doc.add_heading("Limitations", 1)
    for line in notes["limitations"]:
        doc.add_paragraph(line, style="List Bullet")

    doc.add_heading("Files", 1)
    for line in notes["files"]:
        doc.add_paragraph(line, style="List Bullet")

    REPORT.parent.mkdir(exist_ok=True)
    doc.save(REPORT)
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
