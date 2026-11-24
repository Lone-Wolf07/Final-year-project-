import csv
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
RES = HERE / "Results"
REPORT = HERE.parents[1] / "run_reports" / "Pipeline_Design_Rationale.docx"


def read(name):
    with open(RES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, d=2):
    return f"{float(x):.{d}f}"


doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)


def table(header, rows, size=9.5, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for r in rows:
        for cell, text in zip(t.add_row().cells, r):
            cell.text = str(text)
    for row in t.rows:
        for k, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[k])
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
    doc.add_paragraph()


def label(text, body):
    p = doc.add_paragraph()
    p.add_run(text + " ").bold = True
    p.add_run(body)
    return p


def bullets(lines):
    for line in lines:
        doc.add_paragraph(line, style="List Bullet")


def decision(title, choice, source, alternatives, why, evidence, gap, pillar):
    doc.add_heading(title, 2)
    table(["Our choice", "Where it comes from", "Alternatives considered"], [[choice, source, alternatives]],
          widths=[2.3, 1.5, 2.7])
    label("Why.", "")
    bullets(why)
    label("Evidence from this project.", "")
    bullets(evidence)
    if gap:
        label("Gap.", gap)
    p = label("Pillar sentence for the thesis:", "")
    r = p.add_run(pillar)
    r.italic = True


den = {r["method"]: r for r in read("denoising.csv")}
enh = {r["method"]: r for r in read("enhancement.csv")}
stats = read("dataset_stats.csv")
sig = sorted(float(s["sigma"]) for s in stats)
std = sorted(float(s["std"]) for s in stats)
shapes = {}
for s in stats:
    shapes[(s["height"], s["width"])] = shapes.get((s["height"], s["width"]), 0) + 1
q = lambda v, p: v[int(p * (len(v) - 1))]
NLM = "NLM, adaptive h (ours)"
ACL = "ACLAHE, adaptive clip (ours)"
best_psnr = max(den, key=lambda k: float(den[k]["psnr_db"]))
best_ssim = max(den, key=lambda k: float(den[k]["ssim"]))
rank_psnr = sorted(den, key=lambda k: -float(den[k]["psnr_db"])).index(NLM) + 1

# =====================================================================  FRONT
doc.add_heading("Pipeline Design Rationale", 0)
p = doc.add_paragraph(f"Why each stage of the lung CT pipeline is built the way it is · {date.today():%d %B %Y}")
p.runs[0].font.color.rgb = RGBColor(0x59, 0x59, 0x59)

doc.add_heading("How to use this report", 1)
doc.add_paragraph(
    "Each decision in the pipeline is covered in order, from the raw CT slice to the final metric. For every "
    "decision the report gives: what we chose, whether it was inherited from the base paper or is our own change, "
    "the alternatives, the reasons (with references to cite), the evidence we have measured ourselves, any gap "
    "in that evidence, and a ready-to-adapt sentence for the thesis. The sentences are starting points; rewrite "
    "them in your own words.")
doc.add_paragraph(
    "Be honest about provenance. Many choices (NLM, ACLAHE, ConvNeXt, ECA, AdaBelief, the fidelity classifier) "
    "come from the base paper, because the project's first aim was to reproduce it. For those, the thesis argument "
    "is: \"we kept the paper's choice, here is why it is technically sound, and here is what we measured\". For "
    "our own changes (de-duplication, augmenting after the split, two-stage training, tanh bounding, data "
    "re-uploading, the learnable temperature) the argument is stronger, because each fixes a specific, "
    "demonstrable problem.")

doc.add_heading("New measurements made for this report", 1)
doc.add_paragraph(
    f"The preprocessing choices had no measured support before this report, so CNN/Rationale/"
    f"preprocessing_evidence.py was written to compare them against the alternatives on the dataset itself. "
    f"It uses a fixed random sample of 40 slices per class (120 slices) and every one of the "
    f"{len(stats)} de-duplicated slices for the dataset statistics. Results are in CNN/Rationale/Results/.")
bullets([
    "Denoising: synthetic Gaussian noise (σ = 10 grey levels) is added to each slice, each denoiser is applied, "
    "and the output is compared with the original slice (PSNR, SSIM, correlation of Sobel edge maps). Run time is "
    "measured per 512 × 512 slice on the CPU. Caveat: the originals are themselves slightly noisy, so they are a "
    "reference, not a clean ground truth.",
    "Contrast enhancement: each method is applied to the NLM-denoised slice. Measured: entropy, RMS contrast "
    "inside the body, EME (a block-wise contrast measure), shift in mean brightness, noise amplification "
    "(estimated noise after ÷ before), and the spread of contrast across slices.",
    "Dataset statistics: the noise level, contrast and size of every slice, by class, and the noise left after "
    "each preprocessing step for the noisiest slices compared with typical ones.",
    "These are image-quality measurements. They support the choices but do not prove that the classifier "
    "would be less accurate with another method; that needs the ablation listed in the last section.",
])

# =====================================================================  0. DATA
doc.add_heading("Stage 0 — Data", 1)
decision(
    "0.1 Dataset: IQ-OTH/NCCD",
    "IQ-OTH/NCCD chest CT slices, three classes (benign, malignant, normal).",
    "Base paper.",
    "LIDC-IDRI, NLST, LUNA16 (nodule-level, need HU volumes and annotations).",
    ["It is the dataset the base paper reports on, so it is the only way to compare against its numbers.",
     "It is public, slice-level and labelled by oncologists and radiologists at two Iraqi hospitals "
     "[Al-Yasriy et al., 2020], so the task (three-class slice classification) is defined without extra "
     "annotation work.",
     "It is small enough to train repeatedly on a 4 GB laptop GPU, which made five-seed repeats possible."],
    [f"{len(stats)} slices after de-duplication; slice sizes: " +
     ", ".join(f"{h}×{w} ({n})" for (h, w), n in sorted(shapes.items(), key=lambda x: -x[1])[:4]) + "."],
    "It is a single-centre, slice-level set with only 102 unique benign slices. Results may not transfer to "
    "other scanners; say so in Limitations.",
    "IQ-OTH/NCCD was used because it is the benchmark of the reproduced study and is publicly available with "
    "expert slice-level labels.")

decision(
    "0.2 Removing duplicate slices",
    "Exact and near-duplicate images removed before splitting (1,097 → 1,054; 43 removed).",
    "Our change.",
    "Use the dataset as distributed (the base paper's approach).",
    ["A duplicate that lands in both the training and the test set lets the model be tested on an image it has "
     "already seen. This is data leakage and inflates test accuracy [Kaufman et al., 2012; Roberts et al., "
     "2021].",
     "Removing duplicates before splitting is the only way to guarantee that every test image is new."],
    ["43 duplicate slices were found (18 benign, 14 malignant, 11 normal).",
     "The faithful replica, which keeps duplicates and augments before splitting, scored 98.97% versus "
     "97.73% when augmenting after the split: part of the higher figure comes from leakage."],
    None,
    "Duplicate slices were removed before any split so that no test image, or copy of it, was seen in training.")

decision(
    "0.3 Split first, then augment; stratified 80:20 and 70:30",
    "Stratified split of unique slices, then augmentation of the training set only. Both 80:20 and 70:30.",
    "Ratios from the base paper; ordering is our change.",
    "Augment the whole dataset and then split (base paper); k-fold cross-validation.",
    ["If augmentation is applied before splitting, a rotated copy of a test image can be in the training set. "
     "The two are almost identical, so the test accuracy no longer measures generalisation.",
     "Stratification keeps the class proportions equal in train and test, which matters because benign is "
     "only about 10% of the data.",
     "Using both of the paper's ratios allows a direct comparison with each of its tables."],
    ["Replica: 98.97% (augment before) vs 97.73% (augment after) on the same 80:20 design.",
     "Our 80:20 test set is 20 benign, 110 malignant and 81 normal slices; 70:30 is 31 / 164 / 122."],
    None,
    "The data were split before augmentation, with class-stratified sampling, so that the test set contains "
    "only unseen, un-augmented slices.")

decision(
    "0.4 Handling class imbalance: benign oversampled ×4 by augmentation",
    "Each training benign slice plus three augmented copies (82 → 328).",
    "Our setting (the paper augments benign but does not report the factor reproducibly).",
    "Class-weighted loss, focal loss [Lin et al., 2017], SMOTE, plain random oversampling.",
    ["Benign is the minority class (102 of 1,054). Without correction the network can ignore it and still score "
     "high accuracy.",
     "Augmented copies add new views rather than exact repeats, so the model does not simply memorise the 82 "
     "training benign slices [Shorten and Khoshgoftaar, 2019].",
     "SMOTE interpolates between images in pixel space and produces unrealistic CT slices, so it was not used."],
    ["Benign recall is the hardest number in every experiment: 90% for our final model, 60% for ResNet50, and "
     "66 ± 8% for the quantum classifier after only 5 epochs."],
    "No experiment compares oversampling with a class-weighted loss. A short run with weighted cross-entropy "
    "instead of copies would close this gap.",
    "The minority benign class was balanced by adding augmented copies to the training set only, rather than "
    "by synthetic interpolation.")

# =====================================================================  1. PREPROCESSING
doc.add_heading("Stage 1 — Pre-processing", 1)
doc.add_paragraph("The order matters: denoise first, then enhance contrast. Contrast enhancement amplifies "
                  "whatever is in the image, including noise, so the noise has to be removed first.")

sc = {(r["label"], r["group"].startswith("noisy")): {k: f(v, 1) if k in ("raw", "after_nlm", "after_aclahe") else v
                                                     for k, v in r.items()} for r in read("noise_shortcut.csv")}
odd = sum(n for (h_, w_), n in shapes.items() if h_ != w_)
decision(
    "1.0 Finding: acquisition differences that track the class",
    "Not a design choice: a property of the data that several preprocessing choices have to deal with.",
    "Found while measuring for this report.",
    "—",
    [f"Most slices are almost noise-free (median estimated σ̂ {q(sig, .5):.2f} grey levels), but 83 slices are "
     f"heavily noisy (σ̂ about 25–30). All 83 are normal (75) or benign (8); none is malignant.",
     f"{odd} slices are not 512 × 512 (512 × 623, 512 × 801 and two others). All but one are malignant.",
     "A CNN can learn such acquisition differences instead of anatomy (\"shortcut learning\" [Geirhos et al., "
     "2020]; the same problem was found in many COVID-19 CT models [Roberts et al., 2021]). A model could then "
     "call a slice normal because it is grainy, or malignant because it is wide."],
    [f"Noisy normal slices: σ̂ {sc[('normal', True)]['raw']} raw → {sc[('normal', True)]['after_nlm']} after NLM "
     f"→ {sc[('normal', True)]['after_aclahe']} after ACLAHE. Typical normal slices: "
     f"{sc[('normal', False)]['raw']} → {sc[('normal', False)]['after_nlm']} → "
     f"{sc[('normal', False)]['after_aclahe']}. Typical malignant: {sc[('malignant', False)]['raw']} → "
     f"{sc[('malignant', False)]['after_nlm']} → {sc[('malignant', False)]['after_aclahe']}.",
     "So the adaptive NLM shrinks the noise gap from about 15× to about 3×, but does not remove it completely.",
     "Padding the wide slices to a square adds black bars above and below them, so the shape difference survives "
     "preprocessing as a visual cue."],
    "We have not tested whether the model uses these cues. A simple check: accuracy on the noisy normal slices "
    "versus the clean ones, and on the wide malignant slices versus the square ones; or retrain with the wide "
    "slices cropped to a centred square. Mention the finding in Limitations either way; examiners value it.",
    "The dataset contains acquisition differences correlated with class (noise level and image size); the "
    "adaptive denoising reduces the noise difference by about a factor of five, and the remaining risk of "
    "shortcut learning is acknowledged as a limitation.")

doc.add_heading("Measured: denoisers on IQ-OTH/NCCD", 2)
table(["Method", "PSNR (dB)", "SSIM", "Edge corr.", "Noise left on raw slice (σ̂)", "ms / slice"],
      [[k, f(r["psnr_db"]), f(r["ssim"], 4), f(r["edge_corr"], 4), f(r["residual_sigma"]), f(r["ms_per_image"], 1)]
       for k, r in den.items()])
doc.add_paragraph("PSNR, SSIM and edge correlation are measured after adding σ = 10 noise; higher is better. "
                  "'Noise left' is the estimated noise after denoising the original slice. Mean over 120 slices.")
doc.add_picture(str(RES / "denoisers.png"), width=Inches(6.3))
doc.add_paragraph("Crop of one slice: original, noisy input and each denoiser's output.")

decision(
    "1.1 Denoising: non-local means (NLM)",
    "OpenCV fast NLM, template 7 × 7, search window 21 × 21.",
    "Method from the base paper; parameters ours.",
    "Gaussian blur, median filter, bilateral filter [Tomasi and Manduchi, 1998], total variation [Rudin et al., "
    "1992], wavelet shrinkage [Chang et al., 2000], BM3D [Dabov et al., 2007], deep-learning denoisers.",
    ["Noise in reconstructed CT is close to Gaussian and spatially spread out [Diwakar and Kumar, 2018]. Linear "
     "filters (Gaussian) remove it but also blur edges equally, which smooths exactly the small nodule borders "
     "the classifier needs.",
     "NLM averages each pixel with pixels whose surrounding patch looks similar, wherever they are in the image "
     "[Buades et al., 2005]. CT anatomy is highly repetitive (lung parenchyma, vessels, ribs), so there are many "
     "similar patches to average, and edges are kept because patches across an edge do not match.",
     "Median filtering suits impulse (salt-and-pepper) noise, which CT does not have, and it erodes thin "
     "structures such as vessels. Bilateral filtering compares single pixel values rather than patches, so it "
     "is less robust at the same noise level.",
     "BM3D is generally the strongest classical denoiser but is much slower; deep-learning denoisers need pairs "
     "of noisy and clean scans (e.g. low-dose/full-dose), which IQ-OTH/NCCD does not provide.",
     "Template 7 and search 21 are the values recommended by the OpenCV implementation and by Buades et al. as "
     "the speed/quality balance."],
    [f"NLM is in the top group but is not the winner. PSNR: NLM {f(den[NLM]['psnr_db'])} dB, tied with total "
     f"variation ({f(den['Total variation']['psnr_db'])}), just above bilateral "
     f"({f(den['Bilateral (d=7)']['psnr_db'])}), clearly above Gaussian ({f(den['Gaussian (5x5, s=1)']['psnr_db'])}) "
     f"and median ({f(den['Median (5x5)']['psnr_db'])}), and below BM3D ({f(den['BM3D']['psnr_db'])}).",
     f"Edge correlation: NLM {f(den[NLM]['edge_corr'], 4)} beats Gaussian "
     f"({f(den['Gaussian (5x5, s=1)']['edge_corr'], 4)}) and median ({f(den['Median (5x5)']['edge_corr'], 4)}), "
     "which actually lose edges compared with doing nothing, confirming why linear and median filters are "
     "unsuitable. Bilateral, TV and BM3D keep edges slightly better than NLM.",
     f"SSIM: NLM {f(den[NLM]['ssim'], 4)} is lower than bilateral, TV and BM3D.",
     f"Time per 512 × 512 slice: bilateral {f(den['Bilateral (d=7)']['ms_per_image'], 1)} ms, TV "
     f"{f(den['Total variation']['ms_per_image'], 0)} ms, NLM {f(den[NLM]['ms_per_image'], 0)} ms, BM3D "
     f"{f(den['BM3D']['ms_per_image'], 0)} ms.",
     "BM3D is the best denoiser but about 15× slower than NLM, which is already most of the 193 ms "
     "pre-processing time per image. That rules BM3D out for a pipeline whose selling point is speed.",
     "The strongest argument for NLM is not this test but Section 1.2: with a noise-adaptive strength it "
     "removes the heavy noise of the 83 grainy slices while leaving the clean ones almost untouched.",
     "The synthetic test adds σ = 10 noise to every slice, whereas the real data are mostly clean with a few "
     "very noisy slices, so it is a general comparison rather than a model of this dataset."],
    "Bilateral filtering is almost as good as NLM here and 25× faster. If a retraining run with bilateral "
    "filtering keeps the accuracy, it would cut the pre-processing time from about 193 ms to under 20 ms per "
    "image. This is the most useful ablation to run, and either result strengthens the thesis.",
    "Non-local means, the denoiser of the reproduced study, was retained because it removes noise while "
    "preserving edges far better than linear or median filtering and, unlike BM3D, at a practical speed; our "
    "measurements place it in the leading group of classical denoisers on IQ-OTH/NCCD.")

decision(
    "1.2 Adaptive NLM strength h = max(1.15 σ̂, 3)",
    "Filter strength set per slice from the estimated noise level σ̂ [Donoho and Johnstone, 1994 estimator, "
    "as implemented in scikit-image].",
    "Our change (the paper gives no value of h).",
    "One fixed h for all slices.",
    ["The noise level varies enormously between slices (Figure below, panel a). A fixed h small enough for the "
     "clean slices leaves the grainy ones noisy; one large enough for the grainy slices blurs the clean ones.",
     "Buades et al. relate the best h to the noise standard deviation; scaling h with σ̂ applies that rule "
     "automatically. The floor of 3 gives the clean slices a light, uniform clean-up."],
    [f"Estimated noise σ̂ across the {len(sig)} slices: median {q(sig, .5):.2f}, 95th percentile "
     f"{q(sig, .95):.1f} grey levels. About 91% of slices get the floor h = 3; the 83 grainy slices get h ≈ 28–35.",
     f"Effect on the grainy normal slices: σ̂ {sc[('normal', True)]['raw']} → {sc[('normal', True)]['after_nlm']}; "
     "this is what reduces the class-linked noise difference of Section 1.0."],
    None,
    "Because a small group of slices is far noisier than the rest, the NLM filter strength was set per slice in "
    "proportion to the estimated noise, cleaning the grainy slices strongly without blurring the clean ones.")
doc.add_picture(str(RES / "dataset_variation.png"), width=Inches(6.0))
doc.add_paragraph("Variation of (a) noise level and (b) global contrast across all slices, by class.")

doc.add_heading("Measured: contrast enhancement on NLM-denoised slices", 2)
table(["Method", "Entropy (bits)", "RMS contrast (body)", "EME", "Brightness shift", "Noise gain (×)",
       "Contrast spread across slices", "ms"],
      [[k, f(r["entropy_bits"], 3), f(r["rms_contrast_body"]), f(r["eme"]), f(r["brightness_shift"]),
        f(r["noise_gain"]), f(r["std_across_images"]), f(r["ms_per_image"], 2)] for k, r in enh.items()],
      size=8.5)
doc.add_paragraph("Entropy, RMS contrast and EME: higher means more visible detail. Brightness shift: change in "
                  "mean grey level (lower keeps the original appearance). Noise gain: how much the method "
                  "amplifies remaining noise (lower is better). Contrast spread: standard deviation of the "
                  "per-slice contrast (lower means slices look more alike to the network).")
doc.add_picture(str(RES / "enhancers.png"), width=Inches(6.4))

h, c2, c4, a = enh["Global HE"], enh["CLAHE, fixed clip 2.0"], enh["CLAHE, fixed clip 4.0"], enh[ACL]
decision(
    "1.3 Contrast enhancement: CLAHE",
    "Contrast-limited adaptive histogram equalisation, 8 × 8 tiles.",
    "Base paper (as ACLAHE).",
    "No enhancement, global histogram equalisation (HE), gamma correction, unsharp masking, plain adaptive HE.",
    ["CT slices stored as 8-bit images have most pixels in a narrow grey range; lung texture and small nodules "
     "are low-contrast. Enhancement spreads that range so the network sees the detail.",
     "Global HE uses one mapping for the whole image. The large dark background and air dominate the histogram, "
     "so HE over-stretches it, shifts the overall brightness and amplifies noise [Pizer et al., 1987].",
     "Adaptive HE computes a mapping per tile, so each region is enhanced for its own content; the contrast "
     "limit clips the histogram before equalising, which caps how much noise in flat regions can be amplified "
     "[Zuiderveld, 1994]. Bilinear blending between tiles avoids block edges.",
     "CLAHE is the standard enhancement step in medical imaging pipelines and is cheap (about a millisecond)."],
    [f"Global HE shifts mean brightness by {f(h['brightness_shift'])} grey levels, amplifies noise "
     f"×{f(h['noise_gain'])} and lowers entropy ({f(enh['None']['entropy_bits'], 2)} → "
     f"{f(h['entropy_bits'], 2)} bits: grey levels are merged and detail is lost). Its high EME "
     f"({f(h['eme'])}) mostly reflects that amplified noise.",
     f"ACLAHE shifts brightness by only {f(a['brightness_shift'])}, raises entropy to {f(a['entropy_bits'], 2)} "
     f"bits and local contrast (EME) from {f(enh['None']['eme'])} to {f(a['eme'])}.",
     f"Gamma correction barely changes noise (×{f(enh['Gamma (0.8)']['noise_gain'])}) but also lowers local "
     f"contrast (EME {f(enh['Gamma (0.8)']['eme'])}), so it does not reveal detail.",
     f"Every CLAHE variant amplifies remaining noise 5–9×. This is the measured reason why denoising must come "
     f"before enhancement."],
    None,
    "CLAHE was used because it raises local contrast in lung tissue while limiting the noise amplification and "
    "brightness shift caused by global histogram equalisation.")

decision(
    "1.4 Adaptive clip limit (ACLAHE): clip = clip(2.0 × 60 / σ, 1, 4)",
    "Clip limit chosen per slice from its intensity standard deviation σ: low-contrast slices get a higher "
    "limit (more enhancement), high-contrast slices a lower one.",
    "Idea from the base paper (\"adaptive\" CLAHE); formula ours.",
    "One fixed clip limit (2.0 is the usual default; 4.0 is aggressive).",
    ["The rule is designed for data whose contrast varies: flat slices get a higher limit, contrasty ones a "
     "lower limit, so that no slice is over-enhanced. The range 1–4 keeps the limit within normal values.",
     "It is deterministic with no learned parameters, so it cannot leak test information.",
     "On this dataset the slices are already fairly contrasty and similar, so the rule always picks a "
     "conservative limit below the usual default of 2.0. Its practical effect is a gentler CLAHE that amplifies "
     "less noise."],
    [f"Intensity σ of the raw slices is narrow: 5th–95th percentile {q(std, .05):.1f}–{q(std, .95):.1f}. The "
     "chosen clip is therefore 1.47–1.75 for every slice (median about 1.58).",
     f"Compared with the default fixed clip 2.0, ACLAHE amplifies noise less (×{f(a['noise_gain'])} vs "
     f"×{f(c2['noise_gain'])}, about 18% less) for nearly the same local contrast (EME {f(a['eme'])} vs "
     f"{f(c2['eme'])}). A fixed clip of 4.0 adds more contrast (EME {f(c4['eme'])}) but amplifies noise "
     f"×{f(c4['noise_gain'])}.",
     f"It does not make slices more alike: the spread of per-slice contrast is {f(a['std_across_images'])} "
     f"versus {f(enh['None']['std_across_images'])} without enhancement, about the same as any fixed clip. Do "
     "not claim that it normalises the inputs."],
    "The constants (2.0, 60, 1–4) were set by inspection, not tuned. Because the adaptive limit is almost "
    "constant on this dataset, describe it honestly as a safeguard that selects a conservative clip (≈1.6); its "
    "adaptivity would matter more on multi-scanner data.",
    "The CLAHE clip limit was chosen per slice from its intensity spread; on IQ-OTH/NCCD this selects a "
    "conservative limit of about 1.6, which gives nearly the local contrast of the default setting while "
    "amplifying noise about 18% less.")

decision(
    "1.5 Pad to square, resize to 224 × 224 with area interpolation",
    "Zero-pad the shorter side, then INTER_AREA resize.",
    "Ours (the paper states 224 × 224 only).",
    "Direct resize (stretches), centre crop (cuts tissue), bilinear/bicubic interpolation.",
    ["224 × 224 is the resolution the ImageNet-pretrained ConvNeXt was trained at, so pretrained filters see "
     "structures at the scale they learned.",
     "Some slices are not square. Stretching would distort nodule shape, which is a diagnostic cue; cropping "
     "could remove the lesion. Padding keeps both.",
     "Area interpolation averages all source pixels under each output pixel. For downsampling (512 → 224) this "
     "avoids the aliasing and moiré that bilinear sampling produces on fine lung texture."],
    ["The final model is trained and tested on these images; no slice is distorted or cropped."],
    f"Padding the {odd} wide slices, which are almost all malignant, leaves black bars that only that class "
    "has (Section 1.0). Cropping the body region to a square instead would remove this cue; worth one run.",
    "Slices were padded to a square and downsampled with area interpolation to the 224 × 224 input size of the "
    "pretrained network, preserving aspect ratio and avoiding aliasing.")

decision(
    "1.6 Per-image z-score normalisation, grey channel copied to 3 channels",
    "(x − mean) / std per image; the single channel repeated three times.",
    "Z-score from the base paper; channel copy ours.",
    "ImageNet mean/std normalisation, min–max scaling, adapting the first convolution to one channel.",
    ["Z-scoring gives every image zero mean and unit variance, which keeps the scale of the inputs consistent "
     "and stabilises training.",
     "The images are 8-bit exports, not calibrated Hounsfield units, so there is no absolute intensity "
     "information to preserve.",
     "Pretrained ConvNeXt expects three channels; copying the grey channel uses the pretrained first layer "
     "unchanged, which is the standard approach for transfer learning to grey-scale medical images."],
    [],
    None,
    "Each slice was standardised to zero mean and unit variance and replicated into three channels to match "
    "the pretrained network's input.")

decision(
    "1.7 Training augmentation: rotation ±15°, scale 0.9–1.1, horizontal flip (p = 0.5)",
    "Random geometric transforms on training images only.",
    "Transform types from the base paper; ranges ours.",
    "No augmentation; stronger transforms (elastic, large rotations); intensity changes (brightness, noise).",
    ["Patients lie at slightly different angles and positions, and scanners differ in field of view; small "
     "rotations and scalings imitate this real variation [Shorten and Khoshgoftaar, 2019].",
     "Left–right flips are acceptable for classifying the slice as a whole (a nodule is malignant on either "
     "side).",
     "Intensity changes were avoided because density (brightness) is itself a diagnostic cue, and the "
     "preprocessing already standardises it. Large rotations or elastic warps would create anatomy that does "
     "not occur."],
    ["Applied after the split and only to training data; the test set is never augmented."],
    None,
    "Only mild geometric augmentation consistent with patient positioning was used, and only on training data.")

# =====================================================================  2. FEATURES
doc.add_heading("Stage 2 — Feature extraction", 1)
decision(
    "2.1 Backbone: ImageNet-pretrained ConvNeXt-Tiny",
    "ConvNeXt-Tiny (28 M parameters), ImageNet-1k weights, fine-tuned end to end.",
    "ConvNeXt from the base paper; Tiny instead of Base is ours (hardware).",
    "ConvNeXt-Base (paper's replica: 87.6 M), ResNet50, EfficientNet, Vision Transformer, training from scratch.",
    ["With about 1,000 images, training a deep network from scratch over-fits; starting from ImageNet weights "
     "is the standard remedy in medical imaging [Litjens et al., 2017; Kornblith et al., 2019].",
     "ConvNeXt modernises the ResNet design (large 7 × 7 depthwise kernels, LayerNorm, GELU, inverted "
     "bottlenecks) and matches vision transformers in accuracy while remaining a CNN [Liu et al., 2022], so it "
     "keeps the inductive bias that helps with small datasets.",
     "Tiny fits in 4 GB of GPU memory at batch 16 with mixed precision; Base does not without extra tricks "
     "(the replica needed gradient caching and took about three times longer)."],
    ["ConvNeXt-ECA head alone: 93.84% (80:20); ResNet50 trained the same way: 96.21% with 60% benign recall; "
     "ConvNeXt + quantum classifier: 98.10% with 90% benign recall.",
     "Training per epoch: ConvNeXt-Tiny 15.4 s vs ResNet50 79.5 s on the same machine.",
     "Two-by-two study: with a linear probe on frozen features, ConvNeXt gives 98.58% and ResNet50 about 98.1–"
     "98.6%, so the features of both are strong; ConvNeXt is mainly faster to train."],
    "We did not compare Tiny against Base with the same downstream classifier. Mention the hardware reason.",
    "An ImageNet-pretrained ConvNeXt-Tiny was fine-tuned as the feature extractor because transfer learning is "
    "essential at this dataset size and ConvNeXt combines modern accuracy with the efficiency needed for a 4 GB "
    "GPU.")

decision(
    "2.2 Efficient Channel Attention (ECA, k = 3) before pooling",
    "ECA layer on the last feature map: global average pool → 1-D convolution (3 weights) → sigmoid → rescale "
    "channels.",
    "Base paper.",
    "No attention; Squeeze-and-Excitation (SE) [Hu et al., 2018]; CBAM [Woo et al., 2018].",
    ["Channel attention lets the network weight the feature channels that respond to lesions more than those "
     "that respond to background.",
     "SE uses two fully-connected layers with a dimensionality reduction, which Wang et al. [2020] showed hurts "
     "the channel weights; ECA avoids the reduction and models only local cross-channel interaction, adding just "
     "3 parameters.",
     "Placed after the last stage, it re-weights the 768 channels that become the feature vector, which is "
     "where it most directly shapes the input to the classifier."],
    ["Adds 3 parameters and no measurable time."],
    "No with/without-ECA ablation was run. This is the easiest gap to close (one extra 16-minute training run).",
    "Efficient channel attention was added before pooling to re-weight the feature channels at a cost of only "
    "three parameters, avoiding the dimensionality reduction of SE blocks.")

decision(
    "2.3 Global average pooling + LayerNorm → 768-D feature",
    "ConvNeXt's own head normalisation kept; the 768-D vector is the output of the classical stage.",
    "Standard ConvNeXt design.",
    "Max pooling, flatten, adding a bottleneck layer.",
    ["Average pooling summarises the whole slice and is what the pretrained head was trained with.",
     "LayerNorm keeps every feature vector on the same scale, so the later 768 → 8 projection does not have to "
     "cope with images of very different feature magnitudes."],
    [],
    None,
    "The pooled and layer-normalised 768-dimensional vector of the backbone is used as the image descriptor.")

decision(
    "2.4 Optimiser: AdaBelief, lr 1e-4, β = (0.9, 0.999), ε = 1e-8",
    "AdaBelief for both the network and the quantum classifier.",
    "Base paper.",
    "SGD with momentum, Adam, AdamW [Loshchilov and Hutter, 2019].",
    ["AdaBelief scales each step by how far the gradient deviates from its running average (the \"belief\"), "
     "taking large steps when the gradient is consistent and small ones when it is noisy [Zhuang et al., 2020]. "
     "It is reported to combine Adam's fast convergence with SGD's generalisation.",
     "A small learning rate (1e-4) is standard for fine-tuning pretrained weights without destroying them."],
    ["Training converged in every run; the five seeds of the quantum stage vary by only ±0.19% accuracy."],
    "No AdamW comparison. Present this as the paper's choice, justified by the literature.",
    "AdaBelief was retained from the reproduced study for its adaptive, gradient-belief step size, with a low "
    "learning rate suited to fine-tuning.")

decision(
    "2.5 Batch 32 as two micro-batches of 16, mixed precision, 50 epochs",
    "Gradient accumulation over 2 × 16 images; automatic mixed precision (AMP).",
    "Batch size and epochs from the base paper; accumulation and AMP ours (hardware).",
    "Smaller true batch; a larger GPU.",
    ["A batch of 32 at 224 × 224 does not fit in 4 GB. Accumulating two micro-batches gives exactly the same "
     "gradient as one batch of 32, because ConvNeXt uses LayerNorm (per-sample) rather than BatchNorm, whose "
     "statistics would depend on the micro-batch size.",
     "Mixed precision halves memory and speeds up the tensor cores with no loss of accuracy in standard "
     "practice [Micikevicius et al., 2018]."],
    ["50 epochs take 959.8 s (16 min) on an RTX 3050 Laptop GPU.",
     "With only 5 epochs the head reaches 89.57% and the quantum classifier 94.22 ± 1.21%, so 50 epochs are "
     "needed."],
    None,
    "Gradient accumulation and mixed precision reproduced the paper's batch size of 32 on a 4 GB GPU; because "
    "ConvNeXt uses layer normalisation, accumulation is mathematically equivalent to a true batch of 32.")

decision(
    "2.6 Two-stage training: train the network, freeze it, then train the quantum classifier",
    "Stage 1 trains ConvNeXt-ECA with a temporary linear head; stage 2 trains projection + circuit on the "
    "frozen 768-D features.",
    "Our change (the paper trains end to end through the fidelity).",
    "End-to-end training through the quantum layer.",
    ["The fidelity of a test image depends on every training image, so end-to-end training must re-encode all "
     "reference images through the network at every step. This is slow and memory-hungry; the replica needed "
     "gradient caching and took 41–50 minutes per run.",
     "Freezing the network lets the 768-D features be computed once; the quantum stage then trains on vectors, "
     "not images, and the simulator can encode the references once per step.",
     "It also separates the two contributions, so the effect of the quantum classifier can be measured on its "
     "own (the two-by-two study)."],
    ["Total training 959.8 s + 396 s ≈ 23 min, versus 41–50 min for the end-to-end replica.",
     "Accuracy: 98.10% (ours) vs 97.73% (replica, same augment-after protocol)."],
    None,
    "Training the network and the quantum classifier in two stages halved the training time, made the quantum "
    "stage independent of image size, and allowed each stage's contribution to be measured separately.")

# =====================================================================  3. QUANTUM
doc.add_heading("Stage 3 — Quantum fidelity classifier", 1)
decision(
    "3.1 Eight qubits and a learned 768 → 8 projection with π·tanh",
    "z = π·tanh(W f + b): one angle per qubit, bounded to (−π, π).",
    "Eight qubits and the linear projection from the paper; the π·tanh bound is ours.",
    "PCA to 8 dimensions; amplitude encoding of all 768 features; more qubits.",
    ["Simulating n qubits costs 2ⁿ complex numbers; 8 qubits (256 amplitudes) keeps exact simulation fast and "
     "matches circuit sizes that current noisy hardware can run.",
     "A learned projection chooses the 8 directions that best separate the classes; PCA keeps the directions of "
     "largest variance, which need not be the discriminative ones.",
     "Rotation angles are periodic. Without a bound, two very different images could be mapped to angles that "
     "differ by a full period and give identical quantum states. tanh keeps every angle inside one period and "
     "is smooth, so gradients still flow."],
    ["The paper's unbounded projection is reproduced in the replica; our bounded version trains stably over all "
     "five seeds."],
    None,
    "Features were projected to eight rotation angles bounded by π·tanh, so that distinct images cannot "
    "collide through the periodicity of the rotation gates.")

decision(
    "3.2 Data re-uploading circuit: [RY(x) → RZ(θ) → RY(θ)] × 2 layers + CNOT ring, final RY(x)",
    "The data angles are encoded in every layer, interleaved with trainable rotations and entanglement.",
    "Our change (the paper encodes once, then applies the trainable gates).",
    "The paper's design: encode once, then trainable RY/RZ and CRZ gates.",
    ["The classifier compares states by fidelity |⟨ψ(x)|ψ(x′)⟩|². Any trainable unitary V applied after the "
     "encoding cancels: |⟨ψ(x)|V†V|ψ(x′)⟩|² = |⟨ψ(x)|ψ(x′)⟩|². In the paper's design the trainable gates "
     "therefore have no effect and receive zero gradient; the replica confirmed this.",
     "Re-uploading the data between trainable layers makes the trainable gates act between encodings, so they "
     "do change the similarity measure [Pérez-Salinas et al., 2020; Schuld et al., 2021].",
     "The CNOT ring entangles all qubits, so the similarity depends on combinations of features, not each "
     "feature separately."],
    ["Replica: the gradient of every trainable gate parameter was exactly zero.",
     "Simulator verified against Qiskit statevectors to a maximum difference of 1.45 × 10⁻⁷."],
    None,
    "Because trainable gates placed after a single encoding cancel out of the fidelity, the data were re-uploaded "
    "in every layer so that the circuit parameters actually shape the similarity between images.")

decision(
    "3.3 Class score = average fidelity to the training images of each class",
    "S_c = mean fidelity between the test state and all training states of class c.",
    "Base paper.",
    "Nearest neighbour (maximum fidelity); class-centroid state; quantum-kernel SVM [Havlíček et al., 2019].",
    ["Averaging over all training images of a class is robust to single mislabelled or atypical images, unlike "
     "nearest neighbour.",
     "It is equivalent to comparing with a class \"mean state\" (a density matrix), the fidelity classifier of "
     "Lloyd et al. [2020], and needs no optimisation of support vectors.",
     "The reference states are encoded once per evaluation, so classification of a new image costs only 0.149 ms "
     "in batch."],
    ["Quantum stage time: 0.149 ms per image (batched)."],
    None,
    "Each class is scored by the average fidelity of the test state to that class's training states, a "
    "robust, training-free comparison with the class's mean quantum state.")

decision(
    "3.4 Learnable temperature, initialised at 10",
    "p_c = softmax(τ · S_c), with τ learned.",
    "Our change (the paper uses softmax(S_c)).",
    "Plain softmax of the scores (paper); fixed temperature.",
    ["Fidelities lie in [0, 1]. With three classes, a plain softmax can at best give e¹/(e¹ + 2e⁰) ≈ 0.576 to "
     "the winning class, even for a perfect match. The cross-entropy loss then never approaches zero and the "
     "probabilities are not usable as confidences.",
     "A learned temperature lets the model choose how sharp the probabilities should be; starting at 10 already "
     "allows confident predictions from the first epoch."],
    ["Replica (plain softmax): mean top probability 56.2–57.1%, as the bound predicts."],
    None,
    "A learnable temperature was introduced because a softmax over fidelities in [0, 1] cannot exceed 57.6% "
    "confidence for three classes.")

decision(
    "3.5 Same-source exclusion during training",
    "When a training image is scored, its own state and its augmented copies are removed from the references.",
    "Our change.",
    "Score against all training images.",
    ["An image always has fidelity 1 with itself (and near 1 with its own rotated copy), so including it makes "
     "the training task trivial and the loss uninformative. Excluding it imitates the test situation, where the "
     "image is never among the references."],
    [],
    None,
    "During training each image was compared only with other source images, mirroring the test condition.")

decision(
    "3.6 Exact statevector simulation in PyTorch",
    "Circuit simulated exactly on the GPU, differentiable with autograd.",
    "Ours (the paper does not describe its simulator).",
    "Qiskit/PennyLane simulators; shot-based simulation; real quantum hardware.",
    ["Exact simulation gives noise-free fidelities, so the results measure the method rather than hardware "
     "noise or sampling error.",
     "Writing it in PyTorch lets the gradients flow through the circuit with the same autograd as the network, "
     "and runs batched on the GPU.",
     "Correctness was checked against Qiskit."],
    ["Maximum difference to Qiskit: 1.45 × 10⁻⁷."],
    "Real hardware would add noise and require many measurement shots per fidelity; state this in Future Scope.",
    "The circuit was simulated exactly as a statevector in PyTorch, verified against Qiskit, so that results "
    "reflect the method rather than hardware noise.")

# =====================================================================  4. EVALUATION
doc.add_heading("Stage 4 — Evaluation", 1)
decision(
    "4.1 Five seeds, mean ± standard deviation",
    "Seeds 0–4 for the quantum stage and the comparison classifiers.",
    "Ours (the paper reports single numbers).",
    "One run; k-fold cross-validation.",
    ["With 211 test images, one image is 0.47% accuracy. A single run cannot tell a real 0.5% difference from "
     "luck; the spread over seeds shows the size of that luck [Bouthillier et al., 2021]."],
    ["80:20 quantum: 98.01 ± 0.19%; 70:30: 96.91 ± 0.61%."],
    None,
    "Every stochastic result is reported as the mean and standard deviation over five seeds.")

decision(
    "4.2 Metrics: accuracy, macro precision/recall/F1, MCC, AUC, per-class recall",
    "All metrics reported on the untouched test set.",
    "Paper's metrics plus per-class recall (ours).",
    "Accuracy only.",
    ["Benign is about 10% of the test set, so accuracy can stay high while benign cases are missed. Macro "
     "averages weight each class equally; MCC uses the whole confusion matrix and is reliable under imbalance "
     "[Chicco and Jurman, 2020].",
     "Per-class recall shows exactly where the model fails (always benign) and is what separated our model "
     "(90%) from ResNet50 (60%)."],
    [],
    None,
    "Because the classes are imbalanced, macro-averaged metrics, MCC and per-class recall are reported alongside "
    "accuracy.")

decision(
    "4.3 Controls: ResNet50, classical head, linear probe, MLP",
    "ResNet50 trained with the identical protocol; linear and MLP classifiers on the same frozen features.",
    "ResNet50 from the paper's comparison; the controls are ours.",
    "Copy baseline numbers from other papers (the base paper's approach).",
    ["A baseline is only fair if trained on the same data, split and budget. Numbers copied from other studies "
     "(different datasets, different metrics) cannot support a claim of superiority.",
     "Linear and MLP classifiers on the same features isolate the classifier: if they match the quantum "
     "classifier, the advantage is in the features, not the circuit."],
    ["Linear probe 98.58%, MLP 98.48 ± 0.19%, quantum 98.01 ± 0.19% (80:20)."],
    None,
    "All baselines were retrained under the identical protocol, and simple classical classifiers on the same "
    "features were used as controls to isolate the contribution of the quantum classifier.")

# =====================================================================  PILLARS
doc.add_heading("The supporting pillars of the thesis", 1)
doc.add_paragraph("The decisions above group into five arguments. Each can open a section of the Discussion "
                  "chapter or a paragraph of the Introduction's contributions.")
table(["Pillar", "Claim", "Decisions that support it", "Strongest evidence"],
      [["1. Leak-free evaluation", "Our numbers measure generalisation to unseen slices.",
        "0.2, 0.3, 1.7, 4.1", "Replica: 98.97% before vs 97.73% after splitting"],
       ["2. Image quality before learning", "Preprocessing removes noise without losing edges, then raises "
        "local contrast without amplifying that noise.", "1.0–1.6",
        "Adaptive NLM cuts the grainy slices' noise 25 → 2; global HE amplifies noise ×8 vs ×5 for ACLAHE"],
       ["3. Efficient transfer learning", "A pretrained modern CNN gives strong features cheaply on a laptop "
        "GPU.", "2.1–2.6", "ConvNeXt 15.4 s/epoch vs ResNet50 79.5 s; 98.10% vs 96.21%"],
       ["4. A quantum classifier that actually learns", "Our circuit fixes two flaws of the original design.",
        "3.1–3.6", "Zero gradients and 57.6% confidence cap in the replica"],
       ["5. Honest attribution", "The gain over ResNet50 comes from features and protocol, not quantum "
        "advantage.", "4.1–4.3", "Linear probe 98.58% ≈ quantum 98.01%"]],
      size=9, widths=[1.3, 2.0, 1.2, 2.0])

doc.add_heading("Gaps worth closing before submission", 1)
doc.add_paragraph("Each is one training run (about 16 minutes for stage 1 plus a few minutes for stage 2) on "
                  "the 80:20 split, reported with the linear probe and the quantum classifier:")
bullets([
    "Bilateral instead of NLM: same accuracy would make pre-processing about 10× faster.",
    "Preprocessing ablation: (a) no denoising, (b) fixed-clip CLAHE, (c) no enhancement. This turns pillar 2 "
    "from image-quality evidence into accuracy evidence.",
    "Shortcut check (no retraining needed): test accuracy on grainy versus clean normal slices and on wide "
    "versus square malignant slices; optionally retrain with wide slices cropped instead of padded.",
    "ECA ablation: ConvNeXt-Tiny without ECA.",
    "Imbalance handling: weighted cross-entropy instead of benign copies.",
    "Optional: AdamW instead of AdaBelief.",
])

doc.add_heading("References to cite", 1)
for ref in [
    "Al-Yasriy, H. F., et al. (2020). Diagnosis of lung cancer based on CT scans using CNN. IOP Conf. Ser.: "
    "Mater. Sci. Eng., 928, 022035. (IQ-OTH/NCCD dataset)",
    "Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. MLSys.",
    "Buades, A., Coll, B., and Morel, J.-M. (2005). A non-local algorithm for image denoising. CVPR.",
    "Chang, S. G., Yu, B., and Vetterli, M. (2000). Adaptive wavelet thresholding for image denoising and "
    "compression. IEEE Trans. Image Processing, 9(9).",
    "Chicco, D., and Jurman, G. (2020). The advantages of the Matthews correlation coefficient over F1 score "
    "and accuracy in binary classification evaluation. BMC Genomics, 21.",
    "Dabov, K., Foi, A., Katkovnik, V., and Egiazarian, K. (2007). Image denoising by sparse 3-D "
    "transform-domain collaborative filtering. IEEE Trans. Image Processing, 16(8).",
    "Diwakar, M., and Kumar, M. (2018). A review on CT image noise and its denoising. Biomedical Signal "
    "Processing and Control, 42.",
    "Donoho, D. L., and Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. Biometrika, "
    "81(3).",
    "Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2.",
    "Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature, 567.",
    "He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. CVPR.",
    "Hu, J., Shen, L., and Sun, G. (2018). Squeeze-and-excitation networks. CVPR.",
    "Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. (2012). Leakage in data mining. ACM TKDD, 6(4).",
    "Kornblith, S., Shlens, J., and Le, Q. V. (2019). Do better ImageNet models transfer better? CVPR.",
    "Lin, T.-Y., et al. (2017). Focal loss for dense object detection. ICCV.",
    "Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis, "
    "42.",
    "Liu, Z., et al. (2022). A ConvNet for the 2020s. CVPR.",
    "Lloyd, S., Schuld, M., Ijaz, A., Izaac, J., and Killoran, N. (2020). Quantum embeddings for machine "
    "learning. arXiv:2001.03622.",
    "Loshchilov, I., and Hutter, F. (2019). Decoupled weight decay regularization. ICLR.",
    "Micikevicius, P., et al. (2018). Mixed precision training. ICLR.",
    "Pérez-Salinas, A., et al. (2020). Data re-uploading for a universal quantum classifier. Quantum, 4, 226.",
    "Pizer, S. M., et al. (1987). Adaptive histogram equalization and its variations. Computer Vision, "
    "Graphics, and Image Processing, 39(3).",
    "Roberts, M., et al. (2021). Common pitfalls and recommendations for using machine learning to detect and "
    "prognosticate for COVID-19 using chest radiographs and CT scans. Nature Machine Intelligence, 3.",
    "Rudin, L. I., Osher, S., and Fatemi, E. (1992). Nonlinear total variation based noise removal "
    "algorithms. Physica D, 60.",
    "Schuld, M., Sweke, R., and Meyer, J. J. (2021). Effect of data encoding on the expressive power of "
    "variational quantum machine-learning models. Physical Review A, 103.",
    "Shorten, C., and Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. "
    "Journal of Big Data, 6.",
    "Tomasi, C., and Manduchi, R. (1998). Bilateral filtering for gray and color images. ICCV.",
    "Wang, Q., et al. (2020). ECA-Net: Efficient channel attention for deep convolutional neural networks. "
    "CVPR.",
    "Woo, S., Park, J., Lee, J.-Y., and Kweon, I. S. (2018). CBAM: Convolutional block attention module. ECCV.",
    "Zhuang, J., et al. (2020). AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients. "
    "NeurIPS.",
    "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics Gems IV, Academic Press.",
]:
    doc.add_paragraph(ref, style="List Bullet")

doc.add_heading("Files", 1)
bullets(["CNN/Rationale/preprocessing_evidence.py — the denoising and contrast measurements",
         "CNN/Rationale/Results/ — denoising.csv, enhancement.csv, dataset_stats.csv, noise_shortcut.csv and the "
         "three figures",
         "CNN/Rationale/make_rationale_report.py — builds this report"])

REPORT.parent.mkdir(exist_ok=True)
doc.save(REPORT)
print("Saved", REPORT)
