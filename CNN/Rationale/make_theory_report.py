from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "run_reports" / "Pipeline_Design_Rationale.docx"

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)


def para(text):
    p = doc.add_paragraph()
    parts = text.split("**")
    for i, part in enumerate(parts):
        p.add_run(part).bold = i % 2 == 1
    return p


def bullets(lines):
    for line in lines:
        p = doc.add_paragraph(style="List Bullet")
        for i, part in enumerate(line.split("**")):
            p.add_run(part).bold = i % 2 == 1


def table(header, rows, widths=None, size=9.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for r in rows:
        for cell, text in zip(t.add_row().cells, r):
            cell.text = text
    for row in t.rows:
        for k, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[k])
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
    doc.add_paragraph()


def component(title, what, why, alternatives, origin):
    doc.add_heading(title, 2)
    p = para(f"**What it does.** {what}")
    p = doc.add_paragraph()
    r = p.add_run(f"Origin: {origin}")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    para("**Why it suits this task.**")
    bullets(why)
    table(["Alternative", "Why it is less suitable here"], alternatives, widths=[1.7, 4.8])


# =====================================================================  TITLE
doc.add_heading("Pipeline Design Rationale", 0)
p = doc.add_paragraph(f"Why each component of the lung CT classification pipeline was chosen · Theory · "
                      f"{date.today():%d %B %Y}")
p.runs[0].font.color.rgb = RGBColor(0x59, 0x59, 0x59)

doc.add_heading("1. Purpose and scope", 1)
para("This report explains, component by component, why the pipeline is built the way it is and why each choice "
     "is better suited to this problem than the usual alternatives. It is a theoretical justification drawn "
     "from the properties of the data and the literature. Measured evidence (ablations and benchmarks) will be "
     "appended in Section 9.")
para("Each component lists its **origin**: 'base paper' means we kept the choice from the reproduced QCF-ELCD "
     "study; 'ours' means we changed or added it. For the inherited choices, the argument is that the choice is "
     "technically sound for this task; for our own changes, the argument is that they fix a specific problem.")

doc.add_heading("2. What the problem demands", 1)
para("Every choice below follows from six properties of the task. They are the yardstick for \"better\" "
     "throughout this report.")
table(["Property of the task", "Consequence for the design"],
      [["Small dataset: about 1,050 unique CT slices", "Training a deep network from scratch would over-fit; "
        "transfer learning and strong regularisation are needed. Every image matters, so leakage must be "
        "avoided."],
       ["Class imbalance: benign is about 10% of the data", "The minority class must be boosted, and accuracy "
        "alone is not a sufficient metric."],
       ["Grey-scale, 8-bit CT exports, variable noise", "Images need denoising and contrast enhancement; "
        "absolute Hounsfield values are not available."],
       ["Small, subtle findings (nodules, texture)", "Preprocessing must preserve edges and fine texture; "
        "resizing must not distort shape."],
       ["Limited hardware: 4 GB laptop GPU", "Model size, precision and batch handling must fit in memory and "
        "train in reasonable time."],
       ["Research aim: a quantum classifier", "The classical part must hand the quantum part a compact, "
        "informative vector, and the circuit must be simulable and genuinely trainable."]],
      widths=[2.3, 4.2])

doc.add_heading("3. Pipeline overview", 1)
doc.add_picture(str(HERE / "pipeline.png"), width=Inches(6.4))
para("A CT slice is cleaned and enhanced, a pretrained convolutional network turns it into a 768-number "
     "description, a learned projection compresses that to 8 rotation angles, an 8-qubit circuit turns the "
     "angles into a quantum state, and the slice is assigned to the class whose training states it most "
     "resembles (highest average fidelity).")

# =====================================================================  4. DATA
doc.add_heading("4. Data handling", 1)
component(
    "4.1 Removing duplicate slices",
    "Exact and near-identical images are removed before the data are split (1,097 → 1,054 slices).",
    ["If the same slice appears in both the training and test sets, the model is tested on an image it has "
     "already memorised. This is data leakage and inflates accuracy [Kaufman et al., 2012].",
     "With only about 1,000 images, even a few leaked duplicates shift accuracy by whole percentage points.",
     "Removing duplicates is the only way to guarantee that every test image is genuinely unseen."],
    [["Keep the dataset as distributed", "Leaves leakage in place; reported accuracy no longer measures "
      "generalisation."],
     ["Remove duplicates only from the test set", "Still lets near-copies influence training in an "
      "unpredictable way; cleaner to de-duplicate once, globally."]],
    "ours")

component(
    "4.2 Split first, augment second; stratified split",
    "Unique slices are split into train and test sets with class proportions preserved; augmentation is then "
    "applied to the training set only.",
    ["Augmenting before the split puts rotated or flipped copies of test images into the training set. They "
     "are almost identical to the originals, so the test becomes a memory test.",
     "Stratification keeps the benign share equal in both sets; a random split of a 10% class could leave the "
     "test set with too few benign cases to measure.",
     "The 80:20 and 70:30 ratios match the base paper, so results are directly comparable."],
    [["Augment then split (base paper)", "Leaks augmented copies of test images into training."],
     ["Plain random split", "Class proportions can drift, especially for the small benign class."],
     ["k-fold cross-validation", "Statistically stronger, but k times the training cost on a laptop GPU; "
      "repeated seeds on a fixed split were chosen as the affordable alternative."]],
    "split ratios from the base paper; ordering ours")

component(
    "4.3 Balancing the benign class with augmented copies",
    "Each training benign slice is kept plus three augmented copies, so benign is no longer heavily "
    "under-represented in training.",
    ["With 10% benign, a network can reach high accuracy while largely ignoring the class. Oversampling makes "
     "benign errors cost as much as the others during training.",
     "Using augmented copies (rotated, scaled, flipped) rather than exact repeats gives the network new views "
     "instead of letting it memorise the same 82 images [Shorten and Khoshgoftaar, 2019].",
     "It works unchanged with every downstream classifier, including the fidelity classifier, whose class "
     "scores average over the training images of each class."],
    [["Class-weighted loss", "Equally valid for the network, but it does not add new training views, and the "
      "fidelity classifier would need its own weighting scheme."],
     ["Focal loss", "Designed for extreme imbalance in detection; adds a tuning parameter for a moderate "
      "imbalance."],
     ["SMOTE", "Interpolates between images pixel by pixel and produces unrealistic CT slices."],
     ["Undersampling the majority classes", "Throws away scarce data."]],
    "ours (the base paper augments benign but does not state the factor)")

# =====================================================================  5. PREPROCESSING
doc.add_heading("5. Pre-processing", 1)
para("Order matters: **denoise first, then enhance contrast**. Contrast enhancement amplifies whatever is in "
     "the image, noise included, so noise must be removed before it is amplified.")

component(
    "5.1 Non-local means (NLM) denoising",
    "Replaces each pixel with a weighted average of pixels whose surrounding 7 × 7 patch looks similar, "
    "searched within a 21 × 21 window [Buades et al., 2005].",
    ["Noise in reconstructed CT is approximately Gaussian and spread over the whole image [Diwakar and Kumar, "
     "2018], which is the noise model NLM is designed for.",
     "CT anatomy is highly repetitive (lung parenchyma, vessel cross-sections, rib edges). NLM exploits this: "
     "each pixel finds many similar patches to average with, so noise is reduced strongly.",
     "Edges are preserved because patches on opposite sides of an edge do not look alike and are not "
     "averaged together. Nodule borders and vessel walls are exactly the details the classifier needs.",
     "It needs no training data, so it cannot over-fit or leak information, and it is fast enough to run on "
     "every image at inference time."],
    [["Gaussian (linear) blur", "Averages across edges as much as within regions, so it blurs nodule borders "
      "and fine texture along with the noise."],
     ["Median filter", "Designed for impulse (salt-and-pepper) noise, which CT does not have; erodes thin "
      "structures such as small vessels."],
     ["Bilateral filter", "Edge-preserving, but compares single pixel values rather than whole patches, so it "
      "is less robust when noise is strong. A reasonable, faster alternative."],
     ["Total-variation denoising", "Preserves edges but produces flat, cartoon-like regions (staircasing) that "
      "remove lung texture."],
     ["Wavelet shrinkage", "Good for smooth images; can leave ringing artefacts around sharp edges."],
     ["BM3D", "Usually the best classical denoiser, but many times slower than NLM; it would dominate "
      "inference time."],
     ["Deep-learning denoisers", "Need pairs of noisy and clean scans (e.g. low-dose/full-dose) that this "
      "dataset does not provide."]],
    "method from the base paper; parameters ours")

component(
    "5.2 Noise-adaptive NLM strength",
    "The filter strength h is set per slice from that slice's estimated noise level σ̂: h = max(1.15 σ̂, 3).",
    ["Slices differ in noise level. One fixed strength either leaves noisy slices noisy or blurs clean ones.",
     "Buades et al. tie the best h to the noise standard deviation; estimating σ̂ per slice applies that rule "
     "automatically, without manual tuning.",
     "The floor of 3 guarantees a light, uniform clean-up even on very clean slices.",
     "If noise level happens to differ between classes (as acquisition differences often do), adaptive "
     "strength also stops the network from using \"how grainy is the image\" as a shortcut to the label "
     "[Geirhos et al., 2020]."],
    [["One fixed h for all slices", "Mismatched to most slices: too strong for clean ones, too weak for noisy "
      "ones."],
     ["Hand-tuned h per subset", "Not reproducible and impossible at inference time on a new scan."]],
    "ours (the base paper gives no value)")

component(
    "5.3 Contrast-limited adaptive histogram equalisation (CLAHE)",
    "Splits the slice into 8 × 8 tiles, equalises the histogram of each tile with a cap (clip limit) on how "
    "much any grey level can be stretched, and blends neighbouring tiles smoothly [Zuiderveld, 1994].",
    ["Lung tissue and small nodules occupy a narrow band of grey levels in 8-bit CT exports, so they look "
     "flat. Stretching that band locally makes texture and boundaries visible.",
     "Working tile by tile adapts the enhancement to each region (lung field, mediastinum, chest wall) instead "
     "of letting the large dark background dominate.",
     "The clip limit caps noise amplification in flat regions, which is the main failure of plain adaptive "
     "histogram equalisation [Pizer et al., 1987].",
     "It is cheap (about a millisecond) and deterministic, and is the standard enhancement step in medical "
     "image pipelines."],
    [["No enhancement", "Leaves low-contrast detail hard to separate, especially after 8-bit export."],
     ["Global histogram equalisation", "One mapping for the whole image: the background dominates, brightness "
      "shifts and noise is amplified strongly."],
     ["Plain adaptive HE (no clip)", "Over-amplifies noise in uniform regions."],
     ["Gamma correction", "Brightens or darkens globally but adds no local contrast."],
     ["Unsharp masking", "Sharpens edges but also sharpens noise and adds halo artefacts."]],
    "base paper (as ACLAHE)")

component(
    "5.4 Adaptive clip limit (ACLAHE)",
    "The clip limit is chosen per slice from the slice's intensity spread σ: clip = 2.0 × 60 / σ, limited to "
    "the range 1–4.",
    ["Flat, low-contrast slices need stronger enhancement; already contrasty slices need less, or noise is "
     "amplified for no benefit.",
     "Scaling the limit inversely with σ applies that automatically, so no slice is over- or under-enhanced.",
     "The 1–4 range keeps the limit inside the values normally used with CLAHE.",
     "The rule has no learned parameters, so it cannot leak test information."],
    [["One fixed clip limit", "Simpler; adequate when all slices have similar contrast, but cannot adapt when "
      "they do not."],
     ["Learned enhancement", "Needs training data and adds a component that can over-fit."]],
    "idea from the base paper (\"adaptive\"); formula ours")

component(
    "5.5 Pad to square, resize to 224 × 224 with area interpolation",
    "The shorter side is zero-padded to make the slice square, then the slice is downsampled to 224 × 224 by "
    "averaging all source pixels under each output pixel (INTER_AREA).",
    ["224 × 224 is the resolution the pretrained network learned its filters at, so structures appear at the "
     "scale the filters expect.",
     "Padding keeps the aspect ratio. Nodule shape is a diagnostic cue; stretching would distort it.",
     "Area interpolation is the correct filter for shrinking an image: it averages rather than samples, so "
     "fine, repetitive lung texture does not alias into false patterns (moiré)."],
    [["Stretch to 224 × 224", "Distorts shapes on non-square slices."],
     ["Centre crop", "Can cut off a peripheral lesion."],
     ["Bilinear / bicubic downsampling", "Sample only a few source pixels per output pixel when shrinking by "
      "more than 2×, causing aliasing."],
     ["Keep full 512 × 512", "About five times the memory and compute, with no matching pretrained "
      "resolution."]],
    "ours (the base paper states 224 × 224 only)")

component(
    "5.6 Per-image z-score, grey channel copied to three",
    "Each slice is shifted to zero mean and scaled to unit standard deviation; the single grey channel is "
    "repeated three times.",
    ["Consistent input scale makes optimisation stable and lets one learning rate suit every image.",
     "The images are 8-bit exports, not calibrated Hounsfield units, so there is no absolute intensity "
     "information worth preserving.",
     "The pretrained network expects three colour channels. Copying the grey channel reuses the pretrained "
     "first layer unchanged, which is the standard way to transfer ImageNet models to grey-scale medical "
     "images."],
    [["ImageNet mean/std normalisation", "Calibrated for colour photographs, not CT intensity "
      "distributions."],
     ["Min–max scaling to [0, 1]", "Sensitive to single bright or dark pixels."],
     ["Re-train the first layer for one channel", "Discards pretrained first-layer weights for no gain."]],
    "z-score from the base paper; channel copy ours")

component(
    "5.7 Training augmentation: rotation ±15°, scale 0.9–1.1, horizontal flip",
    "Random small geometric changes are applied to training images at every epoch.",
    ["Patients lie at slightly different angles and positions, and scanners use different fields of view. "
     "Small rotations and scalings imitate exactly this real variation [Shorten and Khoshgoftaar, 2019].",
     "Left–right flipping is valid for whole-slice classification: a malignant nodule is malignant on either "
     "side of the chest.",
     "Intensity augmentation was avoided because brightness (tissue density) is itself a diagnostic cue and "
     "pre-processing already standardises it."],
    [["No augmentation", "With about 1,000 images the network would over-fit quickly."],
     ["Large rotations, elastic warps", "Produce anatomy that never occurs in real scans."],
     ["Brightness / contrast / noise jitter", "Alters the density cues the classifier relies on."],
     ["Mixup / CutMix", "Blend images and labels across classes; the result is not a meaningful CT slice."]],
    "transform types from the base paper; ranges ours")

# =====================================================================  6. FEATURES
doc.add_heading("6. Feature extraction", 1)
component(
    "6.1 Transfer learning from ImageNet",
    "The network starts from weights trained on 1.28 million natural images and is then fine-tuned on the CT "
    "slices.",
    ["About 1,000 training images are far too few to learn a deep network's 28 million parameters from "
     "scratch; it would memorise the training set.",
     "Early convolutional layers learn generic edges, blobs and textures that transfer across domains, "
     "including to medical images [Litjens et al., 2017; Kornblith et al., 2019].",
     "Fine-tuning (rather than freezing) lets the higher layers adapt to CT-specific patterns such as nodule "
     "shape and parenchymal texture."],
    [["Train from scratch", "Severe over-fitting on about 1,000 images; much slower to converge."],
     ["Frozen ImageNet features (no fine-tuning)", "Features tuned to natural photographs, not CT; less "
      "discriminative for lung findings."],
     ["Self-supervised pretraining on CT", "Potentially better, but needs a large unlabelled CT corpus and "
      "far more compute than available."]],
    "base paper")

component(
    "6.2 ConvNeXt backbone",
    "A convolutional network that adopts the design choices of vision transformers: large 7 × 7 depthwise "
    "kernels, inverted bottlenecks, LayerNorm, GELU activations and fewer normalisation layers [Liu et al., "
    "2022].",
    ["ConvNeXt matches the accuracy of vision transformers on ImageNet while remaining a CNN, so it keeps the "
     "convolutional inductive biases (locality, translation equivariance) that help on small datasets.",
     "Large 7 × 7 kernels see more context per layer than ResNet's 3 × 3, useful for judging a nodule "
     "against its surroundings.",
     "It uses LayerNorm instead of BatchNorm. LayerNorm normalises each image on its own, so behaviour does "
     "not depend on batch size: small or split batches on a 4 GB GPU behave exactly like large ones.",
     "Depthwise convolutions give high accuracy per FLOP, so training and inference are fast."],
    [["ResNet50", "Older design with 3 × 3 kernels and BatchNorm; lower ImageNet accuracy at similar cost; "
      "BatchNorm statistics become noisy at small batch sizes."],
     ["Vision Transformer (ViT)", "Lacks convolutional inductive bias and needs very large datasets or heavy "
      "regularisation to train well [Dosovitskiy et al., 2021]."],
     ["EfficientNet", "Very parameter-efficient, but uses BatchNorm and depthwise-separable blocks that are "
      "slower per FLOP on GPUs and sensitive to small batches."],
     ["DenseNet / VGG", "Older architectures: DenseNet is memory-hungry; VGG is large and less accurate."]],
    "base paper")

component(
    "6.3 ConvNeXt-Tiny rather than a larger variant",
    "The smallest standard ConvNeXt: 28 M parameters, 768 output channels.",
    ["Larger variants (Small 50 M, Base 88 M) do not fit training at batch 32 in 4 GB without spilling to "
     "system memory, which slows training many times over.",
     "With about 1,000 training images, a larger network adds capacity the data cannot use and raises the risk "
     "of over-fitting.",
     "Tiny's 768-dimensional output is already ample for a 3-class problem."],
    [["ConvNeXt-Base (as in the base paper's setup)", "About three times the parameters; exceeds 4 GB; much "
      "slower; no data to justify the capacity."],
     ["ConvNeXt-Small", "Same 768-D output as Tiny with nearly double the cost."]],
    "ours (hardware and data size)")

component(
    "6.4 Efficient Channel Attention (ECA)",
    "After the last stage, each of the 768 channels is summarised by its average value; a 1-D convolution "
    "over neighbouring channels (kernel 3) and a sigmoid produce a weight per channel that rescales it "
    "[Wang et al., 2020].",
    ["Not every feature channel is useful for this task; attention lets the network emphasise channels that "
     "respond to lesions and suppress those that respond to background.",
     "ECA adds only 3 parameters, so it cannot over-fit and costs no measurable time.",
     "Unlike Squeeze-and-Excitation, it does not squeeze the channels through a bottleneck; Wang et al. showed "
     "that this dimensionality reduction harms the learned weights.",
     "Placed before pooling, it directly shapes the 768-D vector that the quantum classifier receives."],
    [["No attention", "Simplest; relies on the backbone alone to weight channels."],
     ["Squeeze-and-Excitation (SE)", "Two fully-connected layers with a bottleneck: about 74 k extra "
      "parameters at 768 channels, and the reduction loses channel detail [Hu et al., 2018]."],
     ["CBAM", "Adds spatial attention as well; more parameters and compute for a modest gain on "
      "classification [Woo et al., 2018]."],
     ["Self-attention / transformer head", "Many parameters; prone to over-fitting at this data size."]],
    "base paper")

component(
    "6.5 Global average pooling + LayerNorm → 768-D vector",
    "The 7 × 7 × 768 feature map is averaged over space and layer-normalised.",
    ["Averaging summarises the whole slice into one vector independent of where the finding lies.",
     "It is how the pretrained ConvNeXt head was trained, so it matches what the backbone expects.",
     "LayerNorm puts every image's vector on the same scale, so the later projection to 8 angles behaves the "
     "same for every image."],
    [["Global max pooling", "Driven by one location; sensitive to noise or artefacts."],
     ["Flatten the feature map", "37,632 values per image; far too many for the small dataset and the "
      "8-qubit circuit."]],
    "standard ConvNeXt design")

component(
    "6.6 AdaBelief optimiser (lr 1e-4)",
    "An adaptive optimiser that scales each parameter's step by how much the gradient deviates from its "
    "running average (its \"belief\") rather than by the gradient's raw size [Zhuang et al., 2020].",
    ["When gradients are consistent, AdaBelief takes large steps (like SGD with momentum); when they are noisy, "
     "it takes small ones. This combines Adam's fast convergence with SGD's better generalisation.",
     "Fine-tuning on small, noisy medical datasets is exactly the regime where step-size stability matters.",
     "A small learning rate (10⁻⁴) adapts pretrained weights gently instead of overwriting them."],
    [["SGD with momentum", "Generalises well but converges slowly and needs careful learning-rate "
      "scheduling."],
     ["Adam", "Fast, but scales by gradient magnitude, which can give overly large steps on noisy gradients "
      "and slightly worse generalisation."],
     ["AdamW", "Adam with correct weight decay; a strong, widely used alternative. AdaBelief was kept to "
      "follow the base paper."]],
    "base paper")

component(
    "6.7 Mixed precision and gradient accumulation",
    "Most arithmetic runs in 16-bit floating point; a batch of 32 can be processed as two micro-batches of 16 "
    "whose gradients are summed before the update.",
    ["Mixed precision roughly halves memory and uses the GPU's tensor cores, giving a large speed-up with no "
     "loss of accuracy in standard practice [Micikevicius et al., 2018].",
     "Gradient accumulation reproduces the paper's batch size of 32 within 4 GB.",
     "Because ConvNeXt uses LayerNorm (per image) and not BatchNorm (per batch), accumulating two halves gives "
     "exactly the same gradient as one full batch. The same trick on a BatchNorm network like ResNet50 would "
     "change the result."],
    [["Full 32-bit precision", "About twice the memory and several times slower on a laptop GPU."],
     ["Smaller true batch size", "Noisier gradients and a different training regime from the base paper."]],
    "batch size from the base paper; method ours (hardware)")

component(
    "6.8 Two-stage training: network first, then the quantum classifier",
    "Stage 1 fine-tunes ConvNeXt-ECA with a temporary linear head. Stage 2 freezes it, computes every image's "
    "768-D vector once, and trains the projection and quantum circuit on those vectors.",
    ["The fidelity classifier compares each image with every training image. Training end to end would mean "
     "pushing all reference images through the network at every step, which is slow and memory-hungry.",
     "Freezing the network makes the quantum stage independent of image size: it trains on vectors, not "
     "images, in minutes.",
     "It separates the two contributions, so the effect of the quantum classifier can be measured on its own, "
     "and any classifier (linear, MLP, quantum) can be compared on identical features.",
     "Training the classifier on a fixed feature space is also more stable than chasing a moving one."],
    [["End-to-end training (base paper)", "Several times slower; needs gradient caching to fit in memory; "
      "mixes the effects of network and circuit."],
     ["Frozen ImageNet features", "Not adapted to CT (see 6.1)."]],
    "ours")

# =====================================================================  7. QUANTUM
doc.add_heading("7. Quantum fidelity classifier", 1)
component(
    "7.1 A fidelity (quantum-kernel) classifier",
    "Each image is turned into a quantum state; similarity between two images is the fidelity |⟨ψ(x)|ψ(x′)⟩|², "
    "the squared overlap of their states. A test image is assigned to the class whose training states it "
    "overlaps most.",
    ["Fidelity is a kernel: it measures similarity in a 2⁸ = 256-dimensional complex space that a classical "
     "computer handles only by explicit simulation [Schuld and Killoran, 2019; Havlíček et al., 2019]. This is "
     "the component whose value the project sets out to test.",
     "It is a similarity-based (prototype) classifier, which suits small datasets: it needs no large output "
     "layer, only a measure of resemblance to known examples.",
     "Its predictions are explainable: the most similar training scans can be shown alongside each prediction.",
     "The circuit needs only 8 qubits and is shallow, so it could in principle run on current noisy quantum "
     "hardware."],
    [["Linear / softmax head", "The classical baseline: fast and strong, but not the research question. Used "
      "as a control."],
     ["Small MLP", "Similar to the linear head; a further classical control."],
     ["Quantum-kernel SVM [Havlíček et al., 2019]", "Needs the full train-by-train kernel matrix and a "
      "quadratic optimisation; the class-average rule is simpler and trainable end to end."],
     ["Variational classifier with measured outputs", "Prone to barren plateaus as circuits grow; predictions "
      "depend on noisy expectation values."]],
    "base paper")

component(
    "7.2 Learned 768 → 8 projection, bounded by π·tanh",
    "A linear layer maps the 768-D feature to 8 numbers; z = π·tanh(Wf + b) squeezes each into the range "
    "(−π, π). Each becomes the rotation angle for one qubit.",
    ["Simulating n qubits costs 2ⁿ complex numbers. Eight qubits (256 amplitudes) keep exact simulation fast "
     "and match circuit sizes runnable on today's hardware.",
     "A learned projection picks the 8 directions that best separate the classes; unsupervised reduction "
     "such as PCA keeps the directions of largest variance, which need not be the useful ones.",
     "Rotation gates are periodic: an angle and the same angle plus a full period produce the same state. "
     "Without a bound, two very different images can land on equivalent angles and become indistinguishable. "
     "tanh confines every angle to one period while staying smooth, so gradients still flow."],
    [["Unbounded linear projection (base paper)", "Angles can wrap around and collide."],
     ["PCA to 8 components", "Not trained for classification; keeps variance, not discrimination."],
     ["Amplitude encoding of all 768 values", "Needs only 10 qubits but a deep, hardware-unfriendly "
      "state-preparation circuit, and normalisation discards the vector's length."],
     ["One qubit per feature", "768 qubits: impossible to simulate or run."]],
    "projection and 8 qubits from the base paper; tanh bound ours")

component(
    "7.3 Angle encoding with data re-uploading",
    "Each layer applies RY(x) with the data angles, then trainable RZ(θ) and RY(θ) rotations, then a ring of "
    "CNOT gates; two layers are followed by a final RY(x). The data are encoded in every layer, not just once.",
    ["**The key reason:** fidelity is unchanged by any operation applied after the last encoding, because "
     "|⟨ψ(x)|V†V|ψ(x′)⟩|² = |⟨ψ(x)|ψ(x′)⟩|² for any unitary V. In the base paper's design (encode once, then "
     "trainable gates), the trainable gates cancel out completely: they cannot affect any prediction and "
     "receive no gradient.",
     "Re-uploading the data between trainable layers places the trainable gates between encodings, so they do "
     "change which images look similar. The circuit then genuinely learns a similarity measure "
     "[Pérez-Salinas et al., 2020].",
     "Repeating the encoding also makes the model more expressive: with L encodings the state depends on "
     "higher frequencies of the input angles [Schuld et al., 2021].",
     "Ending on an encoding layer avoids wasted gates that would cancel by the same argument."],
    [["Encode once, trainable gates after (base paper)", "Trainable gates cancel in the fidelity; the "
      "circuit adds nothing learnable beyond the projection."],
     ["Fixed, untrained encoding circuit", "Valid as a fixed quantum kernel, but nothing in the circuit "
      "adapts to the data."],
     ["Deep circuits with many layers", "More parameters and simulation cost; risk of barren plateaus."]],
    "ours")

component(
    "7.4 CNOT ring entanglement",
    "After each layer, CNOT gates connect qubit 0→1, 1→2, …, 7→0.",
    ["Without entanglement the state is a product of single-qubit states, and fidelity factorises into a "
     "product of per-feature similarities: the classifier cannot capture interactions between features.",
     "A ring entangles every qubit with its neighbours using only 8 two-qubit gates per layer, and over two "
     "layers information spreads across the whole register.",
     "Nearest-neighbour connectivity maps directly onto real quantum chips."],
    [["No entanglement", "Fidelity reduces to independent per-feature comparisons."],
     ["All-to-all CNOTs", "28 two-qubit gates per layer; deeper and noisier on hardware for little gain."],
     ["Controlled-RZ ring (base paper)", "Also entangling; CNOT was kept fixed so that all trainable "
      "parameters are single-qubit rotations."]],
    "ring from the base paper; CNOT gates ours")

component(
    "7.5 Class score = average fidelity to the training images of that class",
    "For a test image, the score of class c is the mean fidelity between its state and every training state "
    "of class c.",
    ["Averaging over the whole class is robust: a single atypical or mislabelled training image cannot decide "
     "the outcome, unlike nearest-neighbour rules.",
     "It is equivalent to comparing with the class's mean quantum state (a density matrix), the "
     "\"fidelity classifier\" of Lloyd et al. [2020].",
     "It needs no fitted support vectors; the reference states can be computed once and reused, so "
     "classifying a new image is fast."],
    [["Nearest neighbour (highest single fidelity)", "Sensitive to outliers and label noise."],
     ["k-nearest neighbours", "Adds a tuning parameter; still relies on a few training images."],
     ["Quantum SVM", "More expressive but needs a quadratic optimisation over the full kernel matrix."]],
    "base paper")

component(
    "7.6 Softmax with a learnable temperature",
    "Probabilities are p_c = softmax(T · S_c), where S_c is the class score and T is learned, starting at 10.",
    ["Fidelities lie between 0 and 1. With three classes, a plain softmax can give the winning class at most "
     "e¹ / (e¹ + 2e⁰) ≈ 57.6%, even for a perfect match. The loss can never approach zero, training stalls, "
     "and the output cannot be read as a confidence.",
     "Multiplying by a temperature removes that cap. Learning T lets the model choose how sharp its "
     "probabilities should be; starting at 10 already allows confident predictions from the first epoch."],
    [["Plain softmax of the scores (base paper)", "Confidence capped at 57.6%; weak training signal."],
     ["Fixed temperature", "Works if well chosen, but adds a value to tune by hand."]],
    "ours")

component(
    "7.7 Same-source exclusion during training",
    "When a training image is scored against the training set, its own state and those of its augmented "
    "copies are left out.",
    ["Every image has fidelity 1 with itself and nearly 1 with its own rotated copy. Including them would "
     "make every training image trivially correct, so the loss would teach the circuit nothing.",
     "Excluding them reproduces the test situation, where the image being classified is never among the "
     "references."],
    [["Score against all training images", "Self-matches dominate; training signal collapses."]],
    "ours")

component(
    "7.8 Exact statevector simulation in PyTorch",
    "The 256-amplitude state is computed exactly on the GPU, with gradients from PyTorch's autograd; "
    "correctness was checked against Qiskit.",
    ["Exact simulation gives noise-free fidelities, so results measure the method itself, not hardware noise "
     "or measurement sampling.",
     "Writing the simulator in PyTorch lets gradients flow through the circuit in the same framework as the "
     "network and runs batched on the GPU, which is much faster than calling a separate quantum SDK per "
     "image.",
     "At 8 qubits exact simulation is cheap; this would not hold at 30+ qubits."],
    [["Qiskit / PennyLane simulators", "Correct, but slower for batched training and harder to "
      "integrate with the PyTorch network."],
     ["Shot-based simulation", "Each fidelity estimated from thousands of measurements; adds sampling noise "
      "and cost."],
     ["Real quantum hardware", "Queue times, gate noise and shot noise; appropriate for future work, not for "
      "validating the method."]],
    "ours")

# =====================================================================  8. EVALUATION
doc.add_heading("8. Evaluation", 1)
component(
    "8.1 Metrics: accuracy, macro precision / recall / F1, MCC, AUC, per-class recall",
    "All metrics are computed once, on the untouched test set.",
    ["With 10% benign cases, accuracy can stay high while most benign cases are missed. Macro averages weight "
     "each class equally.",
     "Matthews correlation coefficient uses all four cells of the confusion matrix and stays reliable under "
     "imbalance [Chicco and Jurman, 2020].",
     "AUC measures ranking quality independent of the decision threshold.",
     "Per-class recall shows exactly which class fails, which is where models actually differ."],
    [["Accuracy alone", "Hides failures on the minority class."],
     ["Micro-averaged metrics", "Dominated by the majority class, like accuracy."]],
    "metrics from the base paper; per-class recall ours")

component(
    "8.2 Repeated seeds, mean ± standard deviation",
    "Stochastic stages are run with several random seeds and reported as mean ± standard deviation.",
    ["With 211 test images, one image is about 0.5% accuracy. A single run cannot separate a real difference of "
     "that size from chance [Bouthillier et al., 2021].",
     "The spread over seeds shows how much of any difference is luck."],
    [["Single run", "Differences below about 1% are not interpretable."],
     ["k-fold cross-validation", "Stronger, but multiplies training cost; see 4.2."]],
    "ours")

component(
    "8.3 Baselines trained under the same protocol, plus classical controls",
    "ResNet50 is retrained with the identical data, split and budget; linear and MLP classifiers are trained on "
    "the same frozen features as the quantum classifier.",
    ["A comparison is fair only if every model sees the same data, split and training budget. Numbers taken "
     "from other studies (different datasets, splits or metrics) cannot support a claim of superiority.",
     "Classical classifiers on the same features isolate the classifier's contribution: if they match the "
     "quantum classifier, the advantage lies in the features, not the circuit."],
    [["Quoting baselines from the literature (base paper)", "Not comparable: different data, splits and "
      "metrics."]],
    "ResNet50 from the base paper's comparison; protocol and controls ours")

# =====================================================================  SUMMARY
doc.add_heading("Summary", 1)
table(["Stage", "Choice", "Main reason", "Origin"],
      [["Data", "De-duplicate, split then augment", "No leakage: every test image is unseen", "ours"],
       ["Data", "Benign ×4 by augmentation", "Balances the 10% class with new views", "ours"],
       ["Pre-processing", "NLM denoising", "Removes Gaussian-like CT noise, keeps edges", "paper"],
       ["Pre-processing", "Noise-adaptive NLM strength", "Matches filtering to each slice's noise", "ours"],
       ["Pre-processing", "CLAHE", "Local contrast without large noise amplification", "paper"],
       ["Pre-processing", "Adaptive clip limit", "Enhancement scaled to each slice's contrast", "paper / ours"],
       ["Pre-processing", "Pad + area resize to 224", "No distortion, no aliasing, pretrained scale", "ours"],
       ["Pre-processing", "z-score, 3 channels", "Stable scale; reuse pretrained first layer", "paper / ours"],
       ["Pre-processing", "Mild geometric augmentation", "Imitates real positioning variation", "paper"],
       ["Features", "ImageNet transfer learning", "Too little data to train from scratch", "paper"],
       ["Features", "ConvNeXt-Tiny", "Modern CNN, LayerNorm, fits 4 GB", "paper / ours"],
       ["Features", "ECA attention", "Channel weighting with 3 parameters", "paper"],
       ["Features", "AdaBelief, lr 1e-4", "Stable adaptive steps for fine-tuning", "paper"],
       ["Features", "Mixed precision + accumulation", "Batch 32 in 4 GB, exact with LayerNorm", "ours"],
       ["Features", "Two-stage training", "Faster, stable, contributions measurable", "ours"],
       ["Quantum", "Fidelity classifier", "Quantum-kernel similarity; the research question", "paper"],
       ["Quantum", "Learned projection, π·tanh", "Discriminative angles that cannot wrap around", "paper / ours"],
       ["Quantum", "Data re-uploading", "Trainable gates otherwise cancel in fidelity", "ours"],
       ["Quantum", "CNOT ring", "Feature interactions with few gates", "paper / ours"],
       ["Quantum", "Class-average fidelity", "Robust prototype rule, no optimisation", "paper"],
       ["Quantum", "Learnable temperature", "Removes the 57.6% confidence cap", "ours"],
       ["Quantum", "Same-source exclusion", "Prevents trivial self-matches in training", "ours"],
       ["Quantum", "Exact PyTorch simulation", "Noise-free, differentiable, fast", "ours"],
       ["Evaluation", "Macro metrics, MCC, per-class recall", "Imbalance-aware", "paper / ours"],
       ["Evaluation", "Seeds, identical-protocol baselines", "Fair, interpretable comparisons", "ours"]],
      widths=[1.0, 1.9, 2.8, 0.8], size=9)

doc.add_heading("9. Practical evidence", 1)
para("To be appended: measured results for each component (denoising and contrast benchmarks, feature-extractor "
     "and classifier ablations, memory and speed benchmarks). The experiments are in CNN/Rationale/.")

doc.add_heading("References", 1)
for ref in [
    "Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. MLSys.",
    "Buades, A., Coll, B., and Morel, J.-M. (2005). A non-local algorithm for image denoising. CVPR.",
    "Chicco, D., and Jurman, G. (2020). The advantages of the Matthews correlation coefficient over F1 score "
    "and accuracy in binary classification evaluation. BMC Genomics, 21.",
    "Diwakar, M., and Kumar, M. (2018). A review on CT image noise and its denoising. Biomedical Signal "
    "Processing and Control, 42.",
    "Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at "
    "scale. ICLR.",
    "Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2.",
    "Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature, 567.",
    "Hu, J., Shen, L., and Sun, G. (2018). Squeeze-and-excitation networks. CVPR.",
    "Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. (2012). Leakage in data mining. ACM TKDD, 6(4).",
    "Kornblith, S., Shlens, J., and Le, Q. V. (2019). Do better ImageNet models transfer better? CVPR.",
    "Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis, "
    "42.",
    "Liu, Z., et al. (2022). A ConvNet for the 2020s. CVPR.",
    "Lloyd, S., Schuld, M., Ijaz, A., Izaac, J., and Killoran, N. (2020). Quantum embeddings for machine "
    "learning. arXiv:2001.03622.",
    "Micikevicius, P., et al. (2018). Mixed precision training. ICLR.",
    "Pérez-Salinas, A., et al. (2020). Data re-uploading for a universal quantum classifier. Quantum, 4, 226.",
    "Pizer, S. M., et al. (1987). Adaptive histogram equalization and its variations. Computer Vision, "
    "Graphics, and Image Processing, 39(3).",
    "Schuld, M., and Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. Physical "
    "Review Letters, 122.",
    "Schuld, M., Sweke, R., and Meyer, J. J. (2021). Effect of data encoding on the expressive power of "
    "variational quantum machine-learning models. Physical Review A, 103.",
    "Shorten, C., and Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. "
    "Journal of Big Data, 6.",
    "Wang, Q., et al. (2020). ECA-Net: Efficient channel attention for deep convolutional neural networks. "
    "CVPR.",
    "Woo, S., Park, J., Lee, J.-Y., and Kweon, I. S. (2018). CBAM: Convolutional block attention module. ECCV.",
    "Zhuang, J., et al. (2020). AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients. "
    "NeurIPS.",
    "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics Gems IV, Academic Press.",
]:
    doc.add_paragraph(ref, style="List Bullet")

REPORT.parent.mkdir(exist_ok=True)
doc.save(REPORT)
print("Saved", REPORT)
