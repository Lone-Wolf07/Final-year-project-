# Running the pipeline on another computer

Target: Windows with an NVIDIA GPU (driver 580 or newer for the CUDA 13.0 build of PyTorch), Python 3.12, Git.
The repo is public, so no GitHub login is needed.

## 1. Get the code

```powershell
git clone -b guna https://github.com/Lone-Wolf07/Final-year-project-.git Final_Year_Project
cd Final_Year_Project
```

## 2. Create the environment (run this first)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r CNN\requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line must print `True`. If it prints `False`, update the NVIDIA driver, or replace `cu130` in
`CNN\requirements.txt` with the CUDA build your driver supports (see pytorch.org) and reinstall.

## 3. Get the data from Kaggle

Dataset: https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset

Credentials: nothing is stored in the repo. Each user signs in with their own Kaggle account.

1. Sign in at kaggle.com, open Settings > API and create a token (a new account may need phone verification first).
2. Give the token to the Kaggle package in ONE of these ways:
   - Environment variable, for the current PowerShell window only:
     `$env:KAGGLE_API_TOKEN = "<your token>"`
   - Token file: save the token as `%USERPROFILE%\.kaggleccess_token`.
   - Older style: save the downloaded `kaggle.json` as `%USERPROFILE%\.kaggle\kaggle.json`.
3. Open the dataset page above once in the browser and accept its terms if Kaggle asks.
4. Download and split:

```powershell
python Datasets\download_iqothnccd.py
```

No Kaggle account? Download the zip in the browser and run
`python Datasets\download_iqothnccd.py --zip path	orchive.zip` (no credentials needed).

Never commit the token or `kaggle.json`.

The script puts the three class folders in `Datasets\IQ-OTH_NCCD\Data\` and runs `create_splits.py`, which removes
byte-identical duplicates and rewrites `Datasets\IQ-OTH_NCCD\DataSplit\` (same split every time, seed 42).
Expected: 1,054 images, 790 train and 264 test.

## 4. Run the main pipeline

Settings are at the top of `CNN\run_pipeline.py`. The reported results use 50 epochs for both stages, and the
default is 10, so set `CNN_EPOCHS = 50` and `QUANTUM_EPOCHS = 50` for a full reproduction. Set `TEST_SIZE = 0.3`
for the 70:30 split. Lower `CNN_MICRO_BATCH` if the GPU runs out of memory.

```powershell
cd CNN
python run_pipeline.py          # preprocessing, ConvNeXt-ECA, quantum classifier, evaluation
python evaluate.py              # metrics and timing for a saved model
python predict.py "path\to\scan.jpg"
```

Outputs go to `CNN\Feature_Extraction\Results`, `CNN\Quantum_Classifier\Results` and the matching `Models` folders
(model files are not in git; they are created by this step).

## 5. Optional studies (need the 50-epoch run from step 4 first)

```powershell
cd CNN
python ResNet50\resnet50.py --test-size 0.2        # ResNet50 baseline
python Comparison\compare_2x2.py                   # feature extractor x classifier study
python Rationale\run_all.py                        # ablations (Windows only)
```

## 6. Exact replication of the base paper (CNN_Paper)

`CNN_Paper\paper_qcf.py` is a one-to-one implementation of the paper's design (end-to-end ConvNeXt-Base, augmentation
before or after the split). It reads the raw images from `Datasets\IQ-OTH_NCCD\Data\` (step 3) and takes much longer
than the main pipeline. Run it from the `CNN_Paper` folder:

```powershell
cd CNN_Paper
$env:PYTHONIOENCODING = "utf-8"
python -u paper_qcf.py --test-size 0.2 --mode before    # augmentation before the split (as in the paper)
python -u paper_qcf.py --test-size 0.2 --mode after     # leakage-free
python -u paper_qcf.py --test-size 0.3 --mode before    # 70:30 split
```

Results are written to `CNN_Paper\Results`. The trained weights (about 1 GB) and the cache are not in git.

## Notes

- Runs are one process at a time; keep the machine awake during long runs.
- Python versions and package pins are in `CNN\requirements.txt`; use Python 3.12 to match.
