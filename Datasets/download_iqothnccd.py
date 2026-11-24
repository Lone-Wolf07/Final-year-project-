"""Download IQ-OTH/NCCD from Kaggle into Datasets/IQ-OTH_NCCD/Data/ and then split it.

Usage:
    python download_iqothnccd.py                 # download with the Kaggle API, then run create_splits.py
    python download_iqothnccd.py --zip file.zip  # use a zip you downloaded by hand
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "IQ-OTH_NCCD" / "Data"
SLUG = "hamdallak/the-iqothnccd-lung-cancer-dataset"
CLASS_FOLDERS = ("Bengin cases", "Malignant cases", "Normal cases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", help="path to a manually downloaded Kaggle zip")
    args = parser.parse_args()

    tmp = HERE / "_kaggle_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    if args.zip:
        zip_path = Path(args.zip)
    else:
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", SLUG, "-p", str(tmp)],
                       check=True)
        zip_path = next(tmp.glob("*.zip"))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp / "extracted")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in CLASS_FOLDERS:
        found = next((p for p in (tmp / "extracted").rglob(name) if p.is_dir()), None)
        if found is None:
            sys.exit(f"Could not find '{name}' in the download")
        shutil.rmtree(DATA_DIR / name, ignore_errors=True)
        shutil.copytree(found, DATA_DIR / name)
        print(f"{name}: {sum(1 for _ in (DATA_DIR / name).iterdir())} files")
    shutil.rmtree(tmp, ignore_errors=True)

    subprocess.run([sys.executable, str(HERE / "create_splits.py"), "IQ-OTH_NCCD"], check=True)


if __name__ == "__main__":
    main()
