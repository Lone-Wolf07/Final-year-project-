import ctypes
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Keep the system and display awake while this process runs.
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
for args in (["feature_ablation.py", "quick"], ["classifier_ablation.py"], ["feature_ablation.py", "retrain"]):
    print("==>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", "-W", "ignore", str(HERE / args[0]), *args[1:]], cwd=HERE, check=False)
print("==> all done", flush=True)
