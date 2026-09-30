# Python Requirements

Libraries imported across `Python_Exercises/` (tkinter, os, re, json, time, functools, random are stdlib — no install needed).

```
opencv-python
numpy
pytesseract
PyWavelets
ultralytics
face_recognition
pandas
matplotlib
seaborn
cryptography
requests
beautifulsoup4
```

## Notes by exercise

| File | Library | Extra setup |
|---|---|---|
| Exercises_2/Q9.py | `requests`, `beautifulsoup4` | — |
| Exercises_2/Q10.py | `opencv-python` | — |
| Exercises_3/Q1-Q3.py | `opencv-python`, `numpy` | — |
| Exercises_3/Q4.py, Q8.py | `pandas`, `matplotlib`, `seaborn` | — |
| Exercises_3/Q6.py | `cryptography` | — |
| Exercises_4/Q1.py | `ultralytics` | Pulls in `torch`/`torchvision`; downloads `yolov8n.pt` weights on first run |
| Exercises_4/Q2.py | `opencv-python`, `PyWavelets`, `numpy` | — |
| Exercises_4/Q3-Q4, Q6-Q7 | `opencv-python`, `numpy` | — |
| Exercises_4/Q5.py | `pytesseract`, `opencv-python` | Needs the Tesseract-OCR binary installed separately (e.g. `winget install UB-Mannheim.TesseractOCR`) and on PATH |
| Exercises_4/Q8.py | `face_recognition`, `opencv-python` | Needs `dlib`; on Windows install `cmake` first, then `dlib`, then `face_recognition` |

`DCN_Lab/Assignment2_Main/Code/stop_and_wait_arq.py` only uses stdlib (`os`, `random`, `time`) — no extra requirements.

## Install

```
pip install -r requirements.md
```
(pip accepts any filename with pip-style lines; rename to `requirements.txt` if you'd rather use the conventional name.)
