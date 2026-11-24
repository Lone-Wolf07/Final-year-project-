$env:PYTHONIOENCODING = 'utf-8'
$py = "..\.venv\Scripts\python.exe"
& $py -u paper_qcf.py --test-size 0.2 --mode before *> Results\run_80_20_before.log
& $py -u paper_qcf.py --test-size 0.2 --mode after *> Results\run_80_20_after.log
& $py -u paper_qcf.py --test-size 0.3 --mode before *> Results\run_70_30_before.log
