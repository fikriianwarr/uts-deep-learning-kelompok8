"""Make deployable HF Space sources only after full-data results are available."""
from pathlib import Path
from shutil import copy2,copytree,rmtree
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    result=ROOT/'artifacts/eksperimen/efficientnet/results.json'
    ck=ROOT/'artifacts/eksperimen/efficientnet/best.pt'
    threshold=ROOT/'artifacts/eksperimen/efficientnet/threshold.json'
    if not all(x.exists() for x in [result,ck,threshold]):
        raise RuntimeError('Model final atau kalibrasi belum ada. Jalankan notebook full dataset terlebih dahulu.')
    record=json.loads(result.read_text())
    if record.get('status') != 'FINAL_HOLDOUT_EVALUATED' or not record.get('full_dataset'):
        raise RuntimeError('Hasil smoke test TIDAK BOLEH digunakan untuk deployment final.')
    ckmeta=__import__('torch').load(ck,map_location='cpu',weights_only=True)
    if ckmeta.get('experiment') != 'FULL_DATA' or not ckmeta.get('pretrained'):
        raise RuntimeError('Checkpoint does not reflect FULL_DATA ImageNet transfer learning')
    dest=ROOT/'deploy/space_package'
    if dest.exists():rmtree(dest)
    dest.mkdir(parents=True)
    copy2(ROOT/'app.py',dest/'app.py')
    copy2(ROOT/'requirements-deploy.txt',dest/'requirements.txt')
    copytree(ROOT/'src',dest/'src',ignore=lambda a,b:{'__pycache__'} if '__pycache__' in b else set())
    (dest/'artifacts/eksperimen/efficientnet').mkdir(parents=True)
    copy2(ck,dest/'artifacts/eksperimen/efficientnet/best.pt')
    copy2(threshold,dest/'artifacts/eksperimen/efficientnet/threshold.json')
    if (ROOT/'examples').is_dir():copytree(ROOT/'examples',dest/'examples')
    if (ROOT/'artifacts/eksperimen/demo_examples.json').is_file():copy2(ROOT/'artifacts/eksperimen/demo_examples.json',dest/'artifacts/eksperimen/demo_examples.json')
    (dest/'README.md').write_text('''---\ntitle: EuroSAT Sentinel-2 Kelompok 8\nemoji: 🛰️\ncolorFrom: blue\ncolorTo: green\nsdk: gradio\nsdk_version: 5.49.1\npython_version: '3.12'\napp_file: app.py\npinned: false\n---\n\n# EuroSAT classification demo\nKelompok 8 - Pembelajaran Mendalam 2026/2027.\n''',encoding='utf8')
    print('SPACE_PACKAGE READY:',dest)
if __name__=='__main__':main()
