"""FINAL HOLDOUT evaluation: invoke only once after ALL tuning decisions are frozen.

Final test access is isolated from training, including feature-extraction stage.
Refuses development-only models, incompatible splits, untrained checkpoints,
or non-ImageNet EfficientNet experiments presented as transfer learning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
from .data import CLASS_NAMES, make_manifest, loaders, seed_everything
from .models import build_model
from .train import evaluate


def finalize(project='.', data_root='data/EuroSAT_RGB', batch_size=64, workers=2, cpu=False):
    project = Path(project).resolve()
    dataset_root = Path(data_root)
    if not dataset_root.is_absolute():
        dataset_root = project / dataset_root
    out = project / 'artifacts/eksperimen'
    manifest_path = out / 'split_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Finalization needs shared experiment split_manifest.json')
    manifest = make_manifest(dataset_root, manifest_path)
    sizes = [len(manifest[s]) for s in ('train', 'val', 'test')]
    if sizes != [18900, 4050, 4050] or manifest['class_names'] != CLASS_NAMES:
        raise ValueError('Expected original EuroSAT full-data stratified 70/15/15 split')
    shared_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    seed_everything(manifest['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    # Preflight BOTH models and freeze the validation threshold before opening holdout.
    for name in ('baseline','efficientnet'):
        folder=out/name
        val=json.loads((folder/'validation_results.json').read_text())
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        digest=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()
        if not val['full_dataset'] or ck.get('experiment')!='FULL_DATA' or val['checkpoint_sha256']!=digest or ck['split_manifest_sha256']!=shared_hash:
            raise ValueError(f'{name}: missing or incompatible full-data evidence')
        if name=='efficientnet' and not ck.get('pretrained'):
            raise ValueError('EfficientNet requires ImageNet pretraining')
    from .threshold import calibrate
    calibrate(out/'efficientnet',.70)
    for name in ('baseline','efficientnet'):
        folder = out / name
        if (folder / 'results.json').exists():
            prior=json.loads((folder/'results.json').read_text())
            digest=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()
            if prior.get('checkpoint_sha256')!=digest or prior['split_manifest_sha256']!=shared_hash:
                raise ValueError('Existing test results have different checkpoint or split')
            print(f'{name}: using saved final test results; no reevaluation',flush=True)
            continue
        val_path, checkpoint_path = folder / 'validation_results.json', folder / 'best.pt'
        val_results = json.loads(val_path.read_text(encoding='utf8'))
        ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (not val_results['full_dataset'] or val_results['status']!='VALIDATION_ONLY_NOT_FINAL'
            or ck['experiment']!='FULL_DATA' or ck['model']!=name
            or ck['class_names']!=CLASS_NAMES or ck['seed']!=manifest['seed']
            or val_results['split_manifest_sha256']!=shared_hash
            or ck['split_manifest_sha256']!=shared_hash):
            raise ValueError(f'{name}: checkpoint, manifest or full-data provenance mismatch')
        if name == 'efficientnet' and not ck['pretrained']:
            raise ValueError('EfficientNet was trained without pretrained weights; not valid transfer-learning evidence')
        if val_results['train_samples']!=18900 or val_results['val_samples']!=4050:
            raise ValueError('Counts do not match full-data experiment')
        model = build_model(name, pretrained=False).to(device)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        dl = loaders(dataset_root, manifest, image_size=ck['image_size'],
                     batch_size=batch_size, workers=workers, sample_limit=0)
        loss_fn = nn.CrossEntropyLoss()
        # Validation scores are recomputed from saved selected checkpoint.
        val_m, val_probs, val_y = evaluate(model, dl['val'], device, loss_fn)
        test_m, test_probs, test_y = evaluate(model, dl['test'], device, loss_fn)
        predicted = test_probs.argmax(axis=1)
        report = classification_report(test_y, predicted, labels=range(10),
                                       target_names=CLASS_NAMES, zero_division=0, output_dict=True)
        cm = confusion_matrix(test_y, predicted, labels=range(10))
        result = {**val_results, 'status': 'FINAL_HOLDOUT_EVALUATED',
                  'val_samples': len(val_y), 'test_samples': len(test_y),
                  'validation_metrics': val_m, 'test_metrics': test_m,
                  'classification_report': report, 'confusion_matrix': cm.tolist(),
                  'final_selection_rule': 'maximum validation macro-F1; test reserved for final report only',
                  'checkpoint_best_epoch': ck['best_epoch'],
                  'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
        # Write all outputs AFTER successful evaluation of the selected checkpoint.
        np.savez_compressed(folder / 'probabilities.npz', val_probs=val_probs,
                            val_y=val_y, test_probs=test_probs, test_y=test_y)
        fig, ax = plt.subplots(figsize=(9,7))
        im=ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(10), CLASS_NAMES, rotation=60, ha='right', fontsize=11)
        ax.set_yticks(range(10), CLASS_NAMES, fontsize=11)
        ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f'Confusion matrix: {name} (test holdout)')
        for i in range(10):
            for j in range(10):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color='white' if cm[i,j]>cm.max()/2 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.045)
        fig.tight_layout();fig.savefig(folder / 'confusion_matrix.png', dpi=170);plt.close(fig)
        (folder / 'results.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf8')
        print(f'FINAL TEST: {name}, N={len(test_y)}, accuracy={test_m["accuracy"]:.4f}, macro-F1={test_m["macro_f1"]:.4f}', flush=True)
    return True

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--project', default='.')
    ap.add_argument('--data-root', default='data/EuroSAT_RGB')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args()
    finalize(args.project, args.data_root, args.batch_size, args.workers, args.cpu)
