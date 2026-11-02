"""Validation-driven training. Test split is NOT evaluated by this training module.

Final test evaluation runs exactly in `python -m src.finalize` after the model
and hyperparameters have been selected on training/validation only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from .data import CLASS_NAMES, make_manifest, loaders, seed_everything
from .models import build_model, count_params


def evaluate(model, loader, device, loss_fn):
    """No assumptions about evaluation split; the caller controls access."""
    model.eval()
    targets, probs, total_loss, seen = [], [], 0.0, 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * yb.shape[0]
            seen += yb.shape[0]
            targets.append(yb.cpu().numpy())
            probs.append(logits.softmax(1).cpu().numpy())
    if seen == 0:
        raise ValueError('Evaluation loader is empty')
    y, prob = np.concatenate(targets), np.concatenate(probs)
    predictions = prob.argmax(axis=1)
    pr, re, f1, _ = precision_recall_fscore_support(
        y, predictions, average='macro', zero_division=0, labels=range(len(CLASS_NAMES))
    )
    metrics = {'loss': total_loss / seen, 'accuracy': float(accuracy_score(y, predictions)),
               'macro_precision': float(pr), 'macro_recall': float(re), 'macro_f1': float(f1)}
    return metrics, prob, y


def plot_curves(history, out):
    plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'legend.fontsize':12,'xtick.labelsize':11,'ytick.labelsize':11})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(range(1,len(history['train_loss'])+1), history['train_loss'], label='Training')
    axes[0].plot(range(1,len(history['val_loss'])+1), history['val_loss'], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Train vs validation loss')
    axes[0].legend()
    axes[1].plot(range(1,len(history['train_accuracy'])+1), history['train_accuracy'], label='Train accuracy')
    axes[1].plot(range(1,len(history['val_accuracy'])+1), history['val_accuracy'], label='Validation accuracy')
    axes[1].plot(range(1,len(history['val_macro_f1'])+1), history['val_macro_f1'], ls='--', label='Validation macro-F1')
    axes[1].set(xlabel='Epoch', ylabel='Score', title='Train vs validation accuracy', ylim=(0,1.01))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(out) / 'learning_curves.png', dpi=165)
    plt.close(fig)


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError('epochs, batch-size, patience must be positive')
    if args.learning_rate <= 0 or args.image_size < 64 or args.sample_limit < 0 or args.workers < 0:
        raise ValueError('learning-rate, image-size, sample-limit or workers outside supported range')
    if args.model == 'baseline' and args.freeze_backbone:
        raise ValueError('Cannot freeze backbone of baseline CNN')
    seed_everything(args.seed)
    start = time.monotonic()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    root = Path(args.data_root)
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise FileExistsError('This run has final test results. Create a new experiment directory rather than overwriting it.')
    manifest_path = Path(args.out) / 'split_manifest.json'
    manifest = make_manifest(root, manifest_path, seed=args.seed)
    if manifest['seed'] != args.seed:
        raise ValueError('Existing split manifest does not match requested seed')
    dls = loaders(root, manifest, args.image_size, args.batch_size,
                  args.workers, args.sample_limit, args.seed)
    # CRITICAL: The 'test' loader is intentionally never called in this file.
    model = build_model(args.model, 10, not args.no_pretrained and not args.resume, args.freeze_backbone).to(device)
    resumed_from = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        if ck['model'] != args.model or ck['class_names'] != CLASS_NAMES:
            raise ValueError('Resume checkpoint model or label mismatch')
        if ck.get('image_size') != args.image_size or ck.get('seed') != args.seed:
            raise ValueError('Resume requires same preprocessing size and data split seed')
        if ck.get('split_manifest_sha256') != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('Cannot resume with a different training/validation/testing manifest')
        if ck.get('experiment') != ('SMOKE_TEST_NOT_FINAL' if args.sample_limit else 'FULL_DATA'):
            raise ValueError('Resume cannot mix smoke-test and full-data checkpoints')
        if ck.get('pretrained') != (args.model == 'efficientnet' and not args.no_pretrained):
            raise ValueError('Checkpoint pretrained-state mismatch')
        model.load_state_dict(ck['state_dict'])
        resumed_from = str(args.resume)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [],
               'val_accuracy': [], 'val_macro_f1': []}
    best_score, best_epoch, stale = -1., -1, 0
    meta = {'model': args.model, 'class_names': CLASS_NAMES, 'image_size': args.image_size,
            'seed': args.seed, 'pretrained': args.model == 'efficientnet' and not args.no_pretrained,
            'experiment': 'SMOKE_TEST_NOT_FINAL' if args.sample_limit else 'FULL_DATA',
            'split_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    print(json.dumps({'device': str(device), 'model': args.model,
                      'parameters': count_params(model),
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'train': len(dls['train'].dataset), 'val': len(dls['val'].dataset),
                      'test_reserved_untouched': len(dls['test'].dataset),
                      'sample_limit': args.sample_limit, 'pretrained': args.model == 'efficientnet' and not args.no_pretrained}, indent=2), flush=True)
    # Keep original validation best when fine-tuning from feature-extraction stage.
    if resumed_from:
        initial_m, _, _ = evaluate(model, dls['val'], device, loss_fn)
        best_score = initial_m['macro_f1']
        best_epoch = 0
        torch.save({**meta, 'state_dict': model.state_dict(),
                    'validation_macro_f1': best_score, 'best_epoch': best_epoch,
                    'resumed_from': resumed_from}, out / 'best.pt')
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone and args.model == 'efficientnet':
            model.features.eval()  # freeze BN running statistics too
        seen, total_loss, train_correct = 0, 0.0, 0
        for xb, yb in tqdm(dls['train'], desc=f'{args.model} epoch {epoch}/{args.epochs}', leave=False, disable=not __import__('sys').stderr.isatty()):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        val, _, _ = evaluate(model, dls['val'], device, loss_fn)
        history['train_loss'].append(total_loss / seen)
        history['train_accuracy'].append(train_correct / seen)
        history['val_loss'].append(val['loss'])
        history['val_accuracy'].append(val['accuracy'])
        history['val_macro_f1'].append(val['macro_f1'])
        print(f'Epoch {epoch}: train_acc={train_correct / seen:.4f} '
              f'val_acc={val["accuracy"]:.4f} val_f1={val["macro_f1"]:.4f}', flush=True)
        if val['macro_f1'] > best_score + 1e-5:
            best_score, best_epoch, stale = val['macro_f1'], epoch, 0
            torch.save({**meta, 'state_dict': model.state_dict(),
                        'validation_macro_f1': best_score, 'best_epoch': best_epoch,
                        'resumed_from': resumed_from}, out / 'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                break
    best_ck = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_ck['state_dict'])
    val_m, val_probs, val_y = evaluate(model, dls['val'], device, loss_fn)
    np.savez_compressed(out / 'validation_probabilities.npz', val_probs=val_probs, val_y=val_y)
    results = {'model': args.model, 'status': 'VALIDATION_ONLY_NOT_FINAL',
               'full_dataset': args.sample_limit == 0,
               'training_config': vars(args),
               'train_samples': len(dls['train'].dataset),
               'val_samples': len(dls['val'].dataset),
               'test_samples_reserved': len(dls['test'].dataset),
               'checkpoint_sha256': hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest(),
               'params': count_params(model),
               'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'best_epoch': best_epoch, 'validation_metrics': val_m,
               'history': history,
               'split_manifest_sha256': meta['split_manifest_sha256'],
               'elapsed_seconds': round(time.monotonic() - start, 2),
               'environment': {'python': platform.python_version(), 'pytorch': torch.__version__,
                               'torchvision': torchvision.__version__,
                               'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                               'libraries': {k: version(k) for k in ('numpy','scikit-learn','matplotlib','Pillow','gradio','tqdm') }}}
    (out / 'validation_results.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf8')
    plot_curves(history, out)
    print('VALIDATION ONLY — NO TEST METRICS PRODUCED:', out / 'validation_results.json', flush=True)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['baseline','efficientnet'], default='baseline')
    p.add_argument('--data-root', default='data/EuroSAT_RGB')
    p.add_argument('--out', default='artifacts/eksperimen')
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=0.0003)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--sample-limit', type=int, default=0, help='SMOKE ONLY; never use to report scientific metrics')
    p.add_argument('--no-pretrained', action='store_true', help='Testing only; do not claim transfer learning')
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--resume', default=None)
    return p.parse_args()

if __name__ == '__main__':
    run(parse_args())
