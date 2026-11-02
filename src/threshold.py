"""Calibrate on validation BEFORE any test metrics; evaluate an immutable threshold."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .interpret import select_threshold, selective_metrics


def calibrate(results_dir, minimum_coverage=.70):
    folder=Path(results_dir)
    target=folder/'threshold_calibration.json'
    checkpoint_hash=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()
    if target.exists():
        cal=json.loads(target.read_text())
        if cal['checkpoint_sha256']!=checkpoint_hash or cal['minimum_coverage']!=minimum_coverage:
            raise ValueError('Saved calibration does not match the checkpoint or coverage requirement')
        return cal
    if (folder/'results.json').exists():
        raise RuntimeError('Calibration must be fixed before reading test results')
    result=json.loads((folder/'validation_results.json').read_text())
    if not result['full_dataset'] or not result['training_config'].get('sample_limit')==0:
        raise ValueError('Calibration requires the full training and validation splits')
    if result['checkpoint_sha256']!=checkpoint_hash:
        raise ValueError('Validation results refer to another checkpoint')
    with np.load(folder/'validation_probabilities.npz') as p:
        cal=select_threshold(p['val_probs'],p['val_y'],minimum_coverage)
    cal.update(checkpoint_sha256=checkpoint_hash,selection_split='validation',
               split_manifest_sha256=result['split_manifest_sha256'],selected_before_test=True)
    target.write_text(json.dumps(cal,indent=2),encoding='utf8')
    return cal


def run(results_dir, minimum_coverage=.70):
    folder=Path(results_dir)
    cal=calibrate(folder,minimum_coverage)
    result=json.loads((folder/'results.json').read_text())
    if result.get('status')!='FINAL_HOLDOUT_EVALUATED' or not result['full_dataset']:
        raise RuntimeError('Test analysis requires final full-data evaluation')
    if result['checkpoint_sha256']!=cal['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint differs from calibrated checkpoint')
    with np.load(folder/'probabilities.npz') as p:
        report={**cal,'test_metrics':selective_metrics(p['test_probs'],p['test_y'],cal['threshold']),
                'baseline_without_rejection':selective_metrics(p['test_probs'],p['test_y'],0.)}
    report['note']='Threshold fixed on validation before test; coverage >=70% applies to validation only.'
    (folder/'threshold.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results-dir',default='artifacts/eksperimen/efficientnet')
    ap.add_argument('--minimum-coverage',type=float,default=.70);a=ap.parse_args()
    print(json.dumps(run(a.results_dir,a.minimum_coverage),indent=2))
