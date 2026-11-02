"""One protocol for notebook/CLI: train both CNNs, freeze choices, then test once."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import prepare_archive,find_root

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','baseline','head','finetune','finalize'],default='all');a=ap.parse_args()
    cfg=json.loads((ROOT/'experiment_config.json').read_text())
    import os
    os.chdir(ROOT)
    if not (ROOT/'data/EuroSAT_RGB').is_dir():prepare_archive(ROOT/'data/EuroSAT_RGB.zip',ROOT/'data')
    data=str(find_root(ROOT/'data'))
    common=['--data-root',data,'--image-size',str(cfg['image_size']),'--batch-size',str(cfg['batch_size']),
            '--workers',str(cfg['workers']),'--seed',str(cfg['seed'])]
    stages=[('baseline','baseline','baseline','artifacts/eksperimen',[]),
            ('head','efficientnet','feature_extraction','artifacts/stage1',['--freeze-backbone']),
            ('finetune','efficientnet','fine_tuning','artifacts/eksperimen',['--resume','artifacts/stage1/efficientnet/best.pt'])]
    for stage,model,config_name,out,extra in stages:
        if a.stage not in ['all',stage]:continue
        if (ROOT/out/model/'validation_results.json').exists():
            import hashlib
            old=json.loads((ROOT/out/model/'validation_results.json').read_text())
            wanted={**{k:cfg[k] for k in ('image_size','batch_size','seed','workers')},**cfg[config_name]}
            if any(old['training_config'].get(k)!=v for k,v in wanted.items()):
                raise ValueError('Saved run configuration differs. Use a new experiment directory, not the old test split.')
            if old.get('checkpoint_sha256')!=hashlib.sha256((ROOT/out/model/'best.pt').read_bytes()).hexdigest():
                raise ValueError('Saved run checkpoint hash differs')
            print('Use existing validated training output:',stage,flush=True);continue
        v=cfg[config_name]
        subprocess.run([sys.executable,'-m','src.train','--model',model,'--out',out,
                        '--epochs',str(v['epochs']),'--patience',str(v['patience']),
                        '--learning-rate',str(v['learning_rate']),*common,*extra],check=True)
    if a.stage in ['all','finalize']:
        from src.finalize import finalize
        from src.postprocess import run
        finalize(ROOT,data,cfg['batch_size'],cfg['workers'])
        run(ROOT)

if __name__=='__main__':main()
