"""Aggregate final FULL_DATA model reports, Grad-CAM figures and audit outputs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import CLASS_NAMES,find_root,EuroSATDataset,transform
from .infer import Predictor
from .threshold import run as threshold_run


def run(project='.'):
    root=Path(project)
    base=root/'artifacts/eksperimen/baseline'
    eff=root/'artifacts/eksperimen/efficientnet'
    out=root/'artifacts/eksperimen';out.mkdir(parents=True,exist_ok=True)
    b=json.loads((base/'results.json').read_text());e=json.loads((eff/'results.json').read_text())
    if (b.get('status') != 'FINAL_HOLDOUT_EVALUATED' or e.get('status') != 'FINAL_HOLDOUT_EVALUATED'
        or not b['full_dataset'] or not e['full_dataset']):
        raise RuntimeError('Final comparison REFUSES smoke-test metrics')
    if b['split_manifest_sha256'] != e['split_manifest_sha256']:
        raise ValueError('Incompatible train/val/test splits between models')
    threshold=threshold_run(eff,0.70)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    keys=['accuracy','macro_precision','macro_f1']
    for ax,k in zip(axs,keys):
        ax.bar(['CNN dasar','EfficientNet-B0'],[b['test_metrics'][k],e['test_metrics'][k]],color=['#8296a6','#165d73'])
        for j,val in enumerate([b['test_metrics'][k],e['test_metrics'][k]]):ax.text(j,.03,f'{val:.2%}',ha='center',color='white',fontsize=10)
        ax.set_title(k);ax.set_ylim(0,1)
    fig.suptitle('Perbandingan model pada test set yang sama')
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=180);plt.close(fig)
    predictor=Predictor(eff/'best.pt',eff/'threshold.json')
    split=json.loads((out/'split_manifest.json').read_text())
    rootdata=find_root(root/'data')
    probs=np.load(eff/'probabilities.npz');y=probs['test_y'];pred=probs['test_probs'].argmax(1)
    correct=np.where(y==pred)[0];wrong=np.where(y!=pred)[0]
    indices=list(correct[:1])+list(wrong[:1])
    if not indices:raise ValueError('No test images')
    fig,axes=plt.subplots(len(indices),2,figsize=(7,len(indices)*3.2))
    axes=np.atleast_2d(axes)
    for row,i in enumerate(indices):
        f=rootdata/split['test'][int(i)]['path']
        with Image.open(f) as img: original=img.convert('RGB')
        res=predictor.predict(original)
        axes[row,0].imshow(original);axes[row,0].axis('off')
        axes[row,0].set_title(f'True: {CLASS_NAMES[y[i]]}',fontsize=12)
        axes[row,1].imshow(res['gradcam']);axes[row,1].axis('off')
        axes[row,1].set_title(f'Pred: {res["label"]}, p={res["confidence"]:.4f}',fontsize=12)
    fig.tight_layout();fig.savefig(out/'gradcam_correct_and_wrong.png',dpi=180);plt.close(fig)
    # Real test cases for UI/video: do not invent a failure or a rejected image.
    selected=[]
    if len(correct):selected.append(('correct',int(correct[0])))
    if len(wrong):selected.append(('wrong',int(wrong[0])))
    rejected=np.flatnonzero(probs['test_probs'].max(1)<threshold['threshold'])
    if len(rejected):selected.append(('rejected',int(rejected[0])))
    examples=root/'examples';examples.mkdir(exist_ok=True)
    import shutil
    demos=[]
    for kind,i in selected:
        source=rootdata/split['test'][i]['path'];dest=examples/f'{kind}_{source.name}'
        shutil.copy2(source,dest)
        demos.append({'kind':kind,'image':str(dest.relative_to(root).as_posix()),
                      'test_index':i,'original_path':split['test'][i]['path'],
                      'true_label':CLASS_NAMES[int(y[i])],'predicted_label':CLASS_NAMES[int(pred[i])],
                      'confidence':float(probs['test_probs'][i].max()),
                      'accepted':bool(probs['test_probs'][i].max()>=threshold['threshold'])})
    (out/'demo_examples.json').write_text(json.dumps(demos,indent=2),encoding='utf8')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    before=threshold['baseline_without_rejection'];after=threshold['test_metrics']
    for ax,key,title in [(axs[0],'selective_accuracy','Accuracy prediksi yang ditampilkan'),(axs[1],'coverage','Coverage test')]:
        values=[before[key],after[key]]
        ax.bar(['Semua prediksi','Dengan threshold'],[v if v is not None else 0 for v in values],color=['#7891a5','#217678'])
        ax.set(title=title,ylim=(0,1.02))
        for j,v in enumerate(values):ax.text(j,.025,('Tidak tersedia' if v is None else f'{v:.2%}'),ha='center',fontsize=10,color='white')
    fig.tight_layout();fig.savefig(out/'threshold_comparison.png',dpi=180);plt.close(fig)
    summary={'baseline_accuracy':b['test_metrics']['accuracy'],'efficientnet_accuracy':e['test_metrics']['accuracy'],
             'baseline_macro_f1':b['test_metrics']['macro_f1'],'efficientnet_macro_f1':e['test_metrics']['macro_f1'],
             'threshold':threshold}
    (out/'final_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',default='.');a=p.parse_args();run(a.project)
