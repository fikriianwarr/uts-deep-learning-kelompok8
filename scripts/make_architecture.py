"""Measure real tensors/parameters and render both CNN architectures."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.models import build_model,count_params

def main():
    torch.set_num_threads(4)
    cfg=json.loads((ROOT/'experiment_config.json').read_text());size=cfg['image_size']
    details={}
    for name in ['baseline','efficientnet']:
        model=build_model(name,pretrained=False).eval()
        x=torch.zeros(1,3,size,size);rows=[{'stage':'Input RGB','output':list(x.shape),'parameters':0}]
        if name=='baseline':
            for k in range(4):
                block=model.features[4*k:4*(k+1)]
                with torch.no_grad():x=block(x)
                rows.append({'stage':f'Conv BN ReLU MaxPool {k+1}','output':list(x.shape),'parameters':count_params(block)})
        else:
            for k,block in enumerate(model.features):
                with torch.no_grad():x=block(x)
                rows.append({'stage':f'EfficientNet features {k}','output':list(x.shape),'parameters':count_params(block)})
        rows.append({'stage':'AdaptiveAvgPool Dropout Linear','output':[1,10],'parameters':count_params(model.classifier)})
        details[name]={'total_parameters':count_params(model),'leaf_layers':sum(not list(m.children()) for m in model.modules()),'stages':rows}
    out=ROOT/'artifacts/eda';out.mkdir(parents=True,exist_ok=True)
    (out/'architecture_details.json').write_text(json.dumps(details,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,9))
    for ax,(name,info) in zip(axes,details.items()):
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
        ax.set_title(f'{name} — {info["total_parameters"]:,} parameter',fontsize=14)
        n=len(info['stages']);step=.89/n;h=step*.73
        for i,row in enumerate(info['stages']):
            cy=.94-i*step
            box=FancyBboxPatch((.03,cy-h),.94,h,boxstyle='round,pad=.008',facecolor='#f1f4f6',edgecolor='#748796')
            ax.add_patch(box)
            dims=' × '.join(map(str,row['output']))
            ax.text(.5,cy-h/2,f'{row["stage"]}\n{dims}',ha='center',va='center',fontsize=13)
            if i<n-1:ax.annotate('',xy=(.5,cy-step+.006),xytext=(.5,cy-h-.008),arrowprops=dict(arrowstyle='->',color='#40566a'))
    fig.tight_layout();fig.savefig(out/'architecture_diagram.png',dpi=180);plt.close(fig)
    print(json.dumps({k:{x:y for x,y in v.items() if x!='stages'} for k,v in details.items()},indent=2))

if __name__=='__main__':main()
