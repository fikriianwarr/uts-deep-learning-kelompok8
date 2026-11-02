"""Reproducible dataset audit with actual numeric outputs and images."""
import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import CLASS_NAMES,find_root,inventory,make_manifest,compute_duplicate_hashes


def audit(root: str|Path, out: str|Path, hash_duplicates=True):
    root=find_root(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    samples=inventory(root)
    counter=Counter(CLASS_NAMES[x['label']] for x in samples)
    sizes=Counter(); modes=Counter(); broken=[]
    means=[]
    rgb_sample=[]
    for i,r in enumerate(samples):
        try:
            with Image.open(root/r['path']) as im:
                im.verify()
            with Image.open(root/r['path']) as im:
                sizes[str(im.size)]+=1;modes[im.mode]+=1
                if i%20==0:
                    pixels=np.asarray(im.convert('RGB'),dtype=np.float32)
                    means.append(pixels.mean(axis=(0,1)))
                    rgb_sample.append(pixels.reshape(-1,3)[::32])
        except Exception as ex: broken.append({'path':r['path'],'error':str(ex)})
    dup=compute_duplicate_hashes(root,samples) if hash_duplicates else []
    manifest=make_manifest(root,out/'split_manifest.json')
    stats={'total':len(samples),'counts':dict(counter),'sizes':dict(sizes),'modes':dict(modes),
           'verified_broken_files':broken,'sha256_exact_duplicate_pairs':dup,
           'estimated_channel_means_sample':np.mean(means,axis=0).round(2).tolist(),
           'channel_sample_images':len(means),
           'splits':{k:len(manifest[k]) for k in ('train','val','test')},
           'limitation':'Identical-byte hashes do not detect near-duplicate imagery or spatial leakage.'}
    (out/'dataset_audit.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    if rgb_sample:
        all_pixels=np.concatenate(rgb_sample,axis=0)
        fig,ax=plt.subplots(figsize=(9,4.6))
        for channel,label,color in zip(range(3),('Red','Green','Blue'),('#bc4b46','#49805b','#3e78a9')):
            ax.hist(all_pixels[:,channel],bins=32,range=(0,256),alpha=.35,label=label,color=color,density=True)
        ax.set(xlabel='Pixel intensity (0–255)',ylabel='Density',title=f'RGB intensity sample ({len(means)} images, every 20th image)')
        ax.legend();fig.tight_layout();fig.savefig(out/'rgb_histogram_sample.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(list(counter),list(counter.values()),color='#246981')
    ax.tick_params(axis='x',rotation=43);ax.set_ylim(0,3400)
    ax.set_title('Distribusi kelas EuroSAT RGB (hasil pemeriksaan ZIP)');ax.set_ylabel('Jumlah citra')
    for i,v in enumerate(counter.values()):ax.text(i,v+55,str(v),ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(out/'class_distribution.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(6,5,figsize=(12,12))
    for cls_index,c in enumerate(CLASS_NAMES):
        files=[x['path'] for x in samples if CLASS_NAMES[x['label']]==c][:3]
        for k,fn in enumerate(files):
            ax=axs[(cls_index//5)*3+k,cls_index%5]
            with Image.open(root/fn) as im:ax.imshow(im.convert('RGB'))
            ax.set_title(c if k==0 else Path(fn).stem,fontsize=8)
            ax.axis('off')
    fig.suptitle('Tiga contoh citra setiap kelas EuroSAT RGB',fontsize=13)
    fig.tight_layout();fig.savefig(out/'class_samples.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in stats.items() if k!='sha256_exact_duplicate_pairs'},indent=2),flush=True)
    return stats

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='data/EuroSAT_RGB');a.add_argument('--out',default='artifacts/eda')
    x=a.parse_args();audit(x.root,x.out)
