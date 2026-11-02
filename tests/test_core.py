import numpy as np
import torch
from PIL import Image
from src.data import CLASS_NAMES, transform
from src.models import build_model
from src.interpret import select_threshold,selective_metrics,GradCAM,overlay_heatmap

def test_shapes():
    m=build_model('baseline',pretrained=False)
    x=torch.randn(1,3,64,64)
    assert m(x).shape==(1,10)
    with GradCAM(m) as grad:
        heat,probs=grad(x)
    assert heat.shape==(1,64,64)
    assert probs.shape==(1,10)
    assert overlay_heatmap(Image.new('RGB',(64,64)),heat[0]).size==(64,64)

def test_threshold():
    p=np.array([[.9,.1],[.6,.4],[.51,.49],[.2,.8]])
    y=np.array([0,1,0,1])
    out=select_threshold(p,y,.5)
    metric=selective_metrics(p,y,out['threshold'])
    assert metric['coverage']>=.5
    assert metric['accepted']+metric['rejected']==4

def test_transform():
    o=transform(64,False)(Image.new('RGB',(20,20)))
    assert o.shape==(3,64,64)
