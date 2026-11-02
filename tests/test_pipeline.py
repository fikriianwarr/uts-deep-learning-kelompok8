"""Pipeline validation for split separation, model outputs, and finalization."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.data import CLASS_NAMES, make_manifest, loaders
from src.models import build_model
from src.interpret import GradCAM,select_threshold,selective_metrics
from src.finalize import finalize
from src.infer import Predictor


def test_disjoint_and_reproducible_stratified_splits(tmp_path):
    data=tmp_path/'dataset'
    for name in CLASS_NAMES:
        (data/name).mkdir(parents=True)
        for i in range(10):
            Image.new('RGB',(64,64),(i*5,i*3,i*7)).save(data/name/f'{name}_{i}.jpg')
    manifest_path=tmp_path/'split.json'
    m=make_manifest(data,manifest_path)
    assert tuple(len(m[x]) for x in ('train','val','test'))==(70,15,15)
    groups=[{i['path'] for i in m[x]} for x in ('train','val','test')]
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert make_manifest(data,manifest_path)==m
    tiny=loaders(data,m,image_size=64,batch_size=8,workers=0,sample_limit=0)
    assert len(tiny['train'].dataset)==70


def test_validation_threshold_rejects_invalid_arguments():
    a=np.array([[.9,.1],[.4,.6],[.8,.2]])
    y=np.array([0,1,1])
    with pytest.raises(ValueError):select_threshold(a,y,min_coverage=0)
    with pytest.raises(ValueError):select_threshold(np.array([[np.nan,.1]]),np.array([0]))
    t=select_threshold(a,y,min_coverage=2/3)
    assert selective_metrics(a,y,t['threshold'])['coverage']>=2/3


def test_transfer_learning_architecture_gradcam_no_download():
    model=build_model('efficientnet',pretrained=False)
    model.eval()
    assert model.classifier[-1].out_features==10
    with GradCAM(model) as grad:
        heat,prob=grad(torch.randn(1,3,64,64))
    assert heat.shape==(1,64,64)
    assert prob.shape==(1,10)
    assert np.isfinite(heat).all()


def test_finalization_rejects_missing_evidence(tmp_path):
    with pytest.raises(FileNotFoundError):finalize(project=tmp_path, data_root=tmp_path)


def test_predictor_refuses_smoke_checkpoint(tmp_path):
    model=build_model('baseline',pretrained=False)
    p=tmp_path/'untrained.pt'
    torch.save({'model':'baseline','state_dict':model.state_dict(),
                'class_names':CLASS_NAMES,'image_size':64,'experiment':'SMOKE_TEST_NOT_FINAL'},p)
    with pytest.raises(ValueError, match='smoke-test'):
        Predictor(p)
