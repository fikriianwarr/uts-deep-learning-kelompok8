"""Regression checks for leakage, duplicate records, and checkpoint/threshold identity."""
import json
import numpy as np
import pytest
import torch
from PIL import Image
from src.data import CLASS_NAMES,make_manifest
from src.models import build_model
from src.infer import Predictor
from src.threshold import calibrate
from src.interpret import select_threshold


def test_manifest_rejects_duplicate_row_within_same_partition(tmp_path):
    root=tmp_path/'rgb'
    for cls in CLASS_NAMES:
        (root/cls).mkdir(parents=True)
        for i in range(10):Image.new('RGB',(64,64),(i,20,40)).save(root/cls/f'{i}.jpg')
    path=tmp_path/'split.json';m=make_manifest(root,path)
    m['train'].append(m['train'][0]);path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='duplicated'):make_manifest(root,path)


def test_threshold_refuses_unnormalized_negative_or_fractional_labels():
    for p,y in [(np.array([[.9,.9]]),[0]),(np.array([[-.1,1.1]]),[0]),(np.array([[.4,.6]]),[.5])]:
        with pytest.raises(ValueError):select_threshold(p,np.array(y))


def test_calibration_cannot_be_first_selected_after_test(tmp_path):
    (tmp_path/'best.pt').write_bytes(b'unit test checkpoint')
    (tmp_path/'results.json').write_text('{}')
    with pytest.raises(RuntimeError,match='before'):calibrate(tmp_path)


def test_predictor_rejects_threshold_for_different_checkpoint(tmp_path):
    model=build_model('baseline',pretrained=False)
    ck=tmp_path/'model.pt'
    torch.save({'model':'baseline','state_dict':model.state_dict(),'class_names':CLASS_NAMES,
                'image_size':64,'experiment':'FULL_DATA'},ck)
    th=tmp_path/'threshold.json';th.write_text(json.dumps({'threshold':.8,'checkpoint_sha256':'wrong'}))
    with pytest.raises(ValueError,match='another checkpoint'):Predictor(ck,th)


def test_frozen_efficientnet_has_only_head_trainable():
    model=build_model('efficientnet',pretrained=False,freeze_backbone=True)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==12810
