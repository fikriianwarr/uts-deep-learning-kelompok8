"""Data integrity and reproducibility checks."""
import json
import pytest
from PIL import Image
from src.data import CLASS_NAMES, make_manifest


def test_existing_manifest_detects_changed_seed_and_inventory(tmp_path):
    root=tmp_path/'RGB'
    for name in CLASS_NAMES:
        (root/name).mkdir(parents=True)
        for i in range(10):
            Image.new('RGB',(64,64)).save(root/name/f'{i}.jpg')
    manifest_path=tmp_path/'manifest.json'
    original=make_manifest(root,manifest_path,seed=42)
    with pytest.raises(ValueError,match='seed'):
        make_manifest(root,manifest_path,seed=123)
    (root/CLASS_NAMES[0]/'new.jpg').write_bytes(b'extra image')
    with pytest.raises(ValueError,match='stale'):
        make_manifest(root,manifest_path,seed=42)
    assert sum(len(original[k]) for k in ('train','val','test'))==100




def test_manifest_detects_changed_bytes_even_if_names_identical(tmp_path):
    root=tmp_path/'RGB'
    for name in CLASS_NAMES:
        (root/name).mkdir(parents=True)
        for i in range(10):
            Image.new('RGB',(64,64),(i,20,30)).save(root/name/f'{i}.jpg')
    path=tmp_path/'manifest.json'
    original=make_manifest(root,path)
    assert len(original['dataset_sha256'])==64
    Image.new('RGB',(64,64),(200,20,30)).save(root/CLASS_NAMES[0]/'0.jpg')
    with pytest.raises(ValueError,match='contents'):
        make_manifest(root,path)
