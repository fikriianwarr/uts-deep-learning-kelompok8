"""Reproducible EuroSAT preprocessing and strictly disjoint stratified splits."""
from __future__ import annotations
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
ImageFile.LOAD_TRUNCATED_IMAGES = False

CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42


def prepare_archive(zip_file: str|Path, output_dir: str|Path) -> Path:
    """Extract only safe paths and return the directory containing the 10 class folders."""
    zip_file, output_dir = Path(zip_file), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file) as z:
        for item in z.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe ZIP path: {item.filename}')
        z.extractall(output_dir)
    matches = [p for p in output_dir.rglob('*') if p.is_dir() and all((p/c).is_dir() for c in CLASS_NAMES)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one EuroSAT class root; found {matches}')
    return matches[0]


def find_root(path: str|Path) -> Path:
    path = Path(path)
    for p in [path, *(p for p in path.rglob('*') if p.is_dir())]:
        if all((p/c).is_dir() for c in CLASS_NAMES): return p
    raise FileNotFoundError(f'EuroSAT class folders not found below {path}')


def inventory(root: str|Path) -> list[dict]:
    root = find_root(root)
    records = []
    for idx, cls in enumerate(CLASS_NAMES):
        for p in sorted((root/cls).glob('*.jpg')):
            records.append({'path': str(p.relative_to(root).as_posix()), 'label': idx})
    if not records: raise RuntimeError('No *.jpg files found')
    return records


def dataset_fingerprint(root: str|Path, records: list[dict]) -> str:
    """SHA-256 fingerprint over class labels, relative paths AND actual JPEG bytes.

    Detects content changes even when filenames and class counts remain unchanged.
    This audits byte-identical identity, not geographic/semantic independence.
    """
    root = find_root(root)
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda r: r['path']):
        digest.update(f"{item['label']}|{item['path']}\n".encode('utf-8'))
        with (root / item['path']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def make_manifest(root: str|Path, outfile: str|Path, seed: int=SEED, force: bool=False) -> dict:
    root, outfile = find_root(root), Path(outfile)
    if outfile.exists() and not force:
        manifest = json.loads(outfile.read_text(encoding='utf8'))
        if manifest.get('class_names') != CLASS_NAMES or manifest.get('seed') != seed:
            raise ValueError('Existing manifest label order or random seed mismatch')
        flat = [r['path'] for part in ('train', 'val', 'test') for r in manifest[part]]
        if len(flat) != len(set(flat)):
            raise ValueError('Manifest contains duplicated entries within or across splits')
        splits = [{r['path'] for r in manifest[part]} for part in ('train', 'val', 'test')]
        paths = set.union(*splits)
        actual_paths = {r['path'] for r in inventory(root)}
        if (sum(map(len, splits)) != len(paths) or paths != actual_paths
                or any(not (root / rel).is_file() for rel in paths)):
            raise ValueError('Manifest is stale, duplicated, or incompatible with current dataset')
        if manifest.get('dataset_sha256') != dataset_fingerprint(root, inventory(root)):
            raise ValueError('Manifest is stale: dataset contents or original digest changed')
        if any((r['label'] != CLASS_NAMES.index(Path(r['path']).parts[0]))
                for part in ('train','val','test') for r in manifest[part]):
            raise ValueError('Manifest has incorrect labels for class directory')
        return manifest
    records = inventory(root)
    labels = [r['label'] for r in records]
    idx_all = list(range(len(records)))
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.15, random_state=seed, stratify=labels)
    y_tv = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=round(len(records)*0.15), random_state=seed, stratify=y_tv)
    m = {'class_names':CLASS_NAMES, 'seed':seed, 'dataset_sha256':dataset_fingerprint(root, records), 'source':'EuroSAT_RGB.zip / Zenodo 10.5281/zenodo.7711810',
         'train':[records[i] for i in idx_train], 'val':[records[i] for i in idx_val], 'test':[records[i] for i in idx_test]}
    all_paths = [r['path'] for part in ('train', 'val', 'test') for r in m[part]]
    if len(all_paths) != len(set(all_paths)) or len(all_paths) != len(records):
        raise ValueError('Training/validation/testing split overlap or incomplete inventory')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(m, indent=2),encoding='utf8')
    return m


def transform(image_size: int=64, train: bool=False):
    ops = [transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC)]
    if train:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20, interpolation=InterpolationMode.BILINEAR, fill=(124,116,104))])
    ops.extend([transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    return transforms.Compose(ops)


class EuroSATDataset(Dataset):
    def __init__(self, root: str|Path, entries: list[dict], image_size: int=64, train: bool=False):
        self.root = find_root(root)
        self.entries = entries
        self.preprocess = transform(image_size,train)
    def __len__(self): return len(self.entries)
    def __getitem__(self,index):
        r=self.entries[index]
        with Image.open(self.root/r['path']) as img:
            image=self.preprocess(img.convert('RGB'))
        return image,r['label']


def seed_everything(seed=SEED):
    torch.set_num_threads(min(4, __import__('os').cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def loaders(root, manifest, image_size=64, batch_size=32, workers=2,
            sample_limit=0, seed=SEED):
    result={}
    for part in ('train','val','test'):
        dataset=EuroSATDataset(root,manifest[part],image_size,part=='train')
        if sample_limit > 0:
            # Stratified sampling only for QUICK DEVELOPMENT, NOT REPORTABLE FULL EVALUATION.
            rng=np.random.default_rng(seed)
            group=[]
            for i in range(10):
                matching=[j for j,r in enumerate(dataset.entries) if r['label']==i]
                group += rng.choice(matching,min(max(1,sample_limit//10),len(matching)),replace=False).tolist()
            dataset=Subset(dataset, sorted(group))
        generator = torch.Generator().manual_seed(seed + {'train':0,'val':1,'test':2}[part])
        result[part]=DataLoader(dataset,batch_size=batch_size,shuffle=part=='train',
                                num_workers=workers,pin_memory=torch.cuda.is_available(),
                                generator=generator)
    return result


def compute_duplicate_hashes(root, entries):
    """Byte-exact duplicates; near-duplicates/spatial overlap are NOT ruled out."""
    root=find_root(root); digests={}; collisions=[]
    for item in entries:
        name=item['path']; digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if digest in digests: collisions.append((digests[digest], name))
        else:digests[digest]=name
    return collisions
