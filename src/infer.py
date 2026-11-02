"""Safe model checkpoint loading and single image inference."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import CLASS_NAMES,transform
from .models import build_model
from .interpret import GradCAM,overlay_heatmap

class Predictor:
    def __init__(self,checkpoint,threshold_path=None,device='cpu',require_full_data=True):
        self.device=torch.device(device)
        # Only load checkpoints from our own training script; never untrusted third-party pt files.
        ck=torch.load(Path(checkpoint),map_location=self.device,weights_only=True)
        if ck['class_names']!=CLASS_NAMES:raise ValueError('Checkpoint class order mismatch')
        if require_full_data and ck.get('experiment') != 'FULL_DATA':
            raise ValueError('A demo/report cannot use a smoke-test model checkpoint')
        self.model_name=ck['model'];self.size=int(ck['image_size'])
        self.model=build_model(self.model_name,10,pretrained=False).to(self.device)
        self.model.load_state_dict(ck['state_dict']);self.model.eval()
        self.threshold=0.0
        if threshold_path is not None:
            if not Path(threshold_path).is_file():
                raise FileNotFoundError(f'Validation-calibrated threshold missing: {threshold_path}')
            import json
            record=json.loads(Path(threshold_path).read_text())
            import hashlib
            if record.get('checkpoint_sha256')!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                raise ValueError('Threshold belongs to another checkpoint')
            self.threshold=float(record['threshold'])
            if not 0<=self.threshold<=1:raise ValueError('Threshold must be in [0,1]')
    def predict(self,img):
        if not isinstance(img,Image.Image):img=Image.open(img)
        img=img.convert('RGB')
        x=transform(self.size,False)(img)[None].to(self.device)
        with GradCAM(self.model) as cam:
            heat,prob=cam(x)
        prob=prob[0];cls=int(np.argmax(prob));score=float(prob[cls])
        return {'label':CLASS_NAMES[cls], 'confidence':score, 'accepted':score>=self.threshold,
                'threshold':self.threshold,'probabilities':{CLASS_NAMES[i]:float(prob[i]) for i in range(10)},
                'gradcam':overlay_heatmap(img.resize((self.size,self.size)),heat[0])}
