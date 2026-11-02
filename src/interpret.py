"""Grad-CAM and validation-only confidence threshold calibration."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from .models import last_conv

class GradCAM:
    def __init__(self,model):
        self.model=model; self.activations=None;self.gradients=None
        self.layer=last_conv(model)
        self.a_handle=self.layer.register_forward_hook(self._forward)
        self.g_handle=self.layer.register_full_backward_hook(self._backward)
    def _forward(self,mod,inputs,output):self.activations=output.detach()
    def _backward(self,mod,grad_inputs,grad_outputs):self.gradients=grad_outputs[0].detach()
    def close(self):self.a_handle.remove();self.g_handle.remove()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __call__(self,images,target_index=None):
        self.model.eval();self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits=self.model(images)
            idx=logits.argmax(1) if target_index is None else torch.full((len(images),),int(target_index),device=logits.device)
            logits.gather(1,idx[:,None]).sum().backward()
            weights=self.gradients.mean(dim=(2,3),keepdim=True)
            cam=torch.relu((weights*self.activations).sum(1,keepdim=True))
            cam=torch.nn.functional.interpolate(cam,size=images.shape[-2:],mode='bilinear',align_corners=False)
            cam-=cam.amin(dim=(2,3),keepdim=True)
            cam/=cam.amax(dim=(2,3),keepdim=True).clamp(min=1e-8)
        return cam[:,0].detach().cpu().numpy(),logits.detach().softmax(1).cpu().numpy()

def overlay_heatmap(rgb_image, heatmap, alpha=0.45):
    from matplotlib import colormaps as cm
    base=np.asarray(rgb_image.convert('RGB'),dtype=np.float32)/255
    heat=np.asarray(Image.fromarray(np.uint8(heatmap*255)).resize(rgb_image.size),dtype=np.float32)/255
    colored=cm.get_cmap('jet')(heat)[:,:,:3]
    out=(1-alpha)*base+alpha*colored
    return Image.fromarray(np.uint8(np.clip(out,0,1)*255))

def select_threshold(val_probabilities, val_targets, min_coverage=0.70):
    """Choose max selective accuracy with >= min coverage; tie: highest coverage.
    Uses validation only, never test labels.
    """
    if not (0 < min_coverage <= 1):
        raise ValueError('min_coverage must be in (0,1]')
    p=np.asarray(val_probabilities)
    y=np.asarray(val_targets)
    if (y.ndim!=1 or len(y)==0 or p.ndim!=2 or p.shape[1]<2 or p.shape[0]!=len(y)
        or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1)
        or not np.allclose(p.sum(1),1,atol=1e-5) or np.any(y<0)
        or np.any(y>=p.shape[1]) or not np.all(y==y.astype(int))):
        raise ValueError('Expected finite non-empty [N,classes] validation probabilities')
    conf=p.max(1);correct=p.argmax(1)==y
    choices=np.r_[0,np.unique(conf)]
    records=[]
    for t in choices:
        accept=conf>=t
        if accept.mean()>=min_coverage and accept.any():
            records.append((correct[accept].mean(),accept.mean(),float(t)))
    if not records:raise ValueError('No valid confidence threshold')
    best=max(records,key=lambda row:(row[0],row[1]))
    return {'threshold':best[2],'validation_selective_accuracy':float(best[0]),
            'validation_coverage':float(best[1]),'minimum_coverage':min_coverage}

def selective_metrics(probs,labels,threshold):
    p=np.asarray(probs);y=np.asarray(labels)
    if p.ndim!=2 or y.ndim!=1 or not len(y) or len(y)!=len(p) or not np.isfinite(p).all() or not 0<=threshold<=1:
        raise ValueError('Invalid probabilities, labels, or threshold')
    pred=p.argmax(1); accept=p.max(1)>=threshold
    correct=pred==y
    return {'threshold':float(threshold), 'coverage':float(accept.mean()),
            'selective_accuracy':float(correct[accept].mean()) if accept.any() else None,
            'overall_accuracy':float(correct.mean()),'accepted':int(accept.sum()),
            'rejected':int((~accept).sum()),'rejected_errors':int((~accept & ~correct).sum()),
            'rejected_correct':int((~accept & correct).sum())}
