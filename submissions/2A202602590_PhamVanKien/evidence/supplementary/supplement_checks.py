"""New post-hoc local checks, kept separate from original Kaggle experiments."""
from pathlib import Path
import sys, json, zipfile, time, dataclasses
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submissions/2A202602590_PhamVanKien'
sys.path.insert(0,str(OUT/'code'))
sys.path.insert(1,str(OUT))
import runtime
import dataset as ds
import model as models
import lab_workflow as wf
from train import Config, set_seed

set_seed(0)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evidence=OUT/'evidence'
supplement=evidence/'supplementary'; supplement.mkdir(parents=True,exist_ok=True)
plots=OUT/'curves/supplementary'; plots.mkdir(parents=True,exist_ok=True)
latency=[]
with zipfile.ZipFile(ROOT/'results.zip') as archive:
    for exp in ['B01','B02','B03','B04','B05','F01']:
        folder=OUT/f'evidence/runs/{exp}/seed0'
        cfg=Config(**json.loads((folder/'config.json').read_text(encoding='utf-8')))
        name=f'deepweeds-lab/runs/{exp}/seed0/best.pt'
        # A single checkpoint at a time, for fixed random-input timing only.
        checkpoint=ROOT/f'runs/analysis_{exp}_seed0.pt'
        checkpoint.write_bytes(archive.read(name))
        model=models.build_model(cfg.backbone,pretrained=False,num_classes=9,init='finetune').to(device)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']); del ck
        method='hflip_logit' if exp=='F01' else 'single'
        result=wf.measure_method(model,cfg,device,method,iters=100)
        result.update(exp_id=exp,seed=0,method=method,source_checkpoint=name,
                      evidence_date='2026-10-05',evidence_phase='post-hoc local supplementary',
                      warmup=10,bn_fused=False)
        latency.append(result)
        print(exp,cfg.backbone,'p95',result['p95'],'GPU',result['gpu'],flush=True)
        (supplement/'latency_local.json').write_text(json.dumps(latency,indent=2),encoding='utf-8')
        del model
        if device.type=='cuda': torch.cuda.empty_cache()

# Only scratch model, no pretrained download, no test data.
train,_,_=ds.load_split(ROOT/'data/labels')
mini=train.groupby('Label',group_keys=False).head(2).reset_index(drop=True)
loader=ds.make_loader(mini,ROOT/'images',ds.build_transforms(False,224),batch_size=18,
                      train=False,num_workers=0,pin_memory=False)
x,y,names=next(iter(loader)); x,y=x.to(device),y.to(device)
set_seed(0)
model=models.build_model('swin_tiny',pretrained=False,num_classes=9,init='scratch').to(device).eval()
models.freeze_backbone(model)
with torch.no_grad():
    features=model.forward_head(model.forward_features(x),pre_logits=True).detach()
head=models.get_classifier(model)
initial=float(torch.nn.functional.cross_entropy(head(features),y).detach())
optimizer=torch.optim.Adam(head.parameters(),lr=.03)
trace=[]
for step in range(500):
    optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(head(features),y)
    loss.backward(); optimizer.step()
    trace.append(float(loss.detach()))
    if trace[-1]<.001: break
check={'phase':'post-hoc verification, not before original Kaggle training','date':'2026-10-05',
       'model':'swin_tiny scratch; backbone frozen, classifier-only overfit',
       'gpu':str(torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'),
       'n_train_images':18,'seed':0,'initial_ce':initial,'ln9':float(np.log(9)),
       'final_ce':trace[-1],'steps':len(trace),'loss_trace':trace,
       'filenames':list(names),'test_accessed':False}
(supplement/'pipeline_posthoc.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
fig,ax=plt.subplots(figsize=(7,4)); ax.semilogy(range(1,len(trace)+1),trace)
ax.set_xlabel('Classifier update'); ax.set_ylabel('Fixed train batch CE (log scale)')
ax.set_title('Post-hoc scratch Swin head check — 18 train images, no test')
ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(plots/'pipeline_posthoc.png',dpi=150); plt.close(fig)

# Visual evidence from the implemented augmentation, generated after the original experiment.
for aug in ('trivial','basic','color'):
    set_seed(0)
    loader=ds.make_loader(mini.head(8),ROOT/'images',ds.build_transforms(True,224,aug),
                          batch_size=8,train=False,num_workers=0,pin_memory=False)
    images,labels,filenames=next(iter(loader))
    mean=torch.tensor(ds.IMAGENET_MEAN).view(3,1,1); std=torch.tensor(ds.IMAGENET_STD).view(3,1,1)
    fig,axes=plt.subplots(2,4,figsize=(10,5))
    for ax,img,label,filename in zip(axes.flat,images,labels,filenames):
        ax.imshow((img*std+mean).permute(1,2,0).clamp(0,1).numpy())
        ax.set_title(f'{ds.CLASS_NAMES[int(label)]}\n{filename}',fontsize=7); ax.axis('off')
    fig.suptitle(f'Post-hoc augmentation examples: {aug} (train only)')
    fig.tight_layout(); fig.savefig(plots/f'augmentation_{aug}.png',dpi=130); plt.close(fig)
print('Supplementary checks complete',json.dumps({k:check[k] for k in ['initial_ce','final_ce','steps']}),flush=True)
