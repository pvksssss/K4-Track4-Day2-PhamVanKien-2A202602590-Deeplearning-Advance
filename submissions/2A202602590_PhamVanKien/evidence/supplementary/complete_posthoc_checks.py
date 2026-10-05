"""Authorized supplementary checks; original final/model selection unchanged.

Run this script from runs/ in the original workspace. Only val fits temperature;
test CSV probabilities are transformed after all temperatures are locked.
Full-pipeline sanity training reads train_subset0.csv only, never val/test images.
"""
from pathlib import Path
import sys,json,hashlib,time,dataclasses
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'submissions/2A202602590_PhamVanKien'
sys.path.insert(0,str(OUT/'code'));sys.path.insert(1,str(OUT))
import runtime
import torch,pandas as pd,timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import inference as inf,eval as ev,dataset as ds,model as models,train as tr,losses

SUP=OUT/'evidence/supplementary';PRED=OUT/'predictions/posthoc';PLOTS=OUT/'curves/supplementary'
PRED.mkdir(exist_ok=True);torch.set_num_threads(2)
versions={'python':sys.version.split()[0],'torch':torch.__version__,'timm':timm.__version__}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
locks=[]
for seed in range(3):
    source=OUT/f'predictions/F01_seed{seed}_val.csv';p=ev.read_pred(str(source))
    z=np.log(np.clip(p.probs,1e-300,1.0))
    T=inf.fit_temperature(z,p.y_true)
    locks.append({'seed':seed,'T':T,'fit_split':'val','objective':'NLL',
                  'val_source':str(source.relative_to(OUT)), 'val_sha256':digest(source),
                  'val_zero_probabilities':int((p.probs==0).sum()),'val_samples':len(p.y_true)})
lock={'phase':'post-hoc calibration; original F01 unmodified; not selected before original test',
      'method':'F01 hflip_logit; softmax(log(saved probability)/T)',
      'fit_method':'original inference.fit_temperature, NLL log-grid + golden section',
      'protocol':'all three T fitted on val and written before reading test CSV in this script',
      'date':'2026-10-05','temperatures':locks,'local_versions':versions}
(SUP/'calibration_temperature_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf8')
results=[]
for lockrow in locks:
    seed=lockrow['seed'];T=lockrow['T'];row={**lockrow,'splits':{}}
    for split in ['val','test']:
        source=OUT/f'predictions/F01_seed{seed}_{split}.csv';p=ev.read_pred(str(source))
        probs=inf.apply_temperature(np.log(np.clip(p.probs,1e-300,1.0)),T)
        assert np.array_equal(probs.argmax(1),p.y_pred),'Positive T must preserve argmax'
        dest=PRED/f'F01_TS_seed{seed}_{split}.csv'
        ev.save_predictions(dest,p.filenames,p.y_true,probs)
        checked=ev.read_pred(str(dest))
        before=ev.compute_metrics(p.y_true,p.y_pred,p.probs)
        after=ev.compute_metrics(checked.y_true,checked.y_pred,checked.probs)
        assert abs(before['macro_f1']-after['macro_f1'])<1e-12
        row['splits'][split]={'before':plain(before),'after':plain(after),
            'source':str(source.relative_to(OUT)),'source_sha256':digest(source),
            'output':str(dest.relative_to(OUT)),'n':len(p.y_true),
            'argmax_changes':int((p.y_pred!=checked.y_pred).sum()),
            'zero_probability_count':int((p.probs==0).sum())}
    results.append(row)
    print('calibration seed',seed,'T',T,'test ECE',row['splits']['test']['before']['ece'],'->',row['splits']['test']['after']['ece'],flush=True)
summary={}
for split in ['val','test']:
    summary[split]={}
    for phase in ['before','after']:
        summary[split][phase]={}
        for metric in ev.SCALARS:
            v=[r['splits'][split][phase][metric] for r in results]
            summary[split][phase][metric]={'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1))}
cal={**lock,'results':results,'summary':summary,
     'limitations':'original test results were already viewed; supplementary analysis only, lecturer determines I4a eligibility; validation ECE evaluated on fitting data'}
(SUP/'calibration_posthoc.json').write_text(json.dumps(cal,ensure_ascii=False,indent=2),encoding='utf8')
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for ax,key in zip(axes,['ece','nll']):
    vals=[summary['test'][p][key]['mean'] for p in ['before','after']]
    errs=[summary['test'][p][key]['std'] for p in ['before','after']]
    ax.bar(['Original F01','Post-hoc TS'],vals,yerr=errs,capsize=4,color=['#627a9a','#248c74'])
    ax.set_ylabel(key.upper());ax.set_title('Test: mean ± sample std, 3 seeds')
fig.suptitle('Supplementary calibration (T fitted on F01 val only)')
fig.tight_layout();fig.savefig(PLOTS/'calibration_posthoc.png',dpi=150);plt.close(fig)

# All-network overfit using actual project loader/model/loss/optimizer/scheduler/train/eval.
tr.set_seed(0)
train_df=pd.read_csv(ROOT/'data/labels/train_subset0.csv')
mini=train_df.groupby('Label',group_keys=False).head(1).sort_values('Label').reset_index(drop=True)
assert len(mini)==9 and sorted(mini['Label'].tolist())==list(range(9))
cfg=tr.Config(exp_id='POSTHOC_FULL_PIPELINE',seed=0,backbone='swin_tiny',init='scratch',
    img_size=224,epochs=250,batch_size=9,aug='basic',mix=None,loss='ce',
    label_smoothing=0,lr_backbone=1e-4,lr_head=1e-3,weight_decay=0,
    warmup_epochs=0,min_lr_ratio=.5,amp=True,num_workers=0,cache_images=True,
    grad_clip=1,save_test_predictions=False)
loader=ds.make_loader(mini,ROOT/'images',ds.build_transforms(False,224),batch_size=9,
    train=False,num_workers=0,pin_memory=False,cache_in_ram=True)
device=torch.device('cuda');m=models.build_model('swin_tiny',pretrained=False,num_classes=9,init='scratch').to(device)
assert all(p.requires_grad for p in m.parameters())
criterion=losses.build_criterion('ce')
optimizer=tr.build_optimizer(m,cfg);scheduler=tr.build_scheduler(optimizer,cfg,len(loader))
scaler=torch.amp.GradScaler('cuda',enabled=True)
_,yy,zz,initial=tr.evaluate(m,loader,criterion,device)
tracked={k:p.detach().cpu().clone() for k,p in m.named_parameters() if p.requires_grad and ('head' in k or k=='patch_embed.proj.weight')}
trace=[];begin=time.perf_counter()
for step in range(cfg.epochs):
    rr=tr.train_one_epoch(m,loader,criterion,optimizer,scheduler,scaler,cfg,device)
    _,yy,zz,ce=tr.evaluate(m,loader,criterion,device)
    acc=float((zz.argmax(1)==yy).mean())
    trace.append({'step':step+1,**rr,'eval_ce':float(ce),'eval_top1':acc})
    if (step+1)%10==0 or (ce<.01 and acc==1):print('full pipeline step',step+1,'CE',ce,'accuracy',acc,flush=True)
    if ce<.01 and acc==1:break
changed={k:float((dict(m.named_parameters())[k].detach().cpu()-v).abs().max()) for k,v in tracked.items()}
check={'phase':'post-hoc full-network pipeline check; not evidence of pre-training timing',
    'date':'2026-10-05','gpu':torch.cuda.get_device_name(0),'local_versions':versions,
    'config':dataclasses.asdict(cfg),'overrides':'fixed evaluation transform, no random augmentation/mix, CE, scratch all layers; baseline learning rates and zero weight decay for diagnostic only',
    'pipeline_functions':['dataset.make_loader','model.build_model','losses.build_criterion','train.build_optimizer','train.build_scheduler','train.train_one_epoch','train.evaluate'],
    'n_train_images':9,'filenames':mini['Filename'].tolist(),'labels':mini['Label'].tolist(),
    'initial_ce':initial,'ln9':float(np.log(9)),'final_ce':trace[-1]['eval_ce'],
    'final_top1':trace[-1]['eval_top1'],'steps':len(trace),'seconds':time.perf_counter()-begin,
    'all_parameters_trainable':all(p.requires_grad for p in m.parameters()),
    'parameter_max_abs_changes':changed,'backbone_changed':changed['patch_embed.proj.weight']>0,
    'passed':bool(trace[-1]['eval_ce']<.01 and trace[-1]['eval_top1']==1 and changed['patch_embed.proj.weight']>0),
    'test_images_accessed':False,'validation_images_accessed':False,'trace':trace}
(SUP/'pipeline_full_posthoc.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf8')
pd.DataFrame(trace).to_csv(SUP/'pipeline_full_posthoc_history.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
axes[0].semilogy([t['step'] for t in trace],[t['train_loss'] for t in trace],label='train CE (AMP)')
axes[0].semilogy([t['step'] for t in trace],[t['eval_ce'] for t in trace],label='same batch CE (FP32 eval)')
axes[0].axhline(.01,ls=':',color='grey',label='CE threshold .01');axes[0].legend();axes[0].set_ylabel('CE');axes[0].set_xlabel('optimizer step')
axes[1].plot([t['step'] for t in trace],[t['eval_top1'] for t in trace]);axes[1].set_ylim(0,1.05);axes[1].set_ylabel('same batch accuracy');axes[1].set_xlabel('optimizer step')
fig.suptitle('Post-hoc full-network pipeline: 9 train images, Swin scratch')
fig.tight_layout();fig.savefig(PLOTS/'pipeline_full_posthoc.png',dpi=150);plt.close(fig)
print('FULL PIPELINE',json.dumps({k:check[k] for k in ['initial_ce','final_ce','final_top1','steps','passed','backbone_changed']}),flush=True)
assert check['passed'],'Actual full-network overfit did not meet diagnostic criterion'
