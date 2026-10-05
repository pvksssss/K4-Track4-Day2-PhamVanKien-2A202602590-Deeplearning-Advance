"""Post-hoc batch32 timing on stored checkpoints; no dataset/test forward."""
from pathlib import Path
import sys,json,zipfile
import torch
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'submissions/2A202602590_PhamVanKien'
sys.path.insert(0,str(OUT/'code'));sys.path.insert(1,str(OUT))
import runtime,model as models,lab_workflow as wf
from benchmark import bench
rows=[]
for exp in ['T09','F01']:
    checkpoint=ROOT/f'runs/analysis_{exp}_seed0.pt'
    if not checkpoint.exists():
        with zipfile.ZipFile(ROOT/'results.zip') as z: checkpoint.write_bytes(z.read(f'deepweeds-lab/runs/{exp}/seed0/best.pt'))
    m=models.build_model('swin_tiny',pretrained=False,num_classes=9).cuda().eval()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);del ck
    x=torch.randn(32,3,224,224,device='cuda')
    methods=[('I00','single',1.),('I01','hflip_prob',1.),('I02','fivecrop_prob',1.),('I03','hflip_logit',1.),('I04','temperature',.6380212008953056)] if exp=='T09' else [('F01','hflip_logit',1.)]
    for eid,method,temp in methods:
        def forward():
            with torch.inference_mode():return wf._aggregate(wf._view_logits(m,x,method),method,temp)
        l=bench(forward,warmup=10,iters=50,sync=torch.cuda.synchronize)
        l.update(exp_id=eid,source_exp_id=exp,seed=0,method=method,temperature=temp,
                 gpu=torch.cuda.get_device_name(0),dtype='fp32',batch=32,bn_fused=False,
                 images_per_s=32000/l['mean'],warmup=10,img_size=224,
                 timing_scope='tensor views + forward + aggregation; excludes decoding/initial resize/normalization',
                 evidence_phase='post-hoc local supplementary',evidence_date='2026-10-05',
                 source_checkpoint=f'deepweeds-lab/runs/{exp}/seed0/best.pt',test_accessed=False)
        rows.append(l);(OUT/'evidence/supplementary/latency_batch32.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(eid,method,'batch32 images/s',l['images_per_s'],flush=True)
    del m,x;torch.cuda.empty_cache()
