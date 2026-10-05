"""Time actual calibrated hflip-logit model on random input, not test images."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'submissions/2A202602590_PhamVanKien'
sys.path.insert(0,str(OUT/'code'));sys.path.insert(1,str(OUT))
import runtime,model as models,lab_workflow as wf
from train import Config
cfg=Config(**json.loads((OUT/'evidence/runs/F01/seed0/config.json').read_text()))
T=json.loads((OUT/'evidence/supplementary/calibration_temperature_lock.json').read_text())['temperatures'][0]['T']
m=models.build_model(cfg.backbone,pretrained=False,num_classes=9).cuda()
ck=torch.load(ROOT/'runs/analysis_F01_seed0.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);del ck
result=wf.measure_method(m,cfg,torch.device('cuda'),'hflip_logit',temperature=T,iters=100)
result.update(exp_id='F01_TS',seed=0,temperature=T,method='hflip_logit + TS',
    source_checkpoint='deepweeds-lab/runs/F01/seed0/best.pt',evidence_date='2026-10-05',
    evidence_phase='post-hoc local calibrated model timing',bn_fused=False,test_images_accessed=False)
(OUT/'evidence/supplementary/calibrated_latency.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2),flush=True)
