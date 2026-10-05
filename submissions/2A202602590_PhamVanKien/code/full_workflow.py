"""Sequential all-stage runner; scientific selection uses validation only."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import lab_workflow as wf
from train import Config


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.digest()


def restore_outputs(source, target):
    """Restore one explicitly selected output; never overwrite conflicting files."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('Previous output and working directory must be different')
    if not any((source / 'runs').glob('*/seed*/config.json')):
        raise FileNotFoundError(f'No experiment configs in previous output: {source}')
    planned = []
    for folder in ('runs', 'curves', 'logits', 'predictions'):
        for path in sorted((source / folder).rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(source / folder)
            # Fresh setup regenerates EDA, environment and split checks.
            if folder == 'runs' and len(relative.parts) == 1 and relative.name not in (
                'selection.json', 'full_baseline.json', 'full_seeds.json', 'combination.json'
            ):
                continue
            dest = target / folder / relative
            if dest.exists():
                if not dest.is_file() or _digest(path) != _digest(dest):
                    raise ValueError(f'Output conflict; no files copied: {dest}')
            else:
                planned.append((path, dest))
    for path, dest in planned:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    print(f'Restored {len(planned)} files from {source}')
    return len(planned)


def run_group(configs):
    """Reuse completed runs and resume incomplete runs at the last saved epoch."""
    prepared = []
    for cfg in configs:
        folder = wf.run_dir(cfg)
        if (folder / 'config.json').exists():
            wf.validate_saved_config(cfg)
        prepared.append(dataclasses.replace(cfg, resume=True)
                        if (folder / 'last.pt').exists() and not (folder / 'summary.json').exists()
                        else cfg)
    return wf.run_experiments(prepared)


def combined_config(configs):
    """T09 combines the best nonbaseline augmentation and loss by val macro-F1.

    This is an interaction experiment, not a single-factor ablation. If neither
    axis beats T00, still test their combination and report that limitation.
    """
    baseline = next(c for c in configs if c.exp_id == 'T00')
    aug = wf.best_config([c for c in configs if c.exp_id in ('T03', 'T04', 'T07', 'T08')])
    loss = wf.best_config([c for c in configs if c.exp_id in ('T05', 'T06')])
    combo = dataclasses.replace(baseline, exp_id='T09', aug=aug.aug, mix=aug.mix,
                                mix_alpha=aug.mix_alpha, loss=loss.loss,
                                label_smoothing=loss.label_smoothing,
                                focal_gamma=loss.focal_gamma,
                                save_test_predictions=False)
    record = {'exp_id': 'T09', 'augmentation_source': aug.exp_id, 'loss_source': loss.exp_id,
              'selection_split': 'val', 'config': dataclasses.asdict(combo),
              'note': 'Interaction test of best nonbaseline augmentation and loss; gains not assumed.'}
    path = Path(baseline.out_dir) / 'combination.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != record:
        raise ValueError('Combination selection changed; use a separate study directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return combo


def _same_recipe(left, right):
    left, right = dataclasses.asdict(left), dataclasses.asdict(right)
    for key in ('resume', 'save_test_predictions'):
        left.pop(key)
        right.pop(key)
    return left == right


def run_all(base, *, allow_final_test=False, seeds=(0, 1, 2)):
    """Run every stage, locking validation choices before any final test access."""
    if not allow_final_test:
        raise ValueError('Set ALLOW_FINAL_TEST=True to authorize the full final evaluation')
    seeds = tuple(seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Final evaluation requires at least three distinct nonnegative integer seeds')
    if base.save_test_predictions or base.debug_samples_per_class is not None:
        raise ValueError('Full workflow requires full train/val with test disabled until final')
    if base.seed != 0:
        raise ValueError('Screening seed must be 0 to keep B/T comparisons and restoration consistent')
    root = Path(base.out_dir).parent
    folder = Path(base.out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    selection = folder / 'selection.json'
    baseline_path = folder / 'full_baseline.json'
    seeds_path = folder / 'full_seeds.json'
    status_path = folder / 'full_status.json'
    stage = 'start'

    def status(name, state='running', error=None):
        status_path.write_text(json.dumps({'stage': name, 'status': state, 'error': error}, indent=2),
                               encoding='utf-8')
        print(f'\n=== FULL: {name} | {state} ===', flush=True)

    try:
        if selection.exists():
            # Do not repeat selection after test results may have been exposed.
            locked = json.loads(selection.read_text(encoding='utf-8'))
            if locked.get('selection_split') != 'val' or locked.get('method') not in wf.METHODS:
                raise ValueError('Invalid validation selection lock')
            recipe = Config(**locked['config'])
            baseline_source = baseline_path if baseline_path.exists() else folder / 'T00/seed0/config.json'
            baseline = Config(**json.loads(baseline_source.read_text(encoding='utf-8')))
            baseline = dataclasses.replace(baseline, save_test_predictions=False, resume=False)
            expected = dataclasses.replace(base, exp_id='T00', backbone=baseline.backbone)
            if not _same_recipe(baseline, expected):
                raise ValueError('Base config changed after selection was locked')
            for key in ('out_dir', 'pred_dir', 'curves_dir', 'logit_dir', 'images_dir', 'labels_dir'):
                if getattr(recipe, key) != getattr(base, key):
                    raise ValueError(f'Locked recipe path changed: {key}')
            print('Reusing locked validation selection; no re-selection.', flush=True)
        else:
            if any(Path(base.pred_dir).glob('*_test.csv')) or any(folder.glob('F01/seed*/final_test_arrays.npz')):
                raise ValueError('Test outputs exist without a selection lock; refusing re-selection')
            stage = 'backbones'
            status(stage)
            backbones = wf.experiment_plan(base, 'backbones')
            run_group(backbones)
            winner = wf.best_config(backbones)
            stage = 'training'
            status(stage)
            training = wf.experiment_plan(dataclasses.replace(base, backbone=winner.backbone), 'training')
            run_group(training)
            combo = combined_config(training)
            run_group([combo])
            recipe = wf.best_config(training + [combo])
            stage = 'inference'
            status(stage)
            wf.compare_inference(recipe)
            inference_dir = folder / 'inference' / recipe.exp_id / f'seed{recipe.seed}'
            candidates = [json.loads(p.read_text(encoding='utf-8')) for p in inference_dir.glob('I*.json')]
            if {r['method'] for r in candidates} != set(wf.METHODS) or len(candidates) != len(wf.METHODS):
                raise ValueError('Complete all inference comparisons before final evaluation')
            chosen = min(candidates, key=lambda r: (-r['val_macro_f1'], r['val_ece'], r['latency']['p95']))
            baseline = training[0]
            baseline_path.write_text(json.dumps(dataclasses.asdict(baseline), indent=2), encoding='utf-8')
            selection = wf.lock_selection(recipe, chosen['method'])
        if seeds_path.exists() and json.loads(seeds_path.read_text(encoding='utf-8')) != list(seeds):
            raise ValueError('Final seeds changed after locking; use the original seeds')
        seeds_path.write_text(json.dumps(list(seeds)), encoding='utf-8')
        stage = 'final'
        status(stage)
        # Resume incomplete epoch checkpoints before the existing final runner.
        pending = []
        for seed in seeds:
            b = dataclasses.replace(baseline, exp_id='T00', seed=seed, save_test_predictions=False, resume=False)
            if not (Path(b.pred_dir) / f'T00_seed{seed}_test.csv').exists():
                pending.append(b)
            f = dataclasses.replace(recipe, exp_id='F01', seed=seed, save_test_predictions=False, resume=False)
            if not (Path(f.pred_dir) / f'F01_seed{seed}_test.csv').exists() and not (
                wf.run_dir(f) / 'final_test_arrays.npz').exists():
                pending.append(f)
        for cfg in pending:
            if (wf.run_dir(cfg) / 'last.pt').exists() and not (wf.run_dir(cfg) / 'summary.json').exists():
                run_group([cfg])
        wf.run_final(selection, baseline, seeds=seeds)
        stage = 'export'
        status(stage)
        result = wf.export_results(root)
        status('complete', 'complete')
        return result
    except Exception as exc:
        status(stage, 'failed', f'{type(exc).__name__}: {exc}')
        raise
