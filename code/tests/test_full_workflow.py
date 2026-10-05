"""Full Run All orchestration, without expensive GPU training."""
import dataclasses
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))


class FullWorkflowTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT / 'code/full_workflow.py').exists(), 'full workflow is missing')
        return importlib.import_module('full_workflow')

    def fixture(self, root):
        for index in range(1, 6):
            folder = root / 'runs' / f'B{index:02}' / 'seed0'
            folder.mkdir(parents=True)
            (folder / 'config.json').write_text('{}')
            (folder / 'summary.json').write_text('{"val_macro_f1": 0.8}')
        (root / 'runs/environment.json').write_text('{"old": true}')
        (root / 'runs/split_check_fold0.json').write_text('{"old": true}')

    def test_restore_experiments_not_old_environment_or_split_cache(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'input', Path(tmp) / 'working'
            self.fixture(src)
            mod.restore_outputs(src, dst)
            self.assertTrue((dst / 'runs/B05/seed0/summary.json').exists())
            self.assertFalse((dst / 'runs/environment.json').exists())
            self.assertFalse((dst / 'runs/split_check_fold0.json').exists())
            mod.restore_outputs(src, dst)

    def test_restore_partial_backbones_with_last_checkpoint(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'input', Path(tmp) / 'working'
            folder = src / 'runs/B01/seed0'
            folder.mkdir(parents=True)
            (folder / 'config.json').write_text('{}')
            (folder / 'last.pt').write_bytes(b'partial checkpoint')
            mod.restore_outputs(src, dst)
            self.assertTrue((dst / 'runs/B01/seed0/last.pt').exists())

    def test_restore_empty_source_fails_without_writing(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'input', Path(tmp) / 'working'
            src.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, 'experiment'):
                mod.restore_outputs(src, dst)
            self.assertFalse(dst.exists())

    def test_restore_conflict_does_not_copy_any_other_files(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'input', Path(tmp) / 'working'
            self.fixture(src)
            p = dst / 'runs/B05/seed0/config.json'
            p.parent.mkdir(parents=True)
            p.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'conflict|Conflict'):
                mod.restore_outputs(src, dst)
            self.assertFalse((dst / 'runs/B01/seed0/summary.json').exists())
            self.assertEqual(p.read_text(), '{"changed": true}')

    def test_combination_uses_val_not_test_and_keeps_other_baseline_fields(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = mod.wf.base_config(tmp)
            configs = mod.wf.experiment_plan(base, 'training')
            for cfg in configs:
                folder = mod.wf.run_dir(cfg)
                folder.mkdir(parents=True)
                (folder / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg)))
                score = {'T04': .92, 'T06': .93}.get(cfg.exp_id, .80)
                (folder / 'summary.json').write_text(json.dumps({
                    'val_macro_f1': score, 'test_macro_f1': 1. if cfg.exp_id == 'T03' else 0.
                }))
            combo = mod.combined_config(configs)
            self.assertEqual((combo.exp_id, combo.aug, combo.loss), ('T09', 'color', 'focal'))
            self.assertEqual(combo.init, base.init)
            self.assertEqual(combo.lr_backbone, base.lr_backbone)
            self.assertFalse(combo.save_test_predictions)

    def test_all_requires_consent_and_three_distinct_seeds(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = mod.wf.base_config(tmp)
            with patch.object(mod.wf, 'run_experiments') as runs:
                with self.assertRaisesRegex(ValueError, 'ALLOW_FINAL_TEST'):
                    mod.run_all(base, allow_final_test=False)
                with self.assertRaisesRegex(ValueError, 'seed'):
                    mod.run_all(base, allow_final_test=True, seeds=(0, 0, 1))
                runs.assert_not_called()

    def test_stage_order_locks_before_final_and_includes_combination(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = mod.wf.base_config(tmp)
            events = []
            def group(configs):
                events.append([c.exp_id for c in configs])
            def inference(cfg):
                events.append('inference')
                folder = Path(cfg.out_dir) / 'inference' / cfg.exp_id / f'seed{cfg.seed}'
                folder.mkdir(parents=True)
                for i, method in enumerate(mod.wf.METHODS):
                    (folder / f'I{i:02}.json').write_text(json.dumps({
                        'method': method, 'val_macro_f1': .9, 'val_ece': .1,
                        'latency': {'p95': 10 + i}}))
            def lock(cfg, method):
                events.append('lock')
                return Path(tmp) / 'selection.json'
            with patch.object(mod, 'run_group', side_effect=group), \
                 patch.object(mod.wf, 'best_config', side_effect=lambda configs: configs[0]), \
                 patch.object(mod, 'combined_config', return_value=dataclasses.replace(base, exp_id='T09')), \
                 patch.object(mod.wf, 'compare_inference', side_effect=inference), \
                 patch.object(mod.wf, 'lock_selection', side_effect=lock), \
                 patch.object(mod.wf, 'run_final', side_effect=lambda *a, **k: events.append('final')), \
                 patch.object(mod.wf, 'export_results', side_effect=lambda *a: events.append('export')):
                mod.run_all(base, allow_final_test=True)
            self.assertEqual(events[0], ['B01', 'B02', 'B03', 'B04', 'B05'])
            self.assertEqual(events[1], [f'T{i:02}' for i in range(9)])
            self.assertEqual(events[2], ['T09'])
            self.assertEqual(events[3:], ['inference', 'lock', 'final', 'export'])

    def test_existing_lock_resumes_final_without_reselection(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = mod.wf.base_config(tmp)
            folder = Path(base.out_dir)
            folder.mkdir(parents=True)
            cfg = dataclasses.replace(base, exp_id='T06')
            (folder / 'selection.json').write_text(json.dumps({
                'config': dataclasses.asdict(cfg), 'method': 'temperature', 'selection_split': 'val'}))
            (folder / 'full_baseline.json').write_text(json.dumps(dataclasses.asdict(base)))
            with patch.object(mod, 'run_group') as group, \
                 patch.object(mod.wf, 'compare_inference') as inference, \
                 patch.object(mod.wf, 'run_final') as final, \
                 patch.object(mod.wf, 'export_results'):
                mod.run_all(base, allow_final_test=True)
                group.assert_not_called()
                inference.assert_not_called()
                final.assert_called_once()

    def test_incomplete_run_enables_resume_without_enabling_test(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = mod.wf.base_config(tmp)
            folder = mod.wf.run_dir(cfg)
            folder.mkdir(parents=True)
            (folder / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg)))
            (folder / 'last.pt').write_bytes(b'checkpoint fixture')
            with patch.object(mod.wf, 'run_experiments') as runs:
                mod.run_group([cfg])
                restored = runs.call_args.args[0][0]
                self.assertTrue(restored.resume)
                self.assertFalse(restored.save_test_predictions)

    def test_final_reuses_baseline_trained_with_resume_enabled(self):
        mod = self.module()
        import numpy as np
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            base = mod.wf.base_config(tmp)
            selection = Path(tmp) / 'selection.json'
            selection.write_text(json.dumps({'config': dataclasses.asdict(base), 'method': 'single'}))
            for seed in (0, 1, 2):
                b = dataclasses.replace(base, seed=seed, resume=True)
                folder = mod.wf.run_dir(b)
                folder.mkdir(parents=True)
                (folder / 'config.json').write_text(json.dumps(dataclasses.asdict(b)))
                (folder / 'summary.json').write_text('{}')
                f = dataclasses.replace(base, exp_id='F01', seed=seed)
                folder = mod.wf.run_dir(f)
                folder.mkdir(parents=True)
                (folder / 'config.json').write_text(json.dumps(dataclasses.asdict(f)))
                pred = Path(base.pred_dir) / f'F01_seed{seed}_test.csv'
                pred.parent.mkdir(parents=True, exist_ok=True)
                pred.write_text('fixture')
            ev = Mock()
            ev.compute_metrics.return_value = dict(macro_f1=.9, top1=.9, ece=.1, balanced_acc=.9)
            with patch.object(mod.wf, '_import_eval', return_value=ev), \
                 patch.object(mod.wf, 'load_state', return_value=(object(), 0, .9)), \
                 patch.object(mod.wf.ds, 'load_split', return_value=(None, None, None)), \
                 patch.object(mod.wf, '_loader'), \
                 patch.object(mod.wf, 'predict_method', return_value=(['a.jpg'], np.array([0]), np.ones((1, 9)) / 9)), \
                 patch.object(mod.wf, 'recover_baseline'), \
                 patch.object(mod.wf, 'recover_final'):
                mod.wf.run_final(selection, base)
            self.assertEqual(ev.save_predictions.call_count, 3)

    def test_failure_records_stage_and_prevents_final(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = mod.wf.base_config(tmp)
            with patch.object(mod, 'run_group', side_effect=RuntimeError('fixture training failure')), \
                 patch.object(mod.wf, 'run_final') as final:
                with self.assertRaisesRegex(RuntimeError, 'fixture training failure'):
                    mod.run_all(base, allow_final_test=True)
                final.assert_not_called()
            state = json.loads((Path(base.out_dir) / 'full_status.json').read_text())
            self.assertEqual((state['stage'], state['status']), ('backbones', 'failed'))

    def test_test_outputs_without_lock_prevent_reselection(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = mod.wf.base_config(tmp)
            path = Path(base.pred_dir) / 'F01_seed0_test.csv'
            path.parent.mkdir(parents=True)
            path.write_text('fixture')
            with patch.object(mod, 'run_group') as group:
                with self.assertRaisesRegex(ValueError, 'without a selection lock'):
                    mod.run_all(base, allow_final_test=True)
                group.assert_not_called()

    def test_full_bootstrap_finds_separate_datasets_without_repo(self):
        import csv
        notebook = json.loads((ROOT / 'code/lab_day2_kaggle_full.ipynb').read_text(encoding='utf-8'))
        cell = next(c for c in notebook['cells'] if c['metadata'].get('tags') == ['bootstrap'])
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            labels, images = parent / 'input/labels', parent / 'input/images'
            labels.mkdir(parents=True)
            images.mkdir(parents=True)
            for name in ('labels', 'train_subset0', 'val_subset0', 'test_subset0'):
                with (labels / (name + '.csv')).open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['Filename', 'Label'])
                    writer.writerow(['fixture.jpg', 0])
            (images / 'fixture.jpg').write_bytes(b'fixture')
            env = {'KAGGLE_INPUT_ROOT': parent / 'input', 'LAB_WORK_ROOT': parent / 'working',
                   'INSTALL_MISSING_DEPENDENCIES': False}
            exec(compile(''.join(cell['source']), '<full-bootstrap>', 'exec'), env)
            self.assertEqual(env['KAGGLE_IMAGES_DIR'], images)
            self.assertEqual(env['KAGGLE_LABELS_DIR'], labels)
            self.assertEqual((parent / 'working/eval.py').read_bytes(), (ROOT / 'eval.py').read_bytes())
            self.assertTrue((parent / 'working/code/full_workflow.py').is_file())

    def test_full_notebook_compiles_and_embeds_original_eval(self):
        path = ROOT / 'code/lab_day2_kaggle_full.ipynb'
        self.assertTrue(path.exists(), 'full notebook is missing')
        notebook = json.loads(path.read_text(encoding='utf-8'))
        import ast
        payload = None
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            text = ''.join(cell['source'])
            tree = ast.parse(text)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'BUNDLED_FILES' for t in node.targets
                ):
                    payload = ast.literal_eval(node.value)
        self.assertIsNotNone(payload)
        self.assertEqual(payload['eval.py'].encode('utf-8'), (ROOT / 'eval.py').read_bytes())
        self.assertIn('code/full_workflow.py', payload)
        for name, content in payload.items():
            if name.endswith('.py'):
                compile(content, name, 'exec')
        self.assertTrue(any('run_all(BASE' in ''.join(c['source']) for c in notebook['cells']))


if __name__ == '__main__':
    unittest.main()
