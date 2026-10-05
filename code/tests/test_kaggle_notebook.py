"""The uploadable notebook must bootstrap without a repo beside it."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class KaggleNotebookTests(unittest.TestCase):
    def bootstrap(self, input_dir, output_dir):
        notebook_path = ROOT / 'code' / 'lab_day2_kaggle.ipynb'
        self.assertTrue(notebook_path.exists(), 'No notebook-only Kaggle artifact exists')
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('tags') == ['bootstrap'])
        env = {'KAGGLE_INPUT_ROOT': Path(input_dir), 'LAB_WORK_ROOT': Path(output_dir),
               'INSTALL_MISSING_DEPENDENCIES': False}
        exec(compile(''.join(cell['source']), '<bootstrap>', 'exec'), env)
        return env

    def fixture(self, path):
        # The CSV and image folders may be in two distinct Kaggle datasets.
        labels = path / 'labels-dataset' / 'data' / 'labels'
        images = path / 'image-dataset' / 'images'
        labels.mkdir(parents=True)
        images.mkdir(parents=True)
        for name in ('labels', 'train_subset0', 'val_subset0', 'test_subset0'):
            with (labels / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Filename', 'Label'])
                writer.writerow(['a.jpg', 0])
        (images / 'a.jpg').write_bytes(b'fixture')
        return labels, images

    def test_bootstrap_unpacks_modules_and_finds_external_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            labels, images = self.fixture(parent / 'input')
            env = self.bootstrap(parent / 'input', parent / 'working')
            self.assertEqual(env['KAGGLE_LABELS_DIR'], labels)
            self.assertEqual(env['KAGGLE_IMAGES_DIR'], images)
            self.assertEqual((parent / 'working/eval.py').read_bytes(), (ROOT / 'eval.py').read_bytes())
            self.assertTrue((parent / 'working/code/train.py').exists())
            self.assertTrue((parent / 'working/code/lab_workflow.py').exists())

    def test_missing_labels_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / 'input').mkdir()
            with self.assertRaisesRegex(FileNotFoundError, 'labels|CSV|Input'):
                self.bootstrap(parent / 'input', parent / 'working')


if __name__ == '__main__':
    unittest.main()
