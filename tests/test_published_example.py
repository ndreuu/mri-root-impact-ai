"""The shipped input and report must remain usable and bound to the same case."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('published_example', ROOT/'scripts/example.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class PublishedExampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mri-published-example-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_complete_case_and_report_are_consistent_and_keep_uncertainty(self):
        study, inventory = example.checked_example()
        self.assertFalse(study['synthetic'])
        self.assertEqual([len(s['images']) for s in study['series']], [15,15,30])
        result = json.loads((example.EXAMPLE/'output/fusion.json').read_text())
        self.assertEqual(result['root_candidates'], [])
        self.assertEqual(result['coverage']['indeterminate'],20)
        self.assertTrue(all(Path(i['path']).is_file() for s in study['series'] for i in s['images']))
        self.assertEqual(len([n for n in inventory['files'] if n.endswith('.dcm')]),60)

    def test_preparation_makes_valid_absolute_input_without_changing_template(self):
        template = example.EXAMPLE/'input/study.template.json'
        original = template.read_bytes()
        result = example.prepare(self.directory/'prepared')
        study = json.loads(Path(result['study']).read_text())
        self.assertTrue(all(Path(i['path']).is_absolute() for s in study['series'] for i in s['images']))
        self.assertEqual(template.read_bytes(),original)
        self.assertFalse(result['inference_performed'])
        with self.assertRaises(ValueError):
            example.prepare(self.directory/'prepared')

    def test_changed_dicom_is_rejected_before_prepared_output_is_created(self):
        root=self.directory/'example';shutil.copytree(example.EXAMPLE,root)
        image=next((root/'input/dicom').rglob('*.dcm'));image.write_bytes(image.read_bytes()+b'changed')
        with self.assertRaises(ValueError):
            example.prepare(self.directory/'prepared',root)
        self.assertFalse((self.directory/'prepared').exists())

    def test_export_inventory_cannot_escape_to_an_external_file(self):
        for name in ['../private.dcm','/tmp/private.dcm','input/../../private.dcm','input\\private.dcm']:
            with self.subTest(path=name),self.assertRaises(ValueError):
                example.checked_path(example.EXAMPLE,name)

    def test_harness_allows_only_pinned_example_dicom_bytes(self):
        hs=importlib.util.spec_from_file_location('example_harness',ROOT/'scripts/harness.py')
        harness=importlib.util.module_from_spec(hs);hs.loader.exec_module(harness)
        source=next((example.EXAMPLE/'input/dicom').rglob('*.dcm'))
        name=source.relative_to(ROOT).as_posix()
        git_result=type('Result',(),{'returncode':0,'stdout':name+'\0data/private.dcm\0','stderr':''})()
        with mock.patch.object(harness,'git',return_value=git_result):
            self.assertEqual(harness.unsafe_candidates(),['data/private.dcm'])
        with mock.patch.object(harness,'git',return_value=git_result),mock.patch.object(harness,'read_json',return_value={}):
            self.assertEqual(harness.unsafe_candidates(),sorted([name,'data/private.dcm']))


if __name__ == '__main__':
    unittest.main()
