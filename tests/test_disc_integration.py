"""Synthetic model outputs exercise disc provenance and actual report fusion."""
from array import array
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mri_disc import contract as disc
from mri_fusion.evidence import add_disc_evidence, build_evidence
from mri_fusion.evidence_report import fusion_lines
from mri_fusion.runner import run_fusion
from mri_root_impact.contracts import load_document, manifest_digest, validate_bundle
from tests.fusion_fixtures import build_case, case_by_id
from tests.test_fusion_reporting import _ReportHTML


class DiscIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='disc-integration-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.inputs = self.root / 'inputs'; self.inputs.mkdir()
        self.spine = self.root / 'spine'; self.spine.mkdir()
        self.output = self.root / 'disc'; self.output.mkdir()
        self.study, results = copy.deepcopy(build_case(case_by_id('F001-13'))[:2])
        self.localization, self.stenosis = results
        self.study_path = self.inputs / 'study.json'; disc.write(self.study_path, self.study)
        self.assets = {'source_commit': 'a'*40, 'weights': {'grading/ckpt1.pt': {'sha256': 'b'*64}}}
        self.bindings = {str(self.study_path): disc.file_sha256(self.study_path)}
        self.ref = self.study['series'][0]
        self.image_ref = {'series_id': self.ref['series_id'], 'image_id': self.ref['images'][0]['image_id'], 'frame_index':0}
        self.geometry = {'slices':[{'image_ref':self.image_ref}]}
        self.native = [{'predicted_label':label,'average_polygon':[[0,y],[1,y],[1,y+1],[0,y+1]],
                        'slice_nos':[0], 'polys':[]} for label,y in [('L5',10),('L4',0)]]
        disc.write(self.spine / 'native.json', {'detections':self.native})
        disc.write(self.spine / 'spatial.json', {'geometry':self.geometry})
        disc.write(self.spine / 'localization.json', self.localization)
        self.document = {'schema_version':'1.0','kind':'spinenet_binary_disc','study_id':self.study['study_id'],
                         'study_digest':manifest_digest(self.study),'scale_id':disc.SCALE,'predictions':[]}
        for level in disc.LEVELS:
            row = {'native_level':level,'native_class':None,'logits':None,'side':None,'size_mm':None,
                   'morphology':None,'zone':None,'source_samples':[],'crop':None}
            if level == 'L4-L5':
                row.update(native_class=1,logits=[-1.0,1.0],source_samples=disc.source_samples(self.native,self.geometry,level),
                           crop={'file':'inputs/'+level+'.f64','shape':disc.SHAPE,'image':'images/'+level+'.png','image_plane':4})
            self.document['predictions'].append(row)
        (self.output/'inputs').mkdir(); (self.output/'images').mkdir()
        raw = array('d',[0.0]*(9*112*224)).tobytes()
        (self.output/'inputs/L4-L5.f64').write_bytes(raw)
        (self.output/'images/L4-L5.png').write_bytes(disc.png_from_input(raw))
        self.save()
        for target,value in [('validate_worker_outputs',None),('verify_assets',self.assets),('input_hashes',self.bindings)]:
            patcher = mock.patch.object(disc,target,return_value=value); patcher.start(); self.addCleanup(patcher.stop)

    def save(self):
        for name in ('disc.json','result.json','run.json'):
            (self.output/name).unlink(missing_ok=True)
        disc.write(self.output/'disc.json',self.document)
        disc.write(self.output/'result.json',disc.build_result(self.study,self.localization,self.document,self.assets))
        artifacts = ['disc.json','result.json','inputs/L4-L5.f64','images/L4-L5.png']
        disc.write(self.output/'run.json',{'schema_version':'1.0','kind':'spinenet_binary_disc_run','status':'completed',
             'study_digest':manifest_digest(self.study),'inference_performed':True,'source_bytes_unchanged':True,
             'code_files_sha256':disc.code_hashes(),'input_files_sha256':self.bindings,'source':'/synthetic/model',
             'assets':self.assets,'compatibility':['numpy.bool=numpy.bool_ in isolated worker'],
             'artifacts_sha256':{n:disc.file_sha256(self.output/n) for n in artifacts}})

    def check(self):
        return disc.validate_run(self.study_path,self.spine,self.output)

    def test_real_contract_retains_native_binary_and_unknown_geometry_fields(self):
        document,result,files = self.check()
        validate_bundle(self.study,[self.localization,result])
        self.assertEqual(len(result['assessments']),5)
        row = result['assessments'][3]
        self.assertEqual(row['observation']['label'],'present')
        self.assertEqual(row['status'],'indeterminate')
        self.assertEqual(row['target']['side'],'unknown')
        self.assertEqual(row['observation']['measurements'],[])
        self.assertTrue(all(r['observation'] is None and r['status']=='not_run' for r in result['assessments'][:3]))
        self.assertIn(str(self.output/'images/L4-L5.png'),files)

    def test_absent_native_class_is_a_prediction_not_a_normal_or_clear_root(self):
        self.document['predictions'][3].update(native_class=0,logits=[1.0,-1.0]); self.save()
        _,result,_ = self.check()
        row=result['assessments'][3]
        self.assertEqual(row['observation']['label'],'not_detected')
        self.assertEqual(row['status'],'indeterminate')

    def test_invented_side_size_type_or_nonbinary_class_cannot_enter_report(self):
        original=copy.deepcopy(self.document)
        for field,value in [('side','left'),('size_mm',4),('morphology','extrusion'),('zone','foraminal'),('native_class',True),('native_class',2)]:
            self.document=copy.deepcopy(original); self.document['predictions'][3][field]=value; self.save()
            with self.subTest(field=field,value=value),self.assertRaises(ValueError): self.check()

    def test_forged_native_class_or_wrong_source_frames_is_rejected_even_with_updated_hashes(self):
        original=copy.deepcopy(self.document)
        for change in [{'native_class':0},{'logits':[float('inf'),0]},{'source_samples':[self.image_ref]*9}]:
            self.document=copy.deepcopy(original);self.document['predictions'][3].update(change)
            if 'logits' in change:
                with self.assertRaises(ValueError): self.save()
            else:
                self.save()
                with self.assertRaises(ValueError): self.check()

    def test_missing_crop_does_not_silently_erase_an_available_level(self):
        self.document['predictions'][3].update(native_class=None,logits=None,source_samples=[],crop=None);self.save()
        with self.assertRaises(ValueError):self.check()

    def test_malformed_prediction_row_is_a_validation_error(self):
        doc=copy.deepcopy(self.document);doc['predictions'][0]=[]
        path=self.output/'disc.json';path.unlink();disc.write(path,doc)
        meta=load_document(self.output/'run.json');meta['artifacts_sha256']['disc.json']=disc.file_sha256(path)
        (self.output/'run.json').unlink();disc.write(self.output/'run.json',meta)
        with self.assertRaises(ValueError):self.check()

    def test_changed_crop_png_or_symlink_is_refused(self):
        path=self.output/'images/L4-L5.png';original=path.read_bytes()
        path.write_bytes(original+b'changed')
        with self.assertRaises(ValueError):self.check()
        path.unlink();(self.root/'other.png').write_bytes(original);path.symlink_to(self.root/'other.png')
        with self.assertRaises(ValueError):self.check()

    def test_rehashed_picture_must_still_match_exact_model_input_plane(self):
        path=self.output/'images/L4-L5.png';raw=array('d',[1.0]*(9*112*224)).tobytes()
        path.write_bytes(disc.png_from_input(raw));self.save()
        with self.assertRaisesRegex(ValueError,'illustration'):self.check()

    def test_rehashed_c001_must_match_native_grading(self):
        path=self.output/'result.json';result=load_document(path)
        result['assessments'][3]['observation']['label']='not_detected'
        path.unlink();disc.write(path,result)
        meta=load_document(self.output/'run.json');meta['artifacts_sha256']['result.json']=disc.file_sha256(path)
        (self.output/'run.json').unlink();disc.write(self.output/'run.json',meta)
        with self.assertRaisesRegex(ValueError,'C001'):self.check()

    def test_crop_renderer_rejects_invalid_float_pixels(self):
        for value in (float('nan'),-1.0,1.1):
            raw=array('d',[value]*(9*112*224)).tobytes()
            with self.subTest(value=value),self.assertRaises(ValueError):disc.png_from_input(raw)

    def test_changed_disc_code_or_run_status_is_rejected(self):
        path=self.output/'run.json';meta=load_document(path)
        for change in [{'status':'error'},{'inference_performed':False},{'code_files_sha256':{}}]:
            path.unlink();disc.write(path,{**meta,**change})
            with self.subTest(change=change),self.assertRaises(ValueError):self.check()

    def frozen(self,study,*args):
        return [('spine-localization',self.localization,manifest_digest(self.localization)),
                ('rsna-stenosis',self.stenosis,manifest_digest(self.stenosis))]

    def test_actual_fusion_and_self_contained_report_include_disc_without_promoting_roots(self):
        output=self.root/'fusion'
        kwargs={'spinenet_run':self.spine,'rsna_run':self.root/'rsna','crosscheck_run':self.root/'crosscheck','disc_run':self.output}
        with mock.patch('mri_fusion.runner._frozen_runs',side_effect=self.frozen):
            run_fusion(self.study_path,output,**kwargs)
            before={str(p):p.read_bytes() for p in output.iterdir()}
            run_fusion(self.study_path,output,**kwargs,verify_only=True)
            self.assertEqual(before,{str(p):p.read_bytes() for p in output.iterdir()})
        report=load_document(output/'fusion.json');evidence=load_document(output/'report-evidence.json')
        self.assertEqual(report['root_candidates'],[])
        self.assertEqual(report['coverage']['indeterminate'],20)
        self.assertEqual(len(report['raw_disc_evidence']),5)
        self.assertEqual(report['herniation_context'],'not_evaluated')
        self.assertEqual(evidence['disc_analysis_status'],'spinenet_binary_unreviewed')
        html=_ReportHTML((output/'report.html').read_text())
        self.assertEqual(len(html.h2),4)
        self.assertIn('признак грыжи',html.element('disc-L4-L5'))
        self.assertIn('сторона неизвестна',html.element('fusion'))
        self.assertIn('data:image/png;base64,',(output/'report.html').read_text())
        self.assertNotIn('левосторонняя грыжа',html.visible_text())

    def test_disc_without_stenosis_cannot_name_an_affected_root(self):
        with mock.patch('mri_fusion.runner._frozen_runs',return_value=[self.frozen(self.study)[0]]):
            run_fusion(self.study_path,self.root/'only-disc',spinenet_run=self.spine,rsna_run=self.root/'rsna',
                       crosscheck_run=self.root/'crosscheck',disc_run=self.output)
        report=load_document(self.root/'only-disc/fusion.json');bundle=load_document(self.root/'only-disc/report-evidence.json')
        lines=fusion_lines(report,bundle)
        self.assertEqual(len(lines),1)
        self.assertIn('корешок по этой находке не определён',lines[0][1])
        self.assertEqual(report['root_candidates'],[])


if __name__=='__main__':unittest.main()
