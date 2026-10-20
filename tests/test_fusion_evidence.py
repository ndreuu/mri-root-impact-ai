"""Illustrations must be real model inputs; anatomy remains conditional."""

import base64
import copy
import hashlib
import struct
import tempfile
from pathlib import Path
import unittest
import zlib

from mri_fusion.contracts import FusionContractError
from mri_fusion.evidence import anatomical_facts, build_evidence
from mri_fusion.engine import fuse
from mri_fusion.reporting import render_html
from mri_fusion.runner import _read_report_image
from tests.fusion_fixtures import build_case, case_by_id
from tests.test_fusion_reporting import _ReportHTML


def report(case_id="F001-13"):
    study, results, artifacts, review, sha = build_case(case_by_id(case_id))
    return fuse(study, results, artifacts, review, sha)


def picture():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 48, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\x80" * 16) * 48)) + chunk(b"IEND", b""))


def inputs(r):
    ref = {"series_id": "ax-t2", "image_id": "ax-t2-image", "frame_index": 0}
    rois = [{"roi_id": "roi-"+str(i), "level": "L4-L5", "side": "left", "sequence": "axial_t2",
             "image_ref": ref, "source_crop_box": [1+i, 1, 10+i, 10], "horizontal_flip": False}
            for i in range(5)]
    spatial = {"study_id": r["study_id"], "study_digest": r["study_digest"], "rois": rois,
               "inputs": [{"level": "L4-L5", "side": "left", "roi_ids": {"axial_t2": [x["roi_id"] for x in rois]}}],
               "keypoints": [{"sequence": "axial_t2", "label": "L4-L5", "image_ref": ref}]}
    raw = picture()
    sha = hashlib.sha256(raw).hexdigest()
    overlays = [{"file": "overlays/frame-000.png", "image_ref": ref, "header_pixels": 32,
                 "sha256": sha, "width": 16, "height": 48}]
    def reader(_):
        return {"source_file": "/synthetic/overlay.png", "sha256": sha, "width": 16, "height": 48,
                "data_uri": "data:image/png;base64," + base64.b64encode(raw).decode()}
    return spatial, overlays, reader


class EvidenceTests(unittest.TestCase):
    def test_unreviewed_severe_explains_anatomy_without_promoting_fusion(self):
        r = report()
        before = copy.deepcopy(r)
        facts = anatomical_facts(r)
        self.assertEqual(len(facts), 20)
        f = next(f for f in facts if f["branch_id"] == "subarticular-L4-L5-left")
        self.assertEqual(f["conditional_anatomy"], {"root": "L5", "side": "left", "role": "traversing"})
        self.assertEqual(f["formal_decision"], "indeterminate")
        self.assertIsNone(f["verified_candidate"])
        self.assertEqual(r, before)

    def test_l5_s1_uses_l5_and_s1_not_l6(self):
        for case_id, zone, root in (("F001-03", "subarticular", "S1"), ("F001-04", "foraminal", "L5")):
            fact = next(f for f in anatomical_facts(report(case_id))
                        if f["native_target"]["level"] == "L5-S1" and f["native_target"]["zone"] == zone
                        and f["native_grade"] is not None)
            self.assertEqual(fact["conditional_anatomy"]["root"], root)

    def test_moderate_and_normal_mild_never_become_positive_candidates(self):
        for case_id, effect in (("F001-09", "moderate_relation_only"), ("F001-10", "no_severe_criterion")):
            fact = next(f for f in anatomical_facts(report(case_id)) if f["native_grade"] is not None)
            self.assertEqual(fact["rule_effect"], effect)
            self.assertIsNone(fact["verified_candidate"])
            self.assertFalse(fact["herniation_assessed_by_this_rule"])

    def test_unaccepted_scale_has_no_conditional_root(self):
        facts = anatomical_facts(report("F001-19"))
        self.assertTrue(all(f["conditional_anatomy"] is None for f in facts))
        self.assertEqual(facts[-2]["rule_effect"], "unresolved_source")

    def test_frame_is_selected_from_exact_ordered_input_and_roi(self):
        r = report()
        spatial, overlays, reader = inputs(r)
        before = copy.deepcopy(spatial)
        bundle = build_evidence(r, spatial, overlays, reader)
        self.assertEqual(len(bundle["stage_views"]), 1)
        view = bundle["stage_views"][0]
        self.assertEqual(view["roi_id"], "roi-2")
        self.assertEqual(view["model_sample_index"], 2)
        self.assertEqual(view["sequence"], "axial_t2")
        self.assertEqual(view["source_crop_box"], [3, 1, 12, 10])
        self.assertEqual(spatial, before)
        self.assertFalse(bundle["disc_views"][0]["herniation_detection_performed"])
        self.assertFalse(view["overlay_is_pathology_segmentation"])

    def test_foreign_frame_wrong_side_or_out_of_bounds_crop_is_rejected(self):
        r = report()
        for mutation in (lambda s: s["rois"][2]["image_ref"].update(image_id="foreign"),
                         lambda s: s["rois"][2].update(side="right"),
                         lambda s: s["rois"][2].update(source_crop_box=[-1, 0, 12, 10])):
            spatial, overlays, reader = inputs(r)
            mutation(spatial)
            with self.assertRaises(FusionContractError):
                build_evidence(r, spatial, overlays, reader)

    def test_images_and_facts_precede_quality_without_external_image_fetches(self):
        r = report()
        bundle = build_evidence(r, *inputs(r))
        document = render_html(r, bundle)
        parsed = _ReportHTML(document)
        self.assertTrue({"foraminal-stage", "subarticular-stage", "disc-stage", "facts", "formal-quality"} <= parsed.ids)
        self.assertLess(document.index('id="subarticular-stage"'), document.index('id="facts"'))
        self.assertLess(document.index('id="facts"'), document.index('id="formal-quality"'))
        images = [v for tag,k,v in parsed.attributes if tag == "image" and k == "href"]
        self.assertTrue(images)
        self.assertTrue(all(v.startswith("data:image/png;base64,") for v in images))
        self.assertIn("Если исходные метки верны", parsed.visible_text())
        self.assertIn("не результат обнаружения грыжи", parsed.visible_text())

    def test_network_image_cannot_enter_self_contained_report(self):
        r = report()
        bundle = build_evidence(r, *inputs(r))
        next(iter(bundle["images"].values()))["data_uri"] = "https://example.invalid/mri.png"
        with self.assertRaises(FusionContractError):
            render_html(r, bundle)

    def test_png_reader_binds_actual_bytes_and_refuses_changed_images(self):
        r = report()
        _, overlays, _ = inputs(r)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / overlays[0]["file"]
            path.parent.mkdir()
            path.write_bytes(picture())
            files = {str(path.resolve()): overlays[0]["sha256"]}
            image = _read_report_image(root, overlays[0], files)
            self.assertEqual(base64.b64decode(image["data_uri"].split(",",1)[1]), picture())
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaises(FusionContractError):
                _read_report_image(root, overlays[0], files)


if __name__ == "__main__":
    unittest.main()
