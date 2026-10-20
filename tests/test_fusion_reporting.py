"""Verify readable MRI explanations preserve the checked fusion semantics."""

import copy
from html.parser import HTMLParser
import re
import unittest

from mri_fusion.engine import fuse
from tests.fusion_fixtures import build_case, case_by_id, document_sha256, rebind_review


class _ReportHTML(HTMLParser):
    """Collect visible trace text and resource references without a browser."""

    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.branch_text = {}
        self.tags = []
        self.attributes = []
        self.ids = set()
        self.active_branch = None
        self.detail_depth = 0
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend((tag, key, value) for key, value in attrs)
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "details":
            if self.active_branch is not None:
                self.detail_depth += 1
            elif attributes.get("id", "").startswith("branch-"):
                self.active_branch = attributes["id"][len("branch-"):]
                self.detail_depth = 1
                self.branch_text[self.active_branch] = []

    def handle_endtag(self, tag):
        if tag == "details" and self.active_branch is not None:
            self.detail_depth -= 1
            if self.detail_depth == 0:
                self.active_branch = None

    def handle_data(self, data):
        self.text.append(data)
        if self.active_branch is not None:
            self.branch_text[self.active_branch].append(data)

    def visible_text(self):
        return " ".join(" ".join(self.text).split())

    def branch(self, branch_id):
        return " ".join(" ".join(self.branch_text[branch_id]).split())


def _fusion(case):
    study, results, artifacts, review, review_sha = build_case(case)
    return fuse(study, results, artifacts, review_context=review, review_sha256=review_sha)


class FusionReportingTests(unittest.TestCase):
    def render(self, report):
        from mri_fusion.reporting import render_html, render_markdown
        return render_markdown(report), render_html(report)

    def test_renderers_are_deterministic_and_do_not_modify_checked_report(self):
        report = _fusion(case_by_id("F001-01"))
        before = copy.deepcopy(report)
        markdown, html = self.render(report)
        self.assertEqual(report, before)
        self.assertEqual(self.render(report), (markdown, html))
        self.assertRegex(markdown, r"[А-Яа-я]")
        self.assertTrue(html.lstrip().lower().startswith("<!doctype html"))

    def test_empty_input_does_not_claim_twenty_evaluated_anatomical_sites(self):
        study, _, _, _, _ = build_case(case_by_id("F001-01"))
        report = fuse(study, [], [])
        markdown, html = self.render(report)
        for text in (markdown, _ReportHTML(html).visible_text()):
            self.assertIn("исходные оценки получены для 0 из 20", text)
            self.assertIn("Предусмотрено 20", text)
            self.assertNotIn("Оценено сочетаний уровня, стороны и зоны: 20", text)

    def test_positive_root_retains_missing_second_anatomical_site_in_summary(self):
        report = _fusion(case_by_id("F001-01"))
        _, html = self.render(report)
        parsed = _ReportHTML(html)
        self.assertIn("roots", parsed.ids)
        summary = html.split('<section id="roots">', 1)[1].split('</section>', 1)[0]
        self.assertIn("L5", summary)
        self.assertIn("остаются неопределёнными: 1", summary)
        self.assertIn("L4-L5", summary)

    def test_unaccepted_disc_measurements_are_not_relabelled_as_probabilities(self):
        from mri_fusion.contracts import artifact_binding
        study, results, artifacts, review, _ = build_case(case_by_id("F001-30"))
        disc = next(result for result in results if result["component"] == "disc_analysis")
        disc["assessments"][0]["observation"]["measurements"] = [
            {"name": "p_size", "value": 12, "unit": "mm"}]
        for index, binding in enumerate(artifacts):
            if binding["result_id"] == disc["result_id"]:
                artifacts[index] = artifact_binding(binding["artifact_id"], disc, document_sha256(disc))
        report = fuse(study, results, artifacts, review, rebind_review(review, artifacts))
        markdown, html = self.render(report)
        self.assertNotIn("1200.00%", html)
        self.assertIn("не интерпретируются как вероятности", _ReportHTML(html).visible_text())
        self.assertIn("p_size", markdown)

    def test_severe_in_unaccepted_scale_is_retained_without_medical_translation(self):
        report = _fusion(case_by_id("F001-19"))
        _, html = self.render(report)
        trace = _ReportHTML(html).branch("subarticular-L4-L5-left")
        finding = trace.split("2. Обязательные условия", 1)[0]
        self.assertIn("непринятая шкала", finding)
        self.assertIn("исходная метка «severe»", finding)
        self.assertNotIn("выраженный стеноз", finding)
        self.assertEqual(report["root_candidates"], [])

    def test_current_control_like_uncertainty_has_twenty_explained_branches(self):
        case = copy.deepcopy(case_by_id("F001-28"))
        for source in case["inputs"]["branches"]:
            source["status"] = "indeterminate"
        case["inputs"]["review_context"] = None
        report = _fusion(case)
        markdown, html = self.render(report)
        parsed = _ReportHTML(html)
        self.assertEqual(len(parsed.branch_text), 20)
        self.assertEqual(set(parsed.branch_text), {row["branch_id"] for row in report["branches"]})
        self.assertRegex(parsed.visible_text().lower(), r"неопредел|недостат")
        self.assertRegex(parsed.visible_text().lower(), r"нумерац")
        self.assertRegex(parsed.visible_text().lower(), r"совмещ|регистрац")
        self.assertRegex(markdown.lower(), r"неопредел|недостат")
        for branch_id in parsed.branch_text:
            trace = parsed.branch(branch_id)
            self.assertRegex(trace.lower(), r"неопредел|недостат")
            self.assertNotRegex(trace, r"кореш\w*\s+(?:L[1-5]|S1)\b")
        self.assertEqual(report["root_candidates"], [])

    def test_reviewed_severe_left_subarticular_source_explains_traversing_l5(self):
        report = _fusion(case_by_id("F001-01"))
        _, html = self.render(report)
        trace = _ReportHTML(html).branch("subarticular-L4-L5-left")
        self.assertIn("L4-L5", trace)
        self.assertRegex(trace.lower(), r"лев|слев")
        self.assertRegex(trace.lower(), r"проходящ")
        self.assertRegex(trace, r"кореш\w*\s+L5\b")
        self.assertRegex(trace.lower(), r"выражен|severe")
        self.assertRegex(trace.lower(), r"потенциальн")
        self.assertIn("synthetic-stenosis-item-00", trace)
        self.assertIn("assessment-item-00", trace)

    def test_l5_s1_right_foraminal_source_explains_exiting_l5(self):
        case = copy.deepcopy(case_by_id("F001-04"))
        case["inputs"]["branches"][0]["side"] = "right"
        report = _fusion(case)
        _, html = self.render(report)
        trace = _ReportHTML(html).branch("foraminal-L5-S1-right")
        self.assertRegex(trace.lower(), r"прав|справ")
        self.assertRegex(trace.lower(), r"выходящ")
        self.assertRegex(trace, r"кореш\w*\s+L5\b")
        self.assertNotRegex(trace, r"\bL6\b")

    def test_normal_mild_has_limited_criterion_without_root_clearance(self):
        report = _fusion(case_by_id("F001-10"))
        markdown, html = self.render(report)
        trace = _ReportHTML(html).branch("subarticular-L4-L5-left")
        self.assertRegex(trace.lower(), r"критери")
        self.assertRegex(trace.lower(), r"normal|легк|лёгк|норм")
        self.assertRegex((markdown + " " + _ReportHTML(html).visible_text()).lower(), r"не исключ|не означ")
        self.assertIsNone(next(row for row in report["branches"]
                               if row["branch_id"] == "subarticular-L4-L5-left")["candidate_root"])
        self.assertIs(report["global_root_clearance"], False)

    def test_moderate_grade_does_not_turn_into_positive_or_normal(self):
        report = _fusion(case_by_id("F001-09"))
        _, html = self.render(report)
        trace = _ReportHTML(html).branch("subarticular-L4-L5-left")
        self.assertRegex(trace.lower(), r"умерен|moderate")
        self.assertRegex(trace.lower(), r"неопредел|недостат")
        self.assertRegex(trace.lower(), r"порог|критери")
        self.assertEqual(report["root_candidates"], [])

    def test_model_class_distribution_is_not_presented_as_clinical_confidence(self):
        case = copy.deepcopy(case_by_id("F001-01"))
        case["inputs"]["producer_method"] = "model"
        case["inputs"]["branches"][0]["probabilities"] = [0.05, 0.10, 0.85]
        report = _fusion(case)
        markdown, html = self.render(report)
        trace = _ReportHTML(html).branch("subarticular-L4-L5-left")
        self.assertRegex(trace, r"85(?:[.,]0)?\s*%|0[.,]85")
        self.assertRegex(trace.lower(), r"класс")
        self.assertRegex(trace.lower(), r"модел")
        self.assertRegex((trace + " " + markdown).lower(), r"не (?:явля\w* )?(?:клиничес|вероятност).*|не .*клиничес")
        self.assertIn("fictitious-rsna-model", trace)

    def test_missing_and_unknown_native_evidence_remains_visible(self):
        for case_id in ("F001-11", "F001-18", "F001-33"):
            with self.subTest(case_id=case_id):
                report = _fusion(case_by_id(case_id))
                markdown, html = self.render(report)
                text = _ReportHTML(html).visible_text()
                self.assertIn("synthetic-stenosis-item-00", text)
                self.assertIn("assessment-item-00", text)
                self.assertRegex(text.lower(), r"неопредел|неизвест|не выполн|недостат")
                self.assertEqual(report["root_candidates"], [])
                if case_id == "F001-33":
                    self.assertIn("T12-L1", text)
                    self.assertIn("T12-L1", markdown)
                if case_id in ("F001-18", "F001-33"):
                    self.assertIn("unresolved", _ReportHTML(html).ids)
                    self.assertRegex(text.lower(), r"выражен|severe")

    def test_unsupported_disc_evidence_is_visible_without_causal_attribution(self):
        report = _fusion(case_by_id("F001-30"))
        markdown, html = self.render(report)
        text = _ReportHTML(html).visible_text()
        self.assertIn("synthetic-disc-context", text)
        self.assertIn("fixture:unaccepted-disc-scale:v1", text)
        self.assertIn("discs", _ReportHTML(html).ids)
        self.assertRegex((text + " " + markdown).lower(), r"грыж")
        self.assertRegex((text + " " + markdown).lower(), r"не припис|не установ|не определ|не оцени")
        self.assertEqual(report["etiology"], "unassigned")
        self.assertEqual(len(report["root_candidates"]), 1)

    def test_untrusted_native_label_is_escaped_in_self_contained_html(self):
        case = copy.deepcopy(case_by_id("F001-20"))
        payload = '<script>alert("native")</script><img src="https://example.invalid/mri" onerror="alert(1)">'
        case["inputs"]["branches"][0]["label"] = payload
        report = _fusion(case)
        markdown, html = self.render(report)
        parsed = _ReportHTML(html)
        self.assertNotIn(payload, html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn(payload, parsed.visible_text())
        self.assertNotIn("<script>", markdown)
        self.assertFalse({"script", "iframe", "object", "embed", "base"} & set(parsed.tags))
        for tag, name, value in parsed.attributes:
            self.assertFalse(name.lower().startswith("on"), (tag, name))
            if name.lower() == "src" or (tag == "link" and name.lower() == "href"):
                self.assertTrue((value or "").startswith("data:"), (tag, name, value))
        self.assertNotRegex(html, r"@import\s|url\(\s*['\"]?https?://")


if __name__ == "__main__":
    unittest.main()
