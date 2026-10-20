"""Compact physician views preserve native findings and conditional anatomy."""

import copy
from html.parser import HTMLParser
import unittest

from mri_fusion.engine import fuse
from mri_fusion.evidence import build_evidence
from tests.fusion_fixtures import build_case, case_by_id


class _ReportHTML(HTMLParser):
    """Collect visible text, IDs, and image resources without a browser.

    branch()/branch_text also support callers whose branch IDs now label table
    cells rather than expanded details.
    """

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.text, self.branch_text, self.tags, self.attributes = [], {}, [], []
        self.ids, self.element_text, self.h2 = set(), {}, []
        self._stack, self._heading = [], None
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend((tag, key, value) for key, value in attrs)
        identifier = dict(attrs).get("id")
        if identifier is not None:
            self.ids.add(identifier)
            self.element_text.setdefault(identifier, [])
            if identifier.startswith("branch-"):
                self.branch_text[identifier[len("branch-"):]] = self.element_text[identifier]
        if tag == "h2":
            self._heading = []
            self.h2.append(self._heading)
        if tag not in self.VOID:
            self._stack.append((tag, identifier))

    def handle_endtag(self, tag):
        if tag == "h2":
            self._heading = None
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                break

    def handle_data(self, data):
        if not any(tag in {"head", "style", "script"} for tag, _ in self._stack):
            self.text.append(data)
        if self._heading is not None:
            self._heading.append(data)
        for _, identifier in self._stack:
            if identifier is not None:
                self.element_text[identifier].append(data)

    @staticmethod
    def _join(values):
        return " ".join(" ".join(values).split())

    def visible_text(self):
        return self._join(self.text)

    def branch(self, branch_id):
        return self._join(self.branch_text[branch_id])

    def element(self, identifier):
        return self._join(self.element_text[identifier])

    def headings(self):
        return [self._join(value) for value in self.h2]


def _fusion(case):
    study, results, artifacts, review, review_sha = build_case(case)
    return fuse(study, results, artifacts, review_context=review, review_sha256=review_sha)


class FusionReportingTests(unittest.TestCase):
    def render(self, report, bundle=None):
        from mri_fusion.reporting import render_html, render_markdown
        return render_markdown(report, bundle), render_html(report, bundle)

    def test_renderers_are_deterministic_without_changing_formal_report_or_evidence(self):
        report = _fusion(case_by_id("F001-01"))
        bundle = build_evidence(report)
        before = copy.deepcopy((report, bundle))
        rendered = self.render(report, bundle)
        self.assertEqual(self.render(report, bundle), rendered)
        self.assertEqual((report, bundle), before)
        self.assertTrue(rendered[1].lstrip().lower().startswith("<!doctype html"))
        self.assertRegex(_ReportHTML(rendered[1]).visible_text().lower(), r"синтетичес|вымышлен")

    def test_exactly_four_ordered_sections_without_visible_json_trace(self):
        markdown, html = self.render(_fusion(case_by_id("F001-01")))
        parsed = _ReportHTML(html)
        self.assertEqual(parsed.headings(), ["1. Анализ межпозвонковых дисков", "2. Фораминальный стеноз",
                                             "3. Субартикулярный стеноз", "4. Fusion"])
        section_ids = ("discs", "foraminal-stage", "subarticular-stage", "fusion")
        self.assertTrue(set(section_ids) <= parsed.ids)
        self.assertEqual(sorted(section_ids, key=lambda value: html.index('id="' + value + '"')), list(section_ids))
        self.assertEqual(len([line for line in markdown.splitlines() if line.startswith("## ")]), 4)
        self.assertNotIn("formal-quality", parsed.ids)
        self.assertNotIn("pre", parsed.tags)
        self.assertNotIn("study_digest", parsed.visible_text())

    def test_all_five_disc_levels_remain_unassessed_without_ldh(self):
        report = _fusion(case_by_id("F001-27"))
        _, html = self.render(report)
        parsed = _ReportHTML(html)
        levels = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
        self.assertEqual({value for value in parsed.ids if value.startswith("disc-")},
                         {"disc-" + level for level in levels})
        for level in levels:
            self.assertRegex(parsed.element("disc-" + level).lower(), r"не оцен|не подключ|не определ")
        self.assertEqual(report["raw_disc_evidence"], [])
        self.assertEqual(report["herniation_context"], "not_evaluated")

    def test_stenosis_tables_keep_every_level_side_and_grade_in_its_own_cell(self):
        report = _fusion(case_by_id("F001-01"))
        _, html = self.render(report)
        parsed = _ReportHTML(html)
        self.assertEqual(set(parsed.branch_text), {branch["branch_id"] for branch in report["branches"]})
        self.assertEqual(len([key for key in parsed.branch_text if key.startswith("foraminal-")]), 10)
        self.assertEqual(len([key for key in parsed.branch_text if key.startswith("subarticular-")]), 10)
        self.assertRegex(parsed.branch("subarticular-L4-L5-left").lower(), r"выражен|severe")
        self.assertNotRegex(parsed.branch("subarticular-L4-L5-right").lower(), r"выражен|severe")

    def test_verified_severe_problem_names_traversing_left_l5(self):
        report = _fusion(case_by_id("F001-01"))
        _, html = self.render(report)
        text = _ReportHTML(html).element("fusion-subarticular-L4-L5-left")
        self.assertIn("L4-L5", text.replace("–", "-"))
        self.assertRegex(text.lower(), r"лев|слев")
        self.assertRegex(text.lower(), r"проходящ")
        self.assertRegex(text, r"кореш\w*\s+L5\b")
        self.assertRegex(text.lower(), r"потенциальн")
        self.assertNotIn("предположение:", text.lower())

    def test_unreviewed_severe_is_hypothesis_without_promoting_formal_candidates(self):
        report = _fusion(case_by_id("F001-13"))
        before = copy.deepcopy(report)
        markdown, html = self.render(report)
        parsed = _ReportHTML(html)
        text = parsed.element("fusion-subarticular-L4-L5-left")
        self.assertIn("предположение:", text.lower())
        self.assertRegex(text, r"кореш\w*\s+L5\b")
        self.assertRegex((markdown + " " + parsed.visible_text()).lower(), r"нумерац.*совмещ.*(?:не подтверж|требуют подтверж)")
        self.assertEqual(report["root_candidates"], [])
        self.assertEqual(report["coverage"]["indeterminate"], 20)
        self.assertEqual(report, before)

    def test_l5_s1_maps_to_s1_subarticular_and_l5_foraminal(self):
        for case_id, branch_id, root, role in (("F001-03", "subarticular-L5-S1-right", "S1", "проходящ"),
                                               ("F001-04", "foraminal-L5-S1-left", "L5", "выходящ")):
            with self.subTest(branch_id=branch_id):
                _, html = self.render(_fusion(case_by_id(case_id)))
                text = _ReportHTML(html).element("fusion-" + branch_id)
                self.assertRegex(text, r"кореш\w*\s+" + root + r"\b")
                self.assertRegex(text.lower(), role)
                self.assertNotRegex(text, r"\bL6\b")

    def test_moderate_keeps_anatomical_relation_but_impact_undetermined(self):
        report = _fusion(case_by_id("F001-09"))
        _, html = self.render(report)
        text = _ReportHTML(html).element("fusion-subarticular-L4-L5-left")
        self.assertRegex(text.lower(), r"умерен|moderate")
        self.assertRegex(text, r"кореш\w*\s+L5\b")
        self.assertIn("воздействие не определено", text.lower())
        self.assertNotRegex(text.lower(), r"потенциальн|вероятно пораж")
        self.assertEqual(report["root_candidates"], [])

    def test_normal_mild_is_combined_class_not_a_fusion_problem(self):
        report = _fusion(case_by_id("F001-10"))
        markdown, html = self.render(report)
        parsed = _ReportHTML(html)
        self.assertRegex(parsed.branch("subarticular-L4-L5-left").lower(), r"норм.*л[её]гк")
        self.assertNotIn("fusion-subarticular-L4-L5-left", parsed.ids)
        self.assertRegex((markdown + " " + parsed.visible_text()).lower(), r"норм.*л[её]гк.*един")
        self.assertIs(report["global_root_clearance"], False)

    def test_unknown_or_conflicting_grades_do_not_get_named_conditional_roots(self):
        for case_id in ("F001-19", "F001-23", "F001-18"):
            with self.subTest(case_id=case_id):
                report = _fusion(case_by_id(case_id))
                _, html = self.render(report)
                text = _ReportHTML(html).element("fusion")
                self.assertRegex(text.lower(), r"не определ|неизвест|непринят|противореч|не оцен|недостат")
                self.assertNotRegex(text, r"кореш\w*\s+(?:L[1-5]|S1)\b")
                self.assertEqual(report["root_candidates"], [])

    def test_unaccepted_disc_observation_kept_without_fabricating_herniation(self):
        report = _fusion(case_by_id("F001-30"))
        _, html = self.render(report)
        text = _ReportHTML(html).element("disc-L4-L5")
        self.assertIn("present", text)
        self.assertRegex(text.lower(), r"не ?принят.*шкал|шкал.*не ?принят")
        self.assertRegex(text.lower(), r"не оцен|не определ|не подключ")
        self.assertEqual(report["etiology"], "unassigned")
        self.assertEqual(len(report["root_candidates"]), 1)

    def test_native_labels_escaped_in_html_without_external_resources(self):
        case = copy.deepcopy(case_by_id("F001-20"))
        payload = '<script>alert("native")</script><img src="https://example.invalid/mri" onerror="alert(1)">'
        case["inputs"]["branches"][0]["label"] = payload
        case["inputs"]["producer_method"] = "model"
        case["inputs"]["branches"][0]["probabilities"] = [0.05, 0.1, 0.85]
        report = _fusion(case)
        markdown, html = self.render(report)
        parsed = _ReportHTML(html)
        self.assertNotIn(payload, html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("<script>", parsed.branch("subarticular-L4-L5-left"))
        self.assertNotIn("<script>", markdown)
        self.assertNotIn("85.00%", parsed.visible_text())
        self.assertFalse({"script", "iframe", "object", "embed", "base"} & set(parsed.tags))
        for tag, name, value in parsed.attributes:
            self.assertFalse(name.lower().startswith("on"), (tag, name))
            if name.lower() == "src" or (tag in {"link", "image"} and name.lower() == "href"):
                self.assertTrue((value or "").startswith("data:"), (tag, name, value))
        self.assertNotRegex(html, r"@import\s|url\(\s*['\"]?https?://")


if __name__ == "__main__":
    unittest.main()
