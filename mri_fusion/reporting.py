"""Deterministic physician-facing views of an already validated fusion report."""

import html
import json


SIDES = {"left": "слева", "right": "справа", "unknown": "сторона не установлена", "midline": "по средней линии", "bilateral": "с обеих сторон", "not_applicable": "не применимо"}
ZONES = {"foraminal": "Межпозвонковое отверстие (фораминальная зона)", "subarticular": "Латеральный рецессус (субартикулярная зона)", "disc": "Межпозвонковый диск", "central": "Центральная зона", "extraforaminal": "Экстрафораминальная зона", "unknown": "Зона не установлена"}
GRADES = {"normal_mild": "норма / лёгкий стеноз (объединённый класс)", "moderate": "умеренный стеноз", "severe": "выраженный стеноз"}
STATUSES = {"assessed": "оценка выполнена в исходной шкале", "indeterminate": "исходный результат неопределён", "not_run": "модуль не запускался", "error": "ошибка модуля"}
DECISIONS = {"potential_impact": "Признаки потенциального воздействия", "indeterminate": "Недостаточно данных / неопределённо", "no_candidate_by_evaluated_criterion": "Кандидат по критерию выраженного стеноза не сформирован"}
STATES = {"confirmed": "подтверждено в предоставленной проверке", "verified": "проверено по предоставленным подтверждениям", "not_required_with_evidence": "не требуется согласно предоставленным подтверждениям", "sufficient": "достаточно по предоставленной проверке", "insufficient": "недостаточно", "unreviewed": "не проверено", "unknown": "не установлено", "unsupported": "неподдерживаемая нумерация / анатомия", "failed": "проверка не пройдена", "contradictory": "противоречивые данные", "unverified": "не подтверждено"}
REVIEW_LABELS = {"anatomical_numbering": "Анатомическая нумерация уровней", "side_assignment": "Определение стороны", "source_geometry": "Геометрия исходных изображений", "cross_series_registration": "Совмещение всех использованных серий", "diagnostic_adequacy": "Диагностическая достаточность изображений"}
REASONS = {"source_missing": "Исходная оценка этой зоны не поступила", "source_indeterminate": "Неопределённость исходного MRI-результата сохранена", "source_not_run": "Модуль не был запущен", "source_error": "Модуль завершился ошибкой", "review_context_missing": "Контекст врачебной проверки не предоставлен", "review_target_missing": "Проверка этой конкретной оценки не предоставлена", "numbering_unreviewed": "Нумерация уровней не проверена", "registration_unverified": "Совмещение серий не подтверждено", "unsupported_scale": "Исходная шкала не поддерживается правилом", "unsupported_label": "Исходный класс не поддерживается правилом", "native_probabilities_missing": "Распределение классов модели не предоставлено", "tied_probability_maximum": "У распределения классов нет единственного максимума", "same_target_evidence_conflict": "Исходные оценки одной зоны противоречат друг другу", "review_series_coverage_incomplete": "Проверка не охватывает все использованные серии, включая предшествующие модули", "moderate_threshold_unestablished": "Для умеренного стеноза критерий воздействия в этой версии не установлен", "severe_stenosis_research_candidate": "Выраженный стеноз соответствует исследовательскому порогу после прохождения всех проверок", "no_candidate_by_evaluated_criterion": "Критерий выраженного стеноза в этой зоне не выполнен", "herniation_not_evaluated": "Грыжа не оценена данным fusion", "unknown_target": "Не установлен уровень, сторона или анатомическая зона"}
ALGORITHM = [
    "Проверяются версии, контрольные суммы файлов и связи с исследованием. Ошибка входов не становится медицинским выводом.",
    "Сохраняются 20 отдельных оценок: 5 уровней × 2 стороны × 2 анатомические зоны. Пропуск, ошибка и неопределённость остаются неизвестным результатом.",
    "Для именованной корешковой привязки нужны завершённая исходная оценка, принятая шкала, непротиворечивые данные и все подтверждения: нумерация, сторона, геометрия, диагностическая достаточность и совмещение использованных серий.",
    "После прохождения проверок выраженный стеноз формирует анатомического кандидата. Умеренный стеноз остаётся неопределённым. Объединённый класс «норма / лёгкий» означает только отсутствие кандидата по критерию выраженного стеноза.",
    "Выходящий или проходящий корешок определяется явной анатомической таблицей. Основания разных уровней и зон сохраняются отдельно; вероятности не суммируются, единственный «главный» корешок не выбирается.",
]
LIMITS = "Порог «только выраженный стеноз» — исследовательский выбор, клинически не валидированный. Модельный класс и анатомический кандидат не подтверждают компрессию, симптомность находки или радикулопатию. Клиническое сопоставление относится к Stage II."
REVIEW_NOTE = "Состояния проверки ниже воспроизводят предоставленные подтверждения. Этот отчёт не удостоверяет личность проверяющего и не выполняет независимую врачебную проверку. Согласие локализаторов и технически допущенное неподтверждённое совмещение её не заменяют."
DISC_NOTE = "Контекст грыжи не оценён этим fusion: отсутствие подключённого анализа не означает отсутствия грыжи. Причина стеноза грыже не приписывается; прямое правило воздействия по одной грыже в этой версии отключено."


def _h(value):
    return html.escape(str(value), quote=True)


def _m(value):
    text = _h(value).replace("\\", "\\\\").replace("\n", " ")
    for char in "`*_[]()!|":
        text = text.replace(char, "\\" + char)
    return text


def _root(value):
    if value is None:
        return "Не назначен: условия анатомической привязки не выполнены"
    role = {"exiting": "выходящий", "traversing": "проходящий"}[value["role"]]
    return "{} корешок {} {}".format(role, value["root"], SIDES[value["side"]])


def _title(target):
    level = "уровень не установлен" if target["level"] == "unknown" else target["level"]
    return "{} · {} · {}".format(level, SIDES.get(target["side"], target["side"]), ZONES.get(target["zone"], target["zone"]))


def _label(observation):
    if observation["scale_id"] != "rsna2024:stenosis-severity:v1":
        return "непринятая шкала {}; исходная метка «{}»".format(
            observation["scale_id"], observation["label"])
    return GRADES.get(observation["label"], observation["label"])


def _finding(evidence):
    observation = evidence["observation"]
    if observation is None:
        return STATUSES[evidence["status"]] + "; исходного класса нет"
    if evidence["component"] == "disc_analysis":
        return "Исходная оценка диска: метка «{}», шкала {}; {}. Интерпретация воздействия по этой шкале отключена.".format(
            observation["label"], observation["scale_id"], STATUSES[evidence["status"]])
    label = _label(observation)
    method = evidence["producer"]["method"]
    prefix = "Класс модели" if method == "model" else "Исходная оценка"
    if evidence["status"] != "assessed":
        prefix += " (предварительно; результат не подтверждён)"
    return "{}: {}; {}".format(prefix, label, STATUSES[evidence["status"]])


def _reasons(branch):
    result = []
    prefixes = {"numbering_": "Нумерация", "side_assignment_": "Сторона", "source_geometry_": "Геометрия", "registration_": "Совмещение серий", "diagnostic_adequacy_": "Диагностическая достаточность"}
    for code in branch["reason_codes"]:
        if code == "herniation_not_evaluated":
            continue
        message = REASONS.get(code)
        if message is None:
            message = next((label + ": " + STATES.get(code[len(prefix):], code[len(prefix):]) for prefix, label in prefixes.items() if code.startswith(prefix)), "Дополнительная причина источника: " + code)
        if message not in result:
            result.append(message)
    return result


def _conclusion(report):
    count, unknown = len(report["root_candidates"]), report["coverage"]["indeterminate"]
    if count:
        roots = "; ".join("корешок {} {}".format(row["root"], SIDES[row["side"]]) for row in report["root_candidates"])
        return "По исследовательским критериям сформированы анатомические кандидаты: {}. Это признаки потенциального воздействия; требуется клиническое сопоставление.".format(roots)
    if unknown == 20:
        return "Определить потенциально затронутые корешки по имеющимся данным пока нельзя: все 20 оценок остаются неопределёнными. Отсутствие кандидатов не означает отсутствия патологии."
    if unknown:
        return "Кандидаты по критерию выраженного стеноза не сформированы; {} из 20 оценок неопределённы. Возможность воздействия на корешки не исключена.".format(unknown)
    return "Кандидаты по критерию выраженного стеноза в оценённых зонах не сформированы. Это ограниченный вывод: объединённый класс «норма / лёгкий» не исключает лёгких изменений, других зон и причин воздействия."


def _coverage(report):
    counts = report["coverage"]
    supplied = sum(bool(branch["evidence"]) for branch in report["branches"])
    return "Предусмотрено 20 сочетаний уровня, стороны и зоны; исходные оценки получены для {} из 20. Признаки потенциального воздействия: {}. Без кандидата по этому критерию: {}. Неопределённо: {}. Находок без установленной привязки: {}.".format(supplied, counts["potential_impact"], counts["no_candidate_by_evaluated_criterion"], counts["indeterminate"], len(report["unresolved_native_evidence"]))


def _clarifications(report):
    """List only outstanding conditions actually present in branch evidence."""
    reasons = {code for branch in report["branches"] for code in branch["reason_codes"]}
    messages = []
    if reasons & {"review_context_missing", "review_target_missing"}:
        messages.append("Предоставить проверку анатомической нумерации, стороны, геометрии, совмещения использованных серий и диагностической достаточности изображений.")
    elif any(not b["gate_states"]["reviews_passed"] for b in report["branches"]):
        messages.append("Уточнить непроверенные, недостаточные или противоречивые условия и охват исходных серий; конкретные условия указаны в разборе каждой оценки.")
    if "source_indeterminate" in reasons:
        messages.append("После проверки снимков зафиксировать отдельную пересмотренную исходную оценку. Дополнительное подтверждение само по себе не повышает уже неопределённый результат.")
    if reasons & {"source_missing", "source_not_run", "source_error"}:
        messages.append("Получить отсутствующие исходные оценки или устранить ошибку обработки. Пропуски сохраняются как недостаточность данных.")
    if reasons & {"same_target_evidence_conflict", "unsupported_scale", "unsupported_label", "native_probabilities_missing", "tied_probability_maximum"}:
        messages.append("Уточнить противоречивые или неподдержанные исходные оценки и распределения классов; использовать их без разрешения неопределённости нельзя.")
    if "moderate_threshold_unestablished" in reasons:
        messages.append("Для умеренного стеноза выбранное исследовательское правило не определяет воздействие на корешок. Простое подтверждение привязки не меняет этот порог.")
    return messages


def _checks(branch):
    evidence = branch["evidence"]
    checks = [("Статус исходной оценки допускает корешковый вывод", "да" if branch["gate_states"]["source_assessed"] else "нет — исходная неопределённость или отсутствие оценки сохранены")]
    for gate, label in REVIEW_LABELS.items():
        states = sorted({item["review_states"].get(gate, "unreviewed") for item in evidence}) or ["unreviewed"]
        checks.append((label, "; ".join(STATES[state] for state in states)))
    coverage = sorted({item["review_series_coverage"] for item in evidence}) or ["missing"]
    checks.append(("Охват проверкой всех исходных серий и серий предшествующих модулей", "; ".join({"complete": "полный", "partial": "неполный", "missing": "проверка не предоставлена"}[value] for value in coverage)))
    for gate, label in (("known_scale_and_label", "Принятая шкала и известный класс"), ("native_probabilities_usable", "Допустимость распределения классов или независимой оценки"), ("evidence_consistent", "Согласованность исходных оценок этой зоны")):
        checks.append((label, "да" if branch["gate_states"][gate] else "условие не выполнено"))
    return checks


def _threshold(branch):
    if not all(branch["gate_states"].values()):
        return "Порог не применяется к корешковому выводу: обязательные условия не выполнены. Даже класс «выраженный» сам по себе не разрешает назначить корешок."
    label = branch["evidence"][0]["observation"]["label"]
    threshold = {"severe": "Класс «выраженный стеноз» соответствует исследовательскому порогу после прохождения всех обязательных условий.", "moderate": "Класс «умеренный стеноз»: критерий воздействия в данной версии не установлен; исход остаётся неопределённым.", "normal_mild": "Объединённый класс «норма / лёгкий стеноз»: кандидат по критерию выраженного стеноза не сформирован. Утверждать отсутствие стеноза или воздействия нельзя."}[label]
    return threshold + " Анатомическое соответствие после проверок: {} → {}.".format(_title(branch["target"]), _root(branch["mapped_root"]))


def _rows(report):
    for branch in report["branches"]:
        labels = sorted({_label(e["observation"]) for e in branch["evidence"] if e["observation"] is not None})
        preliminary = any(e["status"] != "assessed" for e in branch["evidence"] if e["observation"] is not None)
        finding = "; ".join(labels) if labels else "Нет исходной оценки"
        if preliminary:
            finding = "Предварительно: " + finding
        zone = {"foraminal": "Фораминальная", "subarticular": "Субартикулярная"}[branch["target"]["zone"]]
        decision = {"potential_impact": "Потенциальное воздействие", "indeterminate": "Недостаточно данных", "no_candidate_by_evaluated_criterion": "Нет кандидата по критерию выраженного стеноза"}[branch["decision"]]
        root = _root(branch["mapped_root"]) if branch["mapped_root"] is not None else "Не установлена †"
        yield branch, [branch["target"]["level"] + (" †" if branch["mapped_root"] is None else ""), SIDES[branch["target"]["side"]], zone, finding, decision, root]


def _technical_evidence(evidence):
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2)


def _probabilities(evidence):
    observation = evidence["observation"]
    if observation is None:
        return "Распределение классов не предоставлено."
    if (evidence["component"] == "disc_analysis"
            or observation["scale_id"] != "rsna2024:stenosis-severity:v1"):
        return "Измерения непринятой шкалы сохранены в исходных подробностях и не интерпретируются как вероятности классов стеноза."
    values = {item["name"]: item for item in observation["measurements"] if item["name"].startswith("p_")}
    if not values:
        return "Распределение классов не предоставлено; для независимой ручной или синтетической оценки оно может отсутствовать."
    return "Распределение исходных классов (для модели — вероятности классов модели): " + "; ".join("{} — {:.2f}%".format(GRADES.get(key[2:], key), value["value"] * 100) for key, value in sorted(values.items())) + ". Это не вероятность поражения корешка, компрессии или радикулопатии."


def _footer(report):
    return {key: report[key] for key in ("fusion_id", "study_id", "study_digest", "synthetic", "policy_id", "policy_version", "policy_sha256", "review_context_sha256", "input_artifacts", "coverage")}


def _root_summary_texts(report):
    branches = {branch["branch_id"]: branch for branch in report["branches"]}
    for summary in report["root_summaries"]:
        sources = "; ".join(_title(branches[key]["target"])
                            for key in summary["branch_refs"])
        unresolved = len(summary["unresolved_expected_branch_ids"])
        coverage = ("Среди ожидаемых оснований по двум правилам остаются неопределёнными: {}."
                    .format(unresolved) if unresolved else
                    "Все ожидаемые основания по двум правилам оценены; другие причины воздействия здесь не исключаются.")
        yield "Корешок {} {}: {}. Источники анатомической привязки: {}. {}".format(
            summary["root"], SIDES[summary["side"]],
            DECISIONS[summary["decision"]], sources, coverage)


def _html_overview(report):
    parts = []
    clarifications = _clarifications(report)
    if clarifications:
        parts.append('<section id="clarifications"><h2>Что требуется для уточнения</h2><ul>')
        parts.extend('<li>' + _h(text) + '</li>' for text in clarifications)
        parts.append('</ul></section>')
    if report["root_summaries"]:
        parts.append('<section id="roots"><h2>Основания корешковых выводов</h2><ul>')
        parts.extend('<li>' + _h(text) + '</li>' for text in _root_summary_texts(report))
        parts.append('</ul></section>')
    return ''.join(parts)


def render_markdown(report):
    """Return a readable Russian Markdown report without changing its input."""
    lines = ["# МРТ: потенциальное анатомическое воздействие на корешки", "", "**Предварительный исследовательский отчёт · Stage I**", ""]
    if report["synthetic"]:
        lines += ["**Синтетический пример: данные и подтверждения вымышлены; это не заключение о пациенте.**", ""]
    lines += ["Исследование: **" + _m(report["study_id"]) + "**.", "", "## Предварительное заключение", "", _m(_conclusion(report)), "", _m(_coverage(report)), "", _m(LIMITS), ""]
    clarifications = _clarifications(report)
    if clarifications:
        lines += ["## Что требуется для уточнения", ""]
        lines += ["- " + _m(text) for text in clarifications]
        lines.append("")
    if report["root_summaries"]:
        lines += ["## Основания корешковых выводов", ""]
        lines += ["- " + _m(text) for text in _root_summary_texts(report)]
        lines.append("")
    lines += ["## Оценки по уровням и сторонам", "", "| Уровень источника | Сторона | Зона | Исходный класс | Вывод | Корешковая привязка |", "|---|---|---|---|---|---|"]
    for _, row in _rows(report):
        lines.append("| " + " | ".join(_m(value) for value in row) + " |")
    lines += ["", "† Исходная метка уровня; полный набор условий для корешковой привязки не выполнен. Детали каждой проверки приведены ниже.", "", "## Как получен вывод", ""]
    lines += ["{}. {}".format(index, _m(text)) for index, text in enumerate(ALGORITHM, 1)]
    lines += ["", _m(REVIEW_NOTE), "", "## Ход решения для каждой оценки", ""]
    for branch in report["branches"]:
        lines += ["### " + _m(_title(branch["target"])), "", "**1. Исходная находка.** " + _m("; ".join(_finding(e) for e in branch["evidence"]) or "Исходная оценка не поступила."), "", "**2. Обязательные условия.**", ""]
        lines += ["- **{}:** {}.".format(_m(label), _m(value)) for label, value in _checks(branch)]
        lines += ["", "**3. Исследовательский порог.** " + _m(_threshold(branch)), "", "**4. Итог.** " + _m(DECISIONS[branch["decision"]]) + ". " + _m(_root(branch["mapped_root"])) + ".", ""]
        lines += ["- " + _m(reason) + "." for reason in _reasons(branch)]
        for evidence in branch["evidence"]:
            lines += ["", _m(_probabilities(evidence)), "", "<details><summary>Исходные значения и привязка к изображениям</summary><pre>" + _h(_technical_evidence(evidence)) + "</pre></details>", ""]
    lines += ["## Находки без установленной привязки", ""]
    for evidence in report["unresolved_native_evidence"]:
        lines += [_m(_title(evidence["target"]) + ". " + _finding(evidence)), "", "Корешок не назначается: уровень, сторона или зона не установлены.", "", "<details><summary>Исходная находка</summary><pre>" + _h(_technical_evidence(evidence)) + "</pre></details>", ""]
    if not report["unresolved_native_evidence"]:
        lines += ["Дополнительных находок без установленной привязки не поступило.", ""]
    lines += ["## Межпозвонковые диски и грыжа", "", _m(DISC_NOTE), ""]
    for evidence in report["raw_disc_evidence"]:
        lines += [_m(_title(evidence["target"]) + ". " + _finding(evidence)), "", "<details><summary>Исходная оценка диска без интерпретации причины</summary><pre>" + _h(_technical_evidence(evidence)) + "</pre></details>", ""]
    if not report["raw_disc_evidence"]:
        lines += ["Исходные оценки дисков не предоставлены.", ""]
    lines += ["## Анатомическая справка", "", "Условная общая схема; это не назначение корешков данному исследованию. Применяется только после прохождения обязательных проверок.", "", "| Уровень | Фораминальная зона: выходящий | Субартикулярная зона: проходящий |", "|---|---|---|"]
    lines += ["| {} | {} | {} |".format(level, exiting, traversing) for level, exiting, traversing in _anatomy()]
    lines += ["", "## Техническая прослеживаемость", "", "Новый анализ МРТ моделями не выполнялся. Ниже — версии, контрольные суммы и ссылки на сохранённые входы.", "", "<details><summary>Идентификаторы и контрольные суммы</summary><pre>" + _h(json.dumps(_footer(report), ensure_ascii=False, sort_keys=True, indent=2)) + "</pre></details>", ""]
    return "\n".join(lines)


def _anatomy():
    return [("L1–L2", "L1", "L2"), ("L2–L3", "L2", "L3"), ("L3–L4", "L3", "L4"), ("L4–L5", "L4", "L5"), ("L5–S1", "L5", "S1")]


def _html_evidence(evidence):
    producer = evidence["producer"]
    source = evidence["source_ref"]
    summary = "Источник: {} / {} · {}".format(source["result_id"], source["assessment_id"], producer["name"])
    images = "; ".join("серия {}, изображение {}, кадр {}".format(ref["series_id"], ref["image_id"], ref["frame_index"]) for ref in evidence["image_refs"]) or "Привязка к изображению отсутствует"
    return '<details class="evidence"><summary>{}</summary><p>{}</p><p>{}</p><p>{}</p><pre>{}</pre></details>'.format(_h(summary), _h(_finding(evidence)), _h(_probabilities(evidence)), _h(images), _h(_technical_evidence(evidence)))


def render_html(report):
    """Return a self-contained, escaped HTML view with no scripts or network."""
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>МРТ: анатомическое воздействие на корешки</title><style>' + _CSS + '</style></head><body><main>', '<header><p class="eyebrow">Stage I · исследовательский МРТ-модуль</p><h1>Потенциальное анатомическое воздействие на корешки</h1><p>Предварительный отчёт по сохранённым результатам МРТ</p></header>']
    if report["synthetic"]:
        parts.append('<p class="notice">Синтетический пример: данные и подтверждения вымышлены. Это не заключение о пациенте.</p>')
    parts.append('<p class="study-id">Исследование: <strong>' + _h(report["study_id"]) + '</strong></p>')
    parts += ['<section id="conclusion" class="conclusion"><h2>Предварительное заключение</h2><p class="lead">' + _h(_conclusion(report)) + '</p><p>' + _h(LIMITS) + '</p></section>', '<section id="coverage"><h2>Полнота оценки</h2><p>' + _h(_coverage(report)) + '</p></section>' + _html_overview(report), '<section id="findings"><h2>Оценки по уровням и сторонам</h2><div class="table-scroll" role="region" aria-label="Двадцать оценок стеноза" tabindex="0"><table><thead><tr>']
    parts += ['<th scope="col">' + _h(label) + '</th>' for label in ("Уровень источника", "Сторона", "Анатомическая зона", "Исходный класс", "Вывод", "Корешковая привязка")]
    parts.append('</tr></thead><tbody>')
    for branch, row in _rows(report):
        parts.append('<tr><th scope="row"><a href="#branch-{}">{}</a></th>'.format(_h(branch["branch_id"]), _h(row[0])) + ''.join('<td>{}</td>'.format(_h(value)) for value in row[1:]) + '</tr>')
    parts += ['</tbody></table></div><p class="muted">† Исходная метка уровня; полный набор условий для корешковой привязки не выполнен. Нажмите уровень, чтобы проверить ход решения.</p></section>', '<section id="algorithm"><h2>Как получен вывод</h2><ol>' + ''.join('<li>' + _h(text) + '</li>' for text in ALGORITHM) + '</ol><p>' + _h(REVIEW_NOTE) + '</p></section>', '<section id="traces"><h2>Ход решения для каждой оценки</h2><p>Откройте нужную строку: исходная находка → обязательные условия → исследовательский порог → итог.</p>']
    for branch in report["branches"]:
        finding = "; ".join(_finding(e) for e in branch["evidence"]) or "Исходная оценка не поступила."
        parts += ['<details id="branch-{}" class="trace"><summary>{}<span class="outcome">{}</span></summary><div class="trace-body">'.format(_h(branch["branch_id"]), _h(_title(branch["target"])), _h(DECISIONS[branch["decision"]])), '<h3>1. Исходная находка</h3><p>' + _h(finding) + '</p><h3>2. Обязательные условия</h3><dl class="checks">']
        parts += ['<div><dt>{}</dt><dd>{}</dd></div>'.format(_h(label), _h(value)) for label, value in _checks(branch)]
        parts += ['</dl><h3>3. Исследовательский порог</h3><p>' + _h(_threshold(branch)) + '</p><h3>4. Итог</h3><p><strong>' + _h(DECISIONS[branch["decision"]]) + '.</strong> ' + _h(_root(branch["mapped_root"])) + '.</p><ul>']
        parts += ['<li>' + _h(reason) + '.</li>' for reason in _reasons(branch)]
        parts += ['</ul>'] + [_html_evidence(evidence) for evidence in branch["evidence"]] + ['</div></details>']
    parts += ['</section><section id="unresolved"><h2>Находки без установленной привязки</h2>']
    for evidence in report["unresolved_native_evidence"]:
        parts += ['<h3>' + _h(_title(evidence["target"])) + '</h3><p>' + _h(_finding(evidence)) + '</p><p>Корешок не назначается: уровень, сторона или зона не установлены.</p>', _html_evidence(evidence)]
    if not report["unresolved_native_evidence"]:
        parts.append('<p>Дополнительных находок без установленной привязки не поступило.</p>')
    parts += ['</section><section id="discs"><h2>Межпозвонковые диски и грыжа</h2><p>' + _h(DISC_NOTE) + '</p>']
    for evidence in report["raw_disc_evidence"]:
        parts += ['<h3>' + _h(_title(evidence["target"])) + '</h3><p>' + _h(_finding(evidence)) + '</p>', _html_evidence(evidence)]
    if not report["raw_disc_evidence"]:
        parts.append('<p>Исходные оценки дисков не предоставлены.</p>')
    parts += ['</section><section id="anatomy"><h2>Анатомическая справка</h2><p>Условная общая схема; это не назначение корешков данному исследованию. Применяется только после прохождения обязательных проверок.</p><table><thead><tr><th scope="col">Уровень</th><th scope="col">Фораминальная зона: выходящий</th><th scope="col">Субартикулярная зона: проходящий</th></tr></thead><tbody>']
    parts += ['<tr><th scope="row">{}</th><td>{}</td><td>{}</td></tr>'.format(*(_h(value) for value in row)) for row in _anatomy()]
    parts += ['</tbody></table></section><footer id="technical"><h2>Техническая прослеживаемость</h2><p>Новый анализ МРТ моделями не выполнялся. Контрольные суммы относятся к сохранённым входам; правила не устанавливают клинический диагноз.</p><details><summary>Идентификаторы, версии и контрольные суммы</summary><pre>' + _h(json.dumps(_footer(report), ensure_ascii=False, sort_keys=True, indent=2)) + '</pre></details></footer></main></body></html>']
    return "\n".join(parts) + "\n"


_CSS = """
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#172b3a;background:#f3f6f8;font-size:16px;line-height:1.55}*{box-sizing:border-box}body{margin:0}main{max-width:1260px;margin:auto;padding:42px 32px 60px;background:white}header{border-bottom:2px solid #234e66;padding-bottom:20px}h1{font-size:2rem;line-height:1.2;max-width:850px;margin:10px 0}h2{font-size:1.3rem;line-height:1.3;margin:0 0 14px}h3{font-size:1.05rem;margin:20px 0 8px}p{margin:10px 0}section,footer{margin-top:34px}.eyebrow{font-size:.85rem;letter-spacing:.04em;text-transform:uppercase;color:#486879}.conclusion{background:#edf4f7;border-left:5px solid #234e66;padding:22px}.lead{font-size:1.2rem;font-weight:600}.notice{padding:14px 18px;background:#fff3d4;border-left:4px solid #b27a16}.muted{font-size:.87rem;color:#526571}.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.85rem}th,td{padding:12px 10px;border-bottom:1px solid #dce4e8;text-align:left;vertical-align:top}thead{background:#edf2f5}thead th{font-weight:650}tbody th{white-space:nowrap}a{color:#205979;text-underline-offset:3px}li{margin:8px 0}.trace{border:1px solid #d4dfe5;border-radius:5px;margin:12px 0}.trace>summary{padding:16px 18px;font-weight:600;background:#f6f8fa;cursor:pointer}.outcome{display:block;font-size:.86rem;font-weight:400;color:#526571;margin:5px 0 0 20px}.trace-body{padding:0 20px 20px}.checks{margin:8px 0}.checks>div{display:grid;grid-template-columns:minmax(220px,1fr) 1.4fr;gap:20px;padding:9px 0;border-bottom:1px solid #e5ebee}dt{font-weight:550}dd{margin:0}.evidence{margin:16px 0;border-top:1px solid #dce4e8;padding-top:10px}.evidence>summary,footer summary{cursor:pointer;font-size:.9rem;overflow-wrap:anywhere}pre{font-size:.75rem;line-height:1.45;background:#f4f6f8;padding:16px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}footer{padding-top:22px;border-top:2px solid #dce4e8;color:#526571}summary:focus-visible,a:focus-visible,.table-scroll:focus-visible{outline:3px solid #b27a16;outline-offset:3px}@media(max-width:700px){main{padding:24px 16px}h1{font-size:1.5rem}.conclusion{padding:16px}.checks>div{grid-template-columns:1fr;gap:3px}table{min-width:800px}#anatomy table{min-width:0}.trace-body{padding:0 14px 16px}}@media print{:root{background:white;font-size:10pt}main{max-width:none;padding:0}section,footer{margin-top:20px}.table-scroll{overflow:visible}table{font-size:8pt}th,td{padding:6px 5px}.trace>summary{background:white}.trace{break-inside:avoid}.conclusion{background:white}a{color:inherit;text-decoration:none}pre{max-width:100%;font-size:7pt}.checks>div{grid-template-columns:1fr 1.4fr}h2,h3{break-after:avoid}}
"""
