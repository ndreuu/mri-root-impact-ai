"""Four concise physician sections, generated from checked MRI evidence."""

from urllib.parse import quote

from .contracts import FusionContractError
from .evidence import build_evidence
from .reporting import _h, _m, _root, SIDES


LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
GRADES = {"normal_mild": "Норма/лёгкий", "moderate": "Умеренный", "severe": "Выраженный"}
ZONES = {"foraminal": "фораминальный", "subarticular": "субартикулярный"}
TITLES = ("1. Анализ межпозвонковых дисков", "2. Фораминальный стеноз",
          "3. Субартикулярный стеноз", "4. Fusion")


def _level(value):
    return value.replace("-", "–")


def _marked(branch):
    return " †" if branch["mapped_root"] is None else ""


def _grade(fact, branch):
    if fact["native_grade"] is not None:
        return GRADES[fact["native_grade"]]
    if fact["native_observations"]:
        return "Непринятая/противоречивая оценка: " + "/".join(o["label"] for o in fact["native_observations"])
    statuses = {e["status"] for e in branch["evidence"]}
    return "Ошибка оценки" if "error" in statuses else "Не запускалось" if "not_run" in statuses else "Не оценено"


def _disc_prediction(bundle, level):
    if bundle.get('disc_analysis_status') != 'spinenet_binary_unreviewed':
        return None
    rows = [r for r in bundle.get('disc_predictions', []) if r['native_level'] == level]
    return rows[0]['native_class'] if len(rows) == 1 else None


def _disc_result(report, level, bundle=None):
    if bundle is not None and bundle.get('disc_analysis_status') == 'spinenet_binary_unreviewed':
        value = _disc_prediction(bundle, level)
        return {1: 'Модель: признак грыжи †', 0: 'Модель: признак грыжи не выявлен †'}.get(value, 'Грыжа не оценена')
    evidence = [e for e in report["raw_disc_evidence"] if e["target"]["level"] == level]
    observations = [e["observation"] for e in evidence if e["observation"] is not None]
    if observations:
        return "Не оценено; исходная метка: " + "/".join(o["label"] for o in observations) + " (шкала не принята)"
    return "Грыжа не оценена"


def fusion_lines(report, bundle):
    """Problems → short conclusion; conditional anatomy never changes JSON."""
    branches = {b["branch_id"]: b for b in report["branches"]}
    lines = []
    represented = set()
    for fact in bundle["facts"]:
        grade = fact["native_grade"]
        branch = branches[fact["branch_id"]]
        if grade == "normal_mild" or not branch["evidence"]:
            continue
        target = fact["native_target"]
        represented.add(target['level'])
        prefix = "{} + {} {} стеноз {}".format(_level(target["level"]),
            {"moderate": "умеренный", "severe": "выраженный"}.get(grade, "неопределённый"),
            ZONES[target["zone"]], SIDES[target["side"]])
        root = fact["conditional_anatomy"]
        if grade == "severe" and fact["verified_candidate"] is not None:
            conclusion = "потенциальное воздействие на " + _root(root)
        elif grade == "severe" and root is not None:
            conclusion = "предположение: потенциальное воздействие на " + _root(root) + " †"
        elif grade == "moderate" and root is not None:
            conclusion = _root(root) + "; воздействие не определено" + _marked(branch)
        else:
            zone = {"foraminal": "фораминальная зона", "subarticular": "субартикулярная зона"}[target["zone"]]
            prefix = "{} + {} {}: {}".format(_level(target["level"]), zone, SIDES[target["side"]], _grade(fact, branch))
            conclusion = "корешок не определён"
        if _disc_prediction(bundle, target['level']) == 1:
            prefix += ' + признак грыжи (SpineNet; сторона неизвестна)'
        lines.append(("fusion-" + fact["branch_id"], prefix + " ⇒ " + conclusion + "."))
    for level in LEVELS:
        if level not in represented and _disc_prediction(bundle, level) == 1:
            lines.append(('fusion-disc-' + level, _level(level) +
                          ' + признак грыжи (SpineNet; сторона неизвестна) ⇒ корешок по этой находке не определён.'))
    for i, evidence in enumerate(report["unresolved_native_evidence"]):
        target = evidence["target"]
        observation = evidence["observation"]
        label = "нет оценки" if observation is None else observation["label"]
        lines.append(("unresolved-"+str(i), "{} + {} ⇒ корешок не определён.".format(target["level"], label)))
    return lines


def _figure(view, bundle, crop=True):
    if view is None:
        return '<span class="missing">Нет снимка</span>'
    image = bundle["images"][view["image_id"]]
    if not image["data_uri"].startswith("data:image/png;base64,"):
        raise FusionContractError("Report illustrations must be embedded PNGs, never network resources")
    width, height = image["width"], image["height"]
    contents = '<image href="{}" width="{}" height="{}"/>'.format(_h(image["data_uri"]), width, height)
    zoom = ""
    if crop:
        x0, y0, x1, y1 = view["source_crop_box"]
        y0, y1 = y0 + view["header_pixels"], y1 + view["header_pixels"]
        contents += '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#ffcc33" stroke-width="2"/>'.format(x0, y0, x1-x0, y1-y0)
        zoom = '<svg role="img" aria-label="Увеличение ROI" viewBox="{} {} {} {}">{}</svg>'.format(x0, y0, x1-x0, y1-y0, contents)
    ref = view["image_ref"]
    source = "{} / {} / кадр {}".format(ref["series_id"], ref["image_id"], ref["frame_index"])
    label = 'Вход модели SpineNet, срез 5/9' if view.get('purpose') == 'spinenet_binary_model_input' else 'Исходный кадр'
    return '<div class="images" title="{}"><svg role="img" aria-label="{}" viewBox="0 0 {} {}">{}</svg>{}</div>'.format(_h(source), _h(label), width, height, contents, zoom)


def _empty_fusion(bundle):
    if all(f["native_grade"] == "normal_mild" for f in bundle["facts"]):
        return "Умеренных/выраженных стенозов не указано. Воздействие по этим классам не установлено."
    return "Исходных оценок для вывода недостаточно."


def render_html(report, bundle):
    bundle = build_evidence(report) if bundle is None else bundle
    facts = {f["branch_id"]: f for f in bundle["facts"]}
    branches = {b["branch_id"]: b for b in report["branches"]}
    views = {v["branch_id"]: v for v in bundle["stage_views"]}
    discs = {v["native_level"]: v for v in bundle["disc_views"]}
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>МРТ: результаты</title><style>' + _CSS + '</style></head><body><main>',
             '<header><h1>МРТ поясничного отдела</h1><p>Исследовательская автооценка · ' + _h(report["study_id"]) + '</p></header>']
    if report["synthetic"]:
        parts.append('<p class="note">Синтетический пример.</p>')
    has_disc = bundle.get('disc_analysis_status') == 'spinenet_binary_unreviewed'
    note = 'SpineNet: бинарный признак грыжи. Размер, сторона и тип не определены.' if has_disc else 'LDH не подключён. Наличие, размер и сторона грыжи не определены.'
    sequence = 'Sagittal T2 · вход SpineNet' if has_disc else 'Axial T2'
    parts += ['<section id="discs"><h2>' + TITLES[0] + '</h2><p>' + note + '</p><table><thead><tr><th>Диск</th><th>Результат</th><th>' + sequence + '</th></tr></thead><tbody>']
    for level in LEVELS:
        parts.append('<tr id="disc-{}"><th scope="row">{}</th><td>{}</td><td>{}</td></tr>'.format(
            level, _level(level), _h(_disc_result(report, level, bundle)), _figure(discs.get(level), bundle, False)))
    parts.append('</tbody></table></section>')
    for zone, sequence, title in (("foraminal", "Sagittal T1", TITLES[1]), ("subarticular", "Axial T2", TITLES[2])):
        parts.append('<section id="{}-stage"><h2>{}</h2><table><thead><tr><th>Уровень</th><th>Слева · {}</th><th>Справа · {}</th></tr></thead><tbody>'.format(zone, title, sequence, sequence))
        for level in LEVELS:
            parts.append('<tr><th scope="row">' + _level(level) + '</th>')
            for side in ("left", "right"):
                key = zone + "-" + level + "-" + side
                label = _grade(facts[key], branches[key]) + _marked(branches[key])
                parts.append('<td id="branch-{}"><strong>{}</strong>{}</td>'.format(
                    key, _h(label), _figure(views.get(key), bundle)))
            parts.append('</tr>')
        parts.append('</tbody></table></section>')
    parts += ['<section id="fusion"><h2>' + TITLES[3] + '</h2><ul>']
    lines = fusion_lines(report, bundle)
    parts += ['<li id="{}">{}</li>'.format(key, _h(text)) for key, text in lines]
    if not lines:
        parts.append('<li>' + _h(_empty_fusion(bundle)) + '</li>')
    parts += ['</ul></section><footer><p>† Условная анатомическая привязка. Нумерация/совмещение требуют подтверждения.</p><p>Норма/лёгкий — единый класс, не исключает изменений. ROI — вход модели, не разметка патологии. Модель учитывает SagT1, SagT2 и AxT2.</p><p><a href="fusion.json">Результаты JSON</a> · <a href="report-evidence.json">Изображения и источники</a></p></footer></main></body></html>']
    return '\n'.join(parts) + '\n'


def render_markdown(report, bundle):
    bundle = build_evidence(report) if bundle is None else bundle
    facts = {f["branch_id"]: f for f in bundle["facts"]}
    branches = {b["branch_id"]: b for b in report["branches"]}
    views = {v["branch_id"]: v for v in bundle["stage_views"]}
    discs = {v["native_level"]: v for v in bundle["disc_views"]}
    def picture(view):
        if view is None:
            return "Нет снимка"
        path = quote(bundle["images"][view["image_id"]]["source_file"], safe="/")
        return '![Кадр](<' + path + '>)'
    lines = ['# МРТ поясничного отдела', '', 'Исследовательская автооценка · ' + _m(report["study_id"]), '']
    if report["synthetic"]:
        lines += ['**Синтетический пример.**', '']
    has_disc = bundle.get('disc_analysis_status') == 'spinenet_binary_unreviewed'
    note = 'SpineNet: бинарный признак грыжи. Размер, сторона и тип не определены.' if has_disc else 'LDH не подключён. Наличие, размер и сторона грыжи не определены.'
    sequence = 'Sagittal T2 · вход SpineNet' if has_disc else 'Axial T2'
    lines += ['## '+TITLES[0], '', note, '',
              '| Диск | Результат | '+sequence+' |', '|---|---|---|']
    for level in LEVELS:
        lines.append('| {} | {} | {} |'.format(_level(level), _m(_disc_result(report, level, bundle)), picture(discs.get(level))))
    for zone, seq, title in (("foraminal", "Sagittal T1", TITLES[1]), ("subarticular", "Axial T2", TITLES[2])):
        lines += ['', '## '+title, '', '| Уровень | Слева · '+seq+' | Справа · '+seq+' |', '|---|---|---|']
        for level in LEVELS:
            cells = []
            for side in ("left", "right"):
                key = zone+'-'+level+'-'+side
                cells.append(_m(_grade(facts[key], branches[key])+_marked(branches[key]))+' '+picture(views.get(key)))
            lines.append('| {} | {} | {} |'.format(_level(level), *cells))
    lines += ['', '## '+TITLES[3], '']
    lines += ['- '+_m(text) for _,text in fusion_lines(report,bundle)] or ['- '+_empty_fusion(bundle)]
    lines += ['', '† Условная анатомическая привязка; нумерация/совмещение требуют подтверждения.',
              'Норма/лёгкий — единый класс, не исключает изменений. ROI — вход модели, не разметка патологии. Модель: SagT1/SagT2/AxT2.', '',
              '[Результаты JSON](fusion.json) · [Изображения и источники](report-evidence.json)', '']
    return '\n'.join(lines)


_CSS = """
*{box-sizing:border-box}body{margin:0;background:#f3f5f6;color:#152735;font:15px/1.4 system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:28px;background:white}h1{font-size:24px;margin:0}header p,footer{font-size:12px;color:#526370}h2{font-size:19px;margin:28px 0 12px}section>p{font-size:13px}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:10px;border:1px solid #dbe2e6;text-align:left;vertical-align:top;overflow-wrap:anywhere}thead{background:#eef2f4}th:first-child{width:95px}td strong{display:block;font-weight:600;margin-bottom:8px}.images{display:flex;align-items:center;justify-content:center;gap:8px;background:#0c1219;margin-top:6px}.images svg{display:block;width:58%;height:210px}.images svg+svg{width:40%;height:150px}#discs .images svg{width:100%;height:180px}#discs td:nth-child(2){width:30%}#fusion li{padding:9px 0;border-bottom:1px solid #e5eaed;line-height:1.5}.missing{font-size:12px;color:#6a7780}footer{margin-top:25px;border-top:1px solid #dbe2e6}footer p{margin:7px 0}a{color:#245e7a}.note{font-weight:600}@media(max-width:650px){main{padding:14px}body{font-size:13px}th,td{padding:6px}th:first-child{width:65px}.images svg{height:160px}.images svg+svg{height:95px}}@media print{body{background:white;font-size:10pt}main{padding:0}tr{break-inside:avoid}.images svg{height:160px}h2{break-after:avoid}a{color:inherit}}
"""
