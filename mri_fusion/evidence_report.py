"""Physician reading order: MRI views, source facts, anatomy, formal quality."""

from urllib.parse import quote

from .evidence import build_evidence
from .contracts import FusionContractError
from .reporting import _h, _m, _title, _root, _html_overview, _coverage, GRADES, LIMITS


PAPER = "https://www.kjronline.org/DOIx.php?id=10.3348/kjr.2017.18.1.249"
VIEW_NOTE = ("Показан один из фактических входных кадров: средний отсчёт в упорядоченном наборе модели, "
             "а не автоматически найденный наиболее выраженный стеноз. Рамки — области входа модели (ROI), "
             "не сегментация грыжи, стеноза или нервного корешка. Изображения сохраняют исходную ориентацию.")
ANATOMY_NOTE = ("Ниже применяется анатомическое соответствие к исходным меткам уровня и стороны. "
                "Оно условно: верно, если эти метки и зона подтверждаются при просмотре. "
                "Соответствие корешку само по себе не устанавливает воздействие на него.")


def summary(bundle):
    facts = bundle["facts"]
    counts = {grade: sum(f["native_grade"] == grade for f in facts) for grade in GRADES}
    return ("Получены исходные оценки: умеренный стеноз — {}; выраженный — {}; объединённый класс "
            "«норма / лёгкий стеноз» — {}. Для {} сочетаний принятого класса нет. "
            "Степени и метки уровня ниже предварительные до подтверждения анатомической привязки.").format(
                counts["moderate"], counts["severe"], counts["normal_mild"],
                sum(f["native_grade"] is None for f in facts))


def relation(fact):
    root = fact["conditional_anatomy"]
    if root is None:
        return "Анатомическое соответствие не построено: исходная оценка неизвестна или непринята."
    return "Если исходные метки верны: {} → {}.".format(_title(fact["native_target"]), _root(root))


def effect(fact):
    if fact["rule_effect"] == "severe_research_threshold":
        return ("Выраженная степень соответствует исследовательскому порогу потенциального воздействия. "
                "Для итогового кандидата требуются пригодный исходный статус и подтверждения привязки.")
    if fact["rule_effect"] == "moderate_relation_only":
        return ("Умеренная степень: указан анатомически соответствующий корешок; "
                "порог положительного вывода в текущей версии не установлен.")
    if fact["rule_effect"] == "no_severe_criterion":
        return "Критерий выраженного стеноза по исходному классу не выполнен; объединённый класс не исключает лёгких изменений."
    return "Правило степени не применяется: исходные классы отсутствуют, противоречивы или не поддерживаются."


def _grade(fact):
    if fact["native_grade"] is not None:
        return GRADES[fact["native_grade"]]
    return " / ".join(o["label"] for o in fact["native_observations"]) or "Исходная оценка отсутствует"


def _figure(view, bundle, crop=True):
    image = bundle["images"][view["image_id"]]
    if not image["data_uri"].startswith("data:image/png;base64,"):
        raise FusionContractError("Report illustrations must be embedded PNGs, never network resources")
    uri = _h(image["data_uri"])
    width, height = image["width"], image["height"]
    contents = '<image href="{}" width="{}" height="{}"/>'.format(uri, width, height)
    zoom = ""
    if crop:
        x0, y0, x1, y1 = view["source_crop_box"]
        y0, y1 = y0 + view["header_pixels"], y1 + view["header_pixels"]
        contents += '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#ffcc33" stroke-width="2"/>'.format(x0, y0, x1-x0, y1-y0)
        zoom = '<svg role="img" aria-label="Увеличение ROI на исходном размеченном кадре" viewBox="{} {} {} {}">{}</svg>'.format(x0, y0, x1-x0, y1-y0, contents)
    ref = view["image_ref"]
    caption = "Серия {} · изображение {} · индекс кадра {}".format(ref["series_id"], ref["image_id"], ref["frame_index"])
    return ('<figure><div class="source-images"><svg role="img" aria-label="Исходный кадр с областями модели" '
            'viewBox="0 0 {} {}">{}</svg>{}</div><figcaption>{}</figcaption></figure>').format(
                width, height, contents, zoom, _h(caption))


def render_html(report, bundle, quality_html):
    bundle = build_evidence(report) if bundle is None else bundle
    facts = {f["branch_id"]: f for f in bundle["facts"]}
    # Reuse the checked detailed quality trace, but keep it after clinical facts.
    css = quality_html.split('<style>', 1)[1].split('</style>', 1)[0]
    quality = quality_html.split('<section id="algorithm">', 1)[1].split('</main>', 1)[0]
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>МРТ: снимки, находки и анатомический разбор</title><style>' + css + _CSS + '</style></head><body><main>',
             '<header><p class="eyebrow">Исследовательский МРТ-отчёт · Stage I</p><h1>Снимки → находки → анатомическое правило</h1><p>Исследование: <strong>' + _h(report["study_id"]) + '</strong></p></header>']
    if report["synthetic"]:
        parts.append('<p class="notice">Синтетический пример: изображения и подтверждения не являются данными пациента.</p>')
    parts += ['<section id="conclusion" class="conclusion"><h2>Что получено</h2><p class="lead">' + _h(summary(bundle)) + '</p><p>Ниже показаны результаты отдельных этапов и условное анатомическое объяснение. Достоверность итогового корешкового вывода проверяется отдельно.</p></section>',
              '<nav><a href="#foraminal-stage">Фораминальный стеноз</a> · <a href="#subarticular-stage">Субартикулярный стеноз</a> · <a href="#disc-stage">Диски</a> · <a href="#facts">Факты и fusion</a></nav>',
              '<p>Выбранная модель оценивает стеноз по совокупности Sagittal T1, Sagittal T2 и Axial T2. Галереи ниже выделяют запрошенную последовательность для просмотра, а не представляют её единственным входом классификатора.</p>']
    for zone, sequence, title in (("foraminal", "Sagittal T1", "Фораминальный стеноз: слева и справа"),
                                  ("subarticular", "Axial T2", "Субартикулярный стеноз: слева и справа")):
        parts.append('<section id="{}-stage"><h2>{} · {}</h2><p>{}</p><div class="gallery">'.format(zone, _h(title), sequence, _h(VIEW_NOTE)))
        views = [v for v in bundle["stage_views"] if v["native_target"]["zone"] == zone]
        for view in views:
            fact = facts[view["branch_id"]]
            parts += ['<article class="image-card"><h3>' + _h(_title(view["native_target"])) + '</h3><p><strong>Предварительная оценка: ' + _h(_grade(fact)) + '</strong></p>',
                      _figure(view, bundle), '<p class="muted">Рамка выделена для этой оценки; справа — увеличение той же области. Отсчёт {} из {} входов {}. Переворот для модели: {}.</p></article>'.format(
                          view["model_sample_index"]+1, view["model_sample_count"], sequence, "да" if view["model_horizontal_flip"] else "нет")]
        if not views:
            parts.append('<p>Проверенные иллюстрации этой последовательности не предоставлены. Отсутствие картинки не считается отрицательной находкой.</p>')
        parts.append('</div></section>')
    parts += ['<section id="disc-stage"><h2>Анализ межпозвонковых дисков</h2><p class="notice">Детектор грыж LDH ещё не подключён. Наличие, размер и положение грыжи автоматически не определены. Ниже — доступные Axial T2 кадры для осмотра дисков, а не результат обнаружения грыжи.</p><div class="gallery">']
    for view in bundle["disc_views"]:
        parts += ['<article class="image-card"><h3>Исходная метка ' + _h(view["native_level"]) + '</h3>', _figure(view, bundle, False), '</article>']
    if not bundle["disc_views"]:
        parts.append('<p>Проверенные кадры дисков не предоставлены. Исходные оценки дисков, если есть, сохранены в подробностях.</p>')
    parts += ['</div></section><section id="facts"><h2>Список фактов и применение fusion-правил</h2><p>' + _h(ANATOMY_NOTE) + '</p><p>Фораминальная зона → выходящий корешок верхнего позвонка; субартикулярная зона → проходящий корешок нижнего позвонка. Для L5–S1 это L5 и S1 соответственно.</p><div class="table-scroll"><table><thead><tr><th>Исходная область</th><th>Предварительный факт</th><th>Условное применение анатомии</th><th>Интерпретация степени</th></tr></thead><tbody>']
    for fact in bundle["facts"]:
        parts.append('<tr id="{}"><th scope="row">{}</th><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            _h(fact["fact_id"]), _h(_title(fact["native_target"])), _h(_grade(fact)), _h(relation(fact)), _h(effect(fact))))
    parts += ['</tbody></table></div><p>Контекст грыжи не оценён, поэтому наличие грыжи и её причинная роль в эти факты не добавляются.</p>',
              '<details><summary>Пример правила, не находка этого исследования</summary><p>L4–L5 + левосторонняя грыжа + выраженный левосторонний субартикулярный стеноз → анатомическая гипотеза потенциального воздействия на левый проходящий корешок L5. Для вывода о данном исследовании каждый исходный факт должен быть установлен отдельно.</p></details>',
              '<p>Основание выбора признаков: <a href="' + PAPER + '">Sung et al., 2017, таблица 2</a>. В статье оцениваются контакт, смещение и компрессия; её шкала 0–3 не конвертируется автоматически в классы RSNA.</p></section>',
              '<section id="formal-quality"><h2>Достоверность итогового корешкового вывода</h2><p>' + _h(LIMITS) + '</p><p>' + _h(_coverage(report)) + '</p><p>Итоговых кандидатов после всех проверок: <strong>' + str(len(report["root_candidates"])) + '</strong>. Оценок с неподтверждённым итогом: <strong>' + str(report["coverage"]["indeterminate"]) + ' из 20</strong>. Этот статус не скрывает перечисленные выше находки модели.</p>' + _html_overview(report) + '<details><summary>Подробные проверки привязки, источники и исходные вероятности</summary><section id="algorithm">' + quality + '</details></section></main></body></html>']
    return '\n'.join(parts) + '\n'


def render_markdown(report, bundle, quality_markdown):
    bundle = build_evidence(report) if bundle is None else bundle
    lines = ['# МРТ: снимки, находки и анатомическое правило', '', 'Исследование: ' + _m(report["study_id"]), '',
             '**Исследовательский предварительный отчёт.**', '', summary(bundle), '',
             '## Изображения этапов', '', VIEW_NOTE, '']
    if report["synthetic"]:
        lines += ['**Синтетический пример; это не заключение о пациенте.**', '']
    facts = {f["branch_id"]: f for f in bundle["facts"]}
    for view in bundle["stage_views"]:
        image = bundle["images"][view["image_id"]]
        lines += ['### ' + _m(_title(view["native_target"])) + ' · ' + view["sequence"], '',
                  'Предварительно: ' + _m(_grade(facts[view["branch_id"]])), '',
                  '![Проверенный исходный кадр](<' + quote(image["source_file"], safe="/") + '>)', '',
                  'ROI исходного изображения: ' + str(view["source_crop_box"]) + '. Увеличение показано в HTML.', '']
    if not bundle["stage_views"]:
        lines += ['Проверенные изображения не предоставлены.', '']
    lines += ['## Межпозвонковые диски', '', 'Детектор грыж не подключён; наличие, размер и положение грыжи не определены.', '']
    for view in bundle["disc_views"]:
        lines += ['Исходная метка ' + _m(view["native_level"]) + '; кадр для осмотра, не результат детектора грыж.', '',
                  '![Axial T2 для осмотра диска](<' + quote(bundle["images"][view["image_id"]]["source_file"], safe="/") + '>)', '']
    lines += ['## Факты → анатомическое правило', '', ANATOMY_NOTE, '']
    for fact in bundle["facts"]:
        lines += ['- **' + _m(_title(fact["native_target"])) + ':** ' + _m(_grade(fact)) + '. ' + _m(relation(fact)) + ' ' + _m(effect(fact))]
    lines += ['', 'Контекст грыжи не оценён; её наличие и причинная роль не добавляются.', '',
              '[Sung et al., таблица 2](' + PAPER + '): критерии контакта, смещения и компрессии; шкала статьи не приравнивается к RSNA.', '',
              '## Достоверность итогового корешкового вывода', '',
              _coverage(report), '',
              'Кандидатов после всех проверок: {}. Неопределённых итоговых оценок: {} из 20.'.format(len(report["root_candidates"]), report["coverage"]["indeterminate"]), '',
              quality_markdown[quality_markdown.index('## Как получен вывод'):]]
    return '\n'.join(lines)


_CSS = """
nav{margin:22px 0}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.image-card{border:1px solid #d4dfe5;border-radius:6px;padding:16px;background:#fafcfd}.image-card h3{margin-top:0}.source-images{display:flex;align-items:center;gap:12px;background:#0d151c;padding:8px;min-height:260px}.source-images svg{width:58%;max-height:310px;background:#0d151c}.source-images svg+svg{width:40%;image-rendering:auto}figure{margin:12px 0}figcaption{font-size:.78rem;overflow-wrap:anywhere;margin-top:7px}.image-card .muted{margin-bottom:0}#facts th{min-width:130px}#facts td{min-width:150px}@media(max-width:760px){.gallery{grid-template-columns:1fr}}@media print{.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.image-card{break-inside:avoid}.source-images{min-height:0}.source-images svg{max-height:200px}}
"""
