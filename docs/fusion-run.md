# F002 — исполняемый MRI fusion

Реализована исследовательская [policy 0.1.0](fusion-spec.md): отдельные `mri_fusion/contracts.py`, `engine.py`, `runner.py` и CLI `scripts/fuse_mri.py`. Нужен Python 3.9+ со стандартной библиотекой. Fusion читает готовые результаты; нейросети не запускает.

[Краткий отчет для врача](physician-report.md) автоматически формируется при каждом успешном запуске. Четыре раздела сохранены: диски, фораминальный стеноз, субартикулярный стеноз и Fusion. D004 добавляет `--disc-run` для проверенного бинарного SpineNet grading, native C001 и входных иллюстраций; подробности — [disc-run.md](disc-run.md). Текущий полный результат — `tool/d004-001`; прежние запуски сохранены. Версия представления 1.3.

F001 policy сохранена побайтно: SHA-256 `c3e574a12603a0a438eb1db50b9ce21b47b53fcae9b5d2900bbbfd3cc8744fed`. Ее историческое поле `implementation_status=specified_not_implemented` описывает момент спецификации. F002 исполняет эту спецификацию; статус реализации отражен здесь и в очереди. Аналогично 34 abstract vectors сохраняют исходное `specified_not_executed`; теперь все они преобразуются в настоящие synthetic C001/review inputs и выполняются в `tests/test_fusion_vectors.py`.

## Исторический контрольный результат без grading

Показанная ниже команда для F005 выполняется кодом commit `4ab0e4c`; новая версия проверяется через [тулзу](tool.md) с D004-конфигом. Из корня соответствующего checkout:

```sh
python3 scripts/fuse_mri.py \
  --study /Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/prepared/study.json \
  --spinenet-run /Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/localization/l002-001 \
  --rsna-run /Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/stenosis/s003-001 \
  --crosscheck-run /Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/geometry/l003-001 \
  --output /Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/fusion/f005-001 \
  --verify-only
```

Для нового запуска убрать `--verify-only` и указать новую внешнюю папку, например `fusion/f002-repeat-001`. Существующие папки не перезаписываются. Выход внутри Git, каталога входных файлов, исходных DICOM или upstream run отклоняется. Все три frozen run directory передаются вместе. Успех — JSON и exit code 0; ошибка структуры, SHA или чтения — JSON error и exit code 2. Корректные неполные данные дают успешный отчет с `indeterminate`.

Frozen-режим повторно проверяет L002/S003 native, C001, spatial, worker/run records, source/code SHA и overlays через существующие валидаторы. L003 пересчитывается по тем же source frames; только его численная арифметика допускает абсолютную разницу до `1e-9` из-за различий суммирования Python 3.9/3.12. Совпадение меток, решения и структуры требуется точное. Сами MRI-адаптеры и их hash-bound package не изменены.

## Контрольный исход

На RSNA 3139265320 результат содержит **20 `indeterminate`, 0 именованных кандидатов и пустые root summaries**. Ни одна ветвь не получает `mapped_root` или `candidate_root`. Это недостаточность подтверждений: источник S003 уже `indeterminate`, нумерация и межсерийное совмещение не подтверждены, независимый review отсутствует. Классы и native probabilities всех 20 оценок сохранены. L003 agreement не становится подтверждением.

`inference_performed=false`, `fusion_performed=true`, `global_root_clearance=false`, `clinical_diagnosis=null`. Отсутствие кандидатов не означает отсутствия воздействия на корешки. Грыжа имеет `herniation_context=not_evaluated`, причина стеноза — `etiology=unassigned`.

Внешняя папка результата содержит:

| Файл | Содержание |
|---|---|
| `fusion.json` | Версия формата 1.0; policy SHA, bindings исходных JSON, 20 ветвей, native evidence, review gates, reasons, summaries и candidates |
| `report.md` | Читаемая таблица native уровня, стороны, зоны, степени, решения и причин неопределенности |
| `report.html` | Отчет для врача с предварительным заключением и раскрываемым разбором каждой оценки; создается автоматически начиная с F003 |
| `report-evidence.json` | F004: предварительные факты, условная анатомия, точный выбор source frame/ROI и встроенные проверенные PNG |
| `run.json` | Статус выполнения, время, SHA всех прочитанных файлов и исходного кода, SHA итоговых JSON, Markdown и HTML |

При завершении F002 два свежих запуска дали побайтно одинаковые `fusion.json` и `report.md`. Исторические SHA: fusion — `7a175e03af51c178987bfdb2567e2ddc9a43b796fb5c5fe863fb2fc42f65dded`, report — `e52bac9a5cf3a6ffa890cee0e1b90c5e8149fccdea64d64ece8c1f1eef63188c`. F003 меняет представление, сохраняя тот же fusion JSON; новый Markdown и HTML проверяются собственными SHA. `run.json` содержит время и потому побайтного совпадения между запусками не требует.

`--verify-only` перечитывает входы, пересчитывает fusion, факты и изображения, сравнивает JSON, evidence JSON, Markdown и HTML с сохраненным результатом. Файловая запись не выполняется. F002/F003/F004 проверяются их кодом: `0d61c27`/`d0d3129`/`28bb52c`. Текущая команда проверяет F005.

## Явные C001 inputs

Для другого bundle можно передать `--result` несколько раз вместо frozen directory arguments:

```sh
python3 scripts/fuse_mri.py --study /external/inputs/study.json \
  --result /external/inputs/localization.json \
  --result /external/inputs/stenosis.json --output /external/fusion/new-run
```

Нужны все результаты, на которые ссылается upstream graph. Пустой набор результатов допустим и дает 20 неопределенных ветвей. У реального исследования каждый image SHA обязателен и сверяется с файлом; этот режим не декодирует снимки и не подтверждает анатомию. Одинаковые повторения result document с одним ID схлопываются; измененное содержимое при том же ID — ошибка. Разные оценки одного target сохраняются отдельно; несовпадающие grades блокируют ветвь.

## Review contract 1.0

Контекст передается отдельно через `--review /external/review/context.json`; evidence documents — повторяемым `--evidence`. Исходный C001 не редактируется и не повышается из `indeterminate` в `assessed`.

| Поле context | Требование |
|---|---|
| `schema_version`, `kind` | `1.0`, `fusion_review_context` |
| `context_id`, `study_id`, `study_digest`, `synthetic` | Идентичность контекста и точное соответствие manifest |
| `input_artifacts_sha256` | Точная карта всех artifact IDs и фактических SHA входных JSON, включая evidence documents |
| `reviews` | Список отдельных записей для source assessment |

Запись review содержит ровно `source_ref`, `target`, `reviewer_id`, `states`, `covered_series_ids`, `evidence_refs`. Source ref — пара `result_id`/`assessment_id`; target обязан совпасть с исходным. Artifact ID результата — `result-<result_id>`, evidence — `evidence-<evidence_id>`; если составной ID длиннее 128 символов, суффикс заменяется первыми 32 символами SHA исходного ID. Bindings сохранены в `fusion.json`.

Для пригодного review нужны `anatomical_numbering=confirmed`, `side_assignment=confirmed`, `source_geometry=verified`, `diagnostic_adequacy=sufficient` и `cross_series_registration=verified` либо `not_required_with_evidence`. Пропущенные states и неполное покрытие серий допустимы как неполные данные, но блокируют именованный вывод. Покрытие включает все серии в транзитивных upstream image refs. Review одного SagT1 не покрывает multi-view RSNA.

Evidence document содержит ровно `schema_version=1.0`, `kind=fusion_review_evidence`, `evidence_id`, `study_id`, `study_digest`, `synthetic`, непустое `description`. Synthetic flag обязан совпадать с manifest. Ссылка на MRI run или согласие локализаторов не заменяет evidence такого типа. Код проверяет привязку и заявленные состояния, но не удостоверяет квалификацию reviewer и достоверность текстового подтверждения. Независимый анатомический review требует фактической экспертной работы; в контрольном запуске он не создавался.

## Проверки и граница результата

34 F001 сценария исполняются через synthetic fixtures, включая L5–S1, обе стороны, многоуровневость, два основания одного L5, partial coverage, дубликаты, конфликты, unknown target, непринятую шкалу грыж, некорректные вероятности и SHA. Дополнительные tests проверяют contracts, невозможность подменить исходный статус, защиту источников, tamper detection и read-only verification.

При завершении F002 все 337 тестов прошли в RSNA runtime; core Python 3.9 прошел 292 и пропустил 45 optional decoder tests. Отдельный harness smoke прошел. Из них 92 новых fusion tests: 42 vector/regression, 26 contract и 24 runner. Проверки нового представления F003 добавлены отдельно.

Эти проверки подтверждают применение правил и сохранность данных. Контрольный train-case и синтетические сценарии не оценивают точность определения корешков. Протокол независимой технической оценки — E001; LDH adapter и его обученные артефакты — D002. Полный Stage I еще не завершен.
