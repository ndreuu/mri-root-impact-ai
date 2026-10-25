# Набор МРТ → отчет: RSNA 3139265320

Реальный контрольный пример MRI Report Tool 0.2.0. Исходные снимки и сохраненный выход лежат в репозитории; модели для просмотра не нужны.

## Вход

- [Исходные DICOM](input/dicom/): 60 файлов, 3 560 654 байта.
- [study.template.json](input/study.template.json): серии, изображения, SHA-256 и ограничения.
- [models.example.json](input/models.example.json): пример конфига с бинарным признаком грыжи SpineNet; пути к установленным моделям нужно заменить.

| Серия | Файлов |
|---|---:|
| Sagittal T2/STIR-like | 15 |
| Sagittal T1 | 15 |
| Axial T2 | 30 |

В шаблоне пути относительные к `input/`, чтобы пример переносился вместе с клоном. Рабочий C001 требует абсолютные пути: команда `prepare` ниже создает такой manifest, не изменяя DICOM.

## Выход

**[Открыть отчет со снимками прямо на GitHub](output/report.md).**

- [report.html](output/report.html) — тот же отчет в одном HTML-файле; скачать и открыть локально.
- [fusion.json](output/fusion.json) — исходные формальные выводы, побайтно сохраненные из запуска `d004-001`.
- [report-evidence.json](output/report-evidence.json) — факты, привязки к кадрам и изображениям.
- [images/](output/images/) — 12 исходных PNG из отчета, без изменения пикселей.

В отчете четыре раздела: диски, фораминальный стеноз, субартикулярный стеноз, Fusion. Модельный бинарный признак грыжи — на L3–L4. Умеренные стенозы — слева на L3–L4 и L4–L5. Нумерация и совмещение не подтверждены: формальные 20 ветвей остаются неопределенными, именованных корешковых кандидатов нет. Размер, сторона и тип грыжи не определены.

Это пример разработки на RSNA train-case, не независимая проверка точности.

## Проверить и повторить

Из корня репозитория, Python 3.9+, без моделей и сети:

```sh
python3 scripts/example.py verify
python3 scripts/example.py prepare --output /tmp/mri-example-input
```

`verify` сверяет SHA файлов и соответствие отчета действующему renderer. `prepare` создает `/tmp/mri-example-input/study.json` с абсолютными путями к включенным DICOM. Папка должна быть новой и находиться вне Git.

Для нового MRI inference нужны установленные SpineNet/RSNA и их Python-окружения, описанные в [инструкции тулзы](../../docs/tool.md). Скопируйте `input/models.example.json` во внешний конфиг и заполните пути:

```sh
python3 scripts/run_mri.py \
  --study /tmp/mri-example-input/study.json \
  --config /path/to/local-models.json \
  --output /tmp/mri-example-new-run
```

Результат — `/tmp/mri-example-new-run/fusion/report.html`. Не используйте `output/` этого примера как папку нового запуска. Включенные `allow_missing_modality` и `allow_unverified_registration` отражают ограничения именно контрольного исследования; подтверждения анатомии они не дают.

Сохраненный пример — переносимый экспорт представления, а не полный frozen runtime bundle. Поэтому для него используется `scripts/example.py verify`; `run_mri.py --verify-only` предназначен для полного нового запуска со всеми промежуточными артефактами.

## Происхождение

[RSNA 2024 Lumbar Spine Degenerative Classification Challenge](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification). Источник, закрепленная версия mirror, SHA исходного запуска и преобразования экспорта записаны в [provenance.json](provenance.json); список файлов — [files.sha256.json](files.sha256.json).

Исходные [условия RSNA](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules) и [некоммерческие исследовательские условия SpineNet, включая outputs](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/LICENCE.md) сохраняются; этот пример не переиздает материалы под иной лицензией. Остальные исследования, веса и runtime-артефакты в Git не включаются.
