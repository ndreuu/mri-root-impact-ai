# S003 — первый MRI inference стенозов

19 октября 2026 года выполнен реальный CPU inference выбранных RSNA-моделей на контрольном study `3139265320`. Получены **20 native оценок**: пять уровней × две стороны × два вида стеноза. Автоматически построены 15 keypoints, 210 source-frame ROI и десять комплектов входов 15 SagT1 / 15 SagT2-like / 10 AxT2. Разметка `train.csv` и `train_label_coordinates.csv` не использовалась в runtime.

Постоянный результат находится вне Git: `/Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/stenosis/s003-001`. Модели, версии и SHA — в [S002](rsna-assets.md). Это сокращенная multi-view конфигурация, не полный призовой ансамбль и не независимая оценка точности на training-case.

## Существенное ограничение совмещения

Три серии контрольного исследования имеют **разные FrameOfReferenceUID**. Это не подтверждает общую систему координат, даже при одном StudyInstanceUID. Исходные заголовки не исправлялись. По умолчанию pipeline требует одинаковый непустой UID на всех исходных кадрах: при конфликте или отсутствии UID перенос уровней на AxT2 блокируется, classifier не вызывается, все 20 оценок стеноза остаются `not_run`, без отрицательной находки.

Для данного технического прогона явно указан `--allow-unverified-registration`: используется **неподтвержденное identity-LPS допущение** между сериями. Оно записано в команде, execution, native и spatial. Это позволяет отладить вычисление и просмотреть ROI; допущение не становится выполненной регистрацией. Все пять локализаций и 20 оценок стеноза сохраняются в C001 как **`indeterminate`**, с native классами/вероятностями и причинами неопределенности. Нумерация RSNA также независимо не подтверждена. `root_impacts=null`; fusion и клиническая интерпретация не выполнялись.

## Выполненный путь

1. Проверка полного runtime lock, pinned Git source и SHA трех checkpoint до модели. Отдельный worker на CPU, seed0, два потока, `weights_only=True`, strict state_dict. Из upstream исполняются только пять model class definitions, top-level `predict.py` не импортируется.
2. SHA-проверка и декодирование 60 single-frame MONOCHROME2 DICOM. Пропущенный Modality допускается только через `--allow-missing-modality`; CT не принимается. Сохраняются индивидуальные IOP/IPP/PixelSpacing, InstanceNumber, source ref и FrameOfReferenceUID.
3. Авторский per-frame percentile0.1/99.9 → uint8; keypoint input512→380 и Normalize(.5,.5). Пять sagittal каналов соответствуют native L1-L2…L5-S1, axial — центру ROI. Используется центр bbox маски heatmap ≥99.8-й перцентиль, а не argmax. Плоский/нечисловой сигнал не заменяется центром по умолчанию.
4. Границы уровней AxT2 определяются midpoint/extrapolation сегментами по автоматическим SagT2 центрам. Проверяется пересечение с собственной плоскостью каждого аксиального кадра и попадание пересечения в его FOV. На этом случае выбраны пять групп по пять кадров; еще пять кадров находятся вне целевых сегментов и не получают принудительный уровень.
5. Для каждого уровня выбирается одна группа с одинаковым IOP: максимальное число кадров, затем минимальное среднее расстояние до центра уровня, затем лексикографический tie-break. Все шесть исходных ориентаций сохраняются; один регулярный affine на 30 кадров не строится. Axial центр проецируется через LPS на каждый выбранный кадр.
6. Dataset-v9 crops: Sag35×50мм со смещением−5мм по высоте; Ax45×75мм, +5мм. Sag side sampling идет от наружной половины к середине; Ax справа берутся первые60% crop и горизонтальное отражение, слева последние60%. Равномерно выбираются 15/15/10 views с явными повторами source refs, resize224, Normalize(.5,.5).
7. Classifier0203 возвращает `[NFN,SS]` × `[normal_mild,moderate,severe]`, softmax(logits /0.9964720472122109), без TTA. Пустой/выходящий за FOV ROI, отсутствующий вход или невалидная геометрия не подменяются нулевым тензором или классом Normal/Mild.

Намеренные отличия от upstream: исправлена формула DICOM row/column spacing; детерминирована группа Ax вместо смешивания ориентаций/случайной серии; добавлены FOV, UID и пустые-input проверки; first adapter требует SagT2 для axial linkage и не делает неявный fallback на SagT1. В manifest допускается одна серия каждого требуемого типа. Другие ориентации и enhanced multi-frame DICOM пока отклоняются явно.

## Артефакты и проверка

| Файл | Содержимое |
|---|---|
| `localization.json` | C001: пять native дисковых уровней с неопределенной анатомической нумерацией |
| `foraminal.json`, `subarticular.json` | По десять C001 assessment, native шкала RSNA, вероятности трех классов, source/upstream refs |
| `native.json` | Исходные вероятности, class order, модельная provenance, ограничения, policy и фактические счетчики вызовов |
| `spatial.json` | Геометрия 60 исходных кадров, keypoints, 210 ROI, порядок и повторы всех classifier views |
| `overlay.png`, `overlays/*.png` | Монтаж и 60 source overlays, явно помеченные unreviewed |
| `roi_montage.png` | Средний фактически выбранный view каждой серии/стороны/уровня |
| `execution.json`, `run.json` | Runtime, hashes, timings, параметры, exit code/timeout и результат независимой проверки выхода |
| `technical-qc.json` | Повторное чтение исходных DICOM headers и независимая проверка pixel/LPS/footprints |
| `reproducibility.json` | Совпадение SHA двух свежих запусков и проверка 20 `not_run` в строгом режиме |

Runner не принимает один exit0 как успех: требует семь обязательных артефактов и все overlays, сверяет SHA, provenance и точное восстановление C001 из native. Проверяются полное покрытие targets, вероятности, same-level upstream refs, физические crops/стороны, 15/15/10 source mappings и неизменность исходных байтов. Нативная оценка остается исходной модельной оценкой; Normal/Mild объединяет норму и легкие изменения.

Координаты DICOM относятся к центрам пикселей: IPP — центр(0,0). Crop box задается `[left,top,right,bottom)`; его физические края находятся на index−0.5. В `spatial.json` сохранены source footprint corners, resize и flip. Для output pixel center `(u,v)` без flip обратное отображение имеет вид `x=left−0.5+(u+0.5)*(right−left)/224` и аналогично для y; при flip x отсчитывается от правого края. Это геометрия входа модели, не сегментация поражения.

Наложения просмотрены: проверены масштаб/положение на source frames и отсутствие выхода используемых ROI за границы. Независимый QC повторно проверил 60 DICOM, 210 ROI footprints и 15 keypoints: max corner error0мм, LPS→pixel error8.53e-14 пикселя. Это технический QC, не независимое подтверждение анатомических номеров или регистрации. Два свежих запуска дают идентичные SHA семи C001/native/spatial/PNG артефактов и 60 overlays. Вычисления моделей заняли 4.76 и 4.52с на двух CPU-потоках; первый worker целиком — 10.16с. Строгий режим отдельно проверен: два sagittal keypoint вызова, ноль classifier вызовов, все 20 стенозов `not_run` с `observation=null`.

## Повторный запуск

Из корня проекта, после подготовки [контрольного исследования](control-study.md) и [S002](rsna-assets.md):

```sh
MRI_DATA_ROOT=/Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320
RSNA_MODEL_ROOT=/Users/andrew/Datasets/mri-root-impact-ai/models/rsna2024-moyasii-v2
RSNA_RUN="$MRI_DATA_ROOT/stenosis/s003-reproduce-001"
python3 -m mri_root_impact stenosis-rsna \
  --study "$MRI_DATA_ROOT/prepared/study.json" \
  --source "$RSNA_MODEL_ROOT/source" --weights "$RSNA_MODEL_ROOT/weights" \
  --python "$PWD/.venv-rsna/bin/python" --output "$RSNA_RUN" \
  --threads 2 --timeout 180 --allow-missing-modality \
  --allow-unverified-registration
.venv-rsna/bin/python scripts/qc_rsna_run.py \
  --study "$MRI_DATA_ROOT/prepared/study.json" --run "$RSNA_RUN" \
  --output "$RSNA_RUN/technical-qc.json"
python3 -m mri_root_impact validate --study "$MRI_DATA_ROOT/prepared/study.json" \
  --result "$RSNA_RUN/localization.json" --result "$RSNA_RUN/foraminal.json" \
  --result "$RSNA_RUN/subarticular.json"
```

Output должен быть новым каталогом вне Git; предыдущие результаты не перезаписываются. Для проверки строгой policy убрать `--allow-unverified-registration` и выбрать другой новый output. QC читает исходные заголовки и проверяет согласованность, но не запускает модель повторно. Подробный RSNA spatial проверяется checked runner/QC; старый `validate --spatial` предназначен для отдельного SpineNet spatial artifact.

Следующий L003 сопоставит native дисковые уровни RSNA с локализацией SpineNet и уточнит геометрические ограничения. LDH, fusion и независимая оценка точности остаются отдельными задачами.
