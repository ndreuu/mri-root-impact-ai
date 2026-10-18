# L002 — первый реальный запуск SpineNet

19 октября 2026 года выполнен свежий CPU-запуск `detect_vb` на Sagittal T2/STIR контрольного RSNA. Модель вернула семь native детекций: T12, L1, L2, L3, L4, L5, S1; всего 71 полигон на 15 исходных кадрах. Первый измеренный вызов `detect_vb` занял 6.92 секунды на двух CPU-потоках Apple M5. Это измерение одного случая, без времени создания окружения, импортов и загрузки checkpoints.

Подключена только локализация тел позвонков. Нумерация остается `indeterminate` до независимого анатомического review; пять дисковых ROI имеют `not_run`. Встроенный grading не вызывался, анализ грыж, стенозов и fusion не выполнялся.

## Окружение и внешние артефакты

Основной harness остается Python 3.9+ со стандартной библиотекой. Модель запускается отдельным процессом в `.venv-spinenet`, с полным проверенным lock [requirements-spinenet.txt](../requirements-spinenet.txt): Python 3.12.14, NumPy 1.26.4, Torch 2.5.1, Torchvision 0.20.1, pydicom 3.0.2. Установка проверена на macOS ARM64; `.venv-data` не менялась. Для восстановления окружения:

```sh
python3.12 -m venv .venv-spinenet
.venv-spinenet/bin/python -m pip install -r requirements-spinenet.txt
.venv-spinenet/bin/python -m pip check
```

Проверенный внешний checkout с весами: `/Users/andrew/Projects/DorsopathyCDS/var/upstream/SpineNet`. Источник и четыре checkpoint закреплены [inventory L001](../harness/upstreams/spinenet.json). Они переиспользованы без изменений и повторной загрузки. Перед каждым inference проверяются commit, байты всех Python-файлов и лицензии по Git blobs, отсутствие дополнительных Python-файлов и SHA каждого checkpoint. Флаги Git `assume-unchanged` не обходят проверку. В каждой папке весов допускается ровно один ожидаемый `.pt`, чтобы выбор upstream по времени создания был однозначным. Существующие upstream `.pyc` не используются.

При восстановлении cache с нуля нужно клонировать [upstream](https://github.com/rwindsor1/SpineNet), checkout commit `5e96ac54b1c1845a37d12e40bfd16f647d40acd8` и получить четыре файла по URL из inventory. Штатный `spinenet.download_weights()` загружает их в `spinenet/weights`; полноту и SHA затем проверяет наш runner. Условия исследовательского использования описаны в [L001](localizer-audit.md).

## Запуск

Из корня проекта. На текущей машине окружение уже создано. `--output` должен указывать на **новую** папку вне Git; для повторного inference замените `l002-001` на новое имя:

```sh
MRI_DATA_ROOT=/Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320
SPINENET_ROOT=/Users/andrew/Projects/DorsopathyCDS/var/upstream/SpineNet
RUN_DIR="$MRI_DATA_ROOT/localization/l002-001"

python3 -m mri_root_impact localize-spinenet \
  --study "$MRI_DATA_ROOT/prepared/study.json" \
  --source "$SPINENET_ROOT" --python .venv-spinenet/bin/python \
  --output "$RUN_DIR" --allow-missing-modality --threads 2 --timeout 600
```

`--allow-missing-modality` нужен только для явно принятого ограничения этого RSNA-входа из I001. Другой явно указанный тип, например CT, не допускается. Серия выбирается как единственная `sagittal_t2`; если таких несколько, требуется `--series-id`.

Успех: exit 0, `ok=true`, `inference_performed=true`, путь к результату. Ошибка: exit 2 и JSON в stderr; сохраненные логи и `run.json` не превращаются в отрицательную находку. Timeout завершает worker и оставляет отчет. Пустой набор подтверждающих файлов или один только exit 0 не считаются успехом.

## Что сохраняется

Все результаты находятся вне Git в `localization/l002-001/`:

- `localization.json` — C001 module_result с native observations, неопределенностью и ссылками на исходные кадры.
- `spatial.json` — [геометрический контракт 1.0](spatial-contract.md): DICOM LPS, slice-to-source mapping, native polygons и покрытие уровней.
- `native.json` — неизмененные поля `detect_vb` после перевода NumPy-объектов в JSON.
- `overlay.png` и `overlays/slice-*.png` — монтаж всех кадров и отдельные наложения. Изображение имеет исходный масштаб 1:1; сверху добавлена полоса 28 пикселей. Полигон рисуется только на том кадре, где его вернула модель.
- `execution.json` — версии всех пакетов, хеши upstream/adapter-кода и весов, параметры, длительность и хеши результатов.
- `worker.log`, `run.json` — фазы выполнения, stdout/stderr, exit code и итог launcher. `run.json` со статусом `completed` подтверждает выполненный технический прогон.

Runner проверяет обязательные четыре артефакта, native/C001/spatial согласованность, каждое наложение, его размер, SHA и исходный кадр. `weights_sha256` в C001 — SHA-256 канонического JSON словаря `relative_filename → {sha256,size_bytes}` для всех четырех весов; определение также записано в execution.

Проверка сохраненного результата без модели и Torch:

```sh
python3 -m mri_root_impact validate \
  --study "$MRI_DATA_ROOT/prepared/study.json" \
  --result "$RUN_DIR/localization.json" --spatial "$RUN_DIR/spatial.json"
python3 scripts/fetch_rsna_control.py --output "$MRI_DATA_ROOT/raw" \
  --study-id 3139265320 --verify-only
```

`inference_performed=false` в ответе валидатора означает, что сам вызов `validate` не выполняет inference; это не отменяет сохраненный реальный запуск. Контрактная проверка не читает DICOM. Исходные геометрия и хеши проверяются loader перед inference и повторной проверкой source SHA после него.

## Проверка результата и ограничения

Loader сохраняет нативные пиксели и порядок осей, сортирует по физической позиции. Первый вариант принимает только однофреймовые MONOCHROME2 с близкой к стандартной sagittal ориентацией, единым изотропным in-plane spacing и регулярным шагом без сдвига в плоскости. Неподдерживаемые orientation, anisotropy, enhanced/multiframe и инвертированная полярность отклоняются, без неявных flips и resampling. Толщина 4 мм и шаг центров около 5 мм сохраняются раздельно.

Независимый технический QC первого прогона подтвердил SHA всех 60 исходных DICOM, геометрию 15 выбранных кадров, 71 полигон и 284 вершины. Все вершины внутри кадров; максимальная ошибка LPS→pixel составляет `8.53e-14` пикселя. Наложения просмотрены: расположение и масштаб изображений согласованы, технического зеркального отображения или смещения не обнаружено. Это проверка координат и отображения, без экспертного подтверждения абсолютной анатомической нумерации. Отчет сохранен вне Git рядом с результатом как `technical-qc.json`.

T12 сохраняется как native метка с C001 target `unknown`, а не переименовывается в L1. Даже наличие всех шести ожидаемых меток не дает `full_expected_numbering=true`. Воксельных масок, ROI дисков и пространственного переноса на T1/AxT2 здесь нет. Следующий шаг — D001: уточнить реальные входы и выходы LDH-DL-MODEL; затем L003 — адаптировать привязку дисковых ROI.
