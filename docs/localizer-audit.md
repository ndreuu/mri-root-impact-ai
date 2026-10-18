# L001 — выбор первого локализатора

Проверено 19 октября 2026 года. Выбран **SpineNet v2, `detect_vb`, как исследовательский baseline** для одной сагиттальной T2-like серии. Код и веса доступны, обе сагиттальные серии контрольного RSNA читаются. Модель в этом аудите не запускалась; качество нумерации и наличие всех уровней еще не установлены. Реализация адаптера и первый inference относятся к L002.

Последующее состояние: L002 выполнен, см. [реальный запуск](localization-run.md) и [геометрический контракт](spatial-contract.md). Ниже сохранены выводы именно исходного аудита L001.

## Сравнение

| Инструмент | Проверенная версия | Реальная роль | Решение для первого запуска |
|---|---|---|---|
| [SpineNet](https://github.com/rwindsor1/SpineNet/tree/5e96ac54b1c1845a37d12e40bfd16f647d40acd8) | commit `5e96ac54b1c1845a37d12e40bfd16f647d40acd8`, документация v2.0.0 | Детекция и нумерация тел позвонков, четырехугольники на срезах | Выбран для исследовательской локализации; маски сегментации не обещает |
| [SpineReport](https://github.com/ivadomed/SpineReport/tree/b07ce0fe65fc913e3162822f2ee82202e0be8808) | commit `b07ce0fe65fc913e3162822f2ee82202e0be8808`, package `20260908` | Морфометрия и отчеты по уже готовым сегментациям и landmarks в пространстве 1 мм | В первый локализатор не включается |
| [TotalSpineSeg](https://github.com/neuropoly/totalspineseg/tree/4502d41bcb4a12e44f4be666411461ca81b02d89) | commit `4502d41bcb4a12e44f4be666411461ca81b02d89`, package `20260807`, веса `r20260730` | Двухэтапная сегментация и нумерация позвонков и дисков | Резерв для задачи полноценной сегментации; нужен NIfTI-конвертер и отдельный аудит условий весов |

SpineNet выбран по узкой задаче первого запуска: вход уже состоит из sagittal DICOM, нужны проверяемые метки и координаты, а веса и подходящее окружение есть локально. Это не сравнение точности трех инструментов. Если обязательным результатом станут voxel masks или изменится назначение проекта, выбор backend нужно пересмотреть.

## Условия использования

[Лицензия SpineNet](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/LICENCE.md) ограничивает **код, pretrained weights и outputs** некоммерческими исследованиями. Она отдельно исключает результаты, предназначенные для коммерческого продукта. Текущий выбор относится к исследовательской работе; разрешение перенести этот backend или его результаты в продукт Pain Smart Point AI из этого решения не следует.

У [SpineReport](https://github.com/ivadomed/SpineReport/blob/b07ce0fe65fc913e3162822f2ee82202e0be8808/LICENSE) и [кода TotalSpineSeg](https://github.com/neuropoly/totalspineseg/blob/4502d41bcb4a12e44f4be666411461ca81b02d89/LICENSE) — LGPL v3. Отдельные условия checkpoints TotalSpineSeg в проверенных README, pyproject и описании релиза не установлены; code license не объявляется проверенной лицензией весов. Это ограничение записано, коммерческая пригодность альтернативы не утверждается.

## Что возвращает SpineNet

[`detect_vb`](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/spinenet/main.py) принимает массив `H×W×S`. Обычный результат после [нумерации](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/spinenet/utils/label_verts.py) оставляет четыре поля:

- `predicted_label` — метка позвонка; набор шире L1–S1, код также обрабатывает S2.
- `slice_nos` — индексы срезов входного массива с нуля.
- `polys` — четырехугольники по соответствующим срезам, координаты `(x, y)` в пикселях.
- `average_polygon` — средний четырехугольник; он не является трехмерной маской.

В обычном результате нет откалиброванной уверенности, сегментации диска или физического преобразования к другой серии. Debug features нельзя автоматически считать вероятностью правильной нумерации.

`get_ivds_from_vert_dicts` формирует ресемплированные crops и названия дисков, но не возвращает достаточную геометрию для обратного переноса ROI. Для Stage I понадобится отдельная проверяемая привязка дисков по соседним позвонкам. Встроенный `grade_ivds` в L002 не используется: его шкалы и бинарные классы не подменяют будущие выходы LDH и RSNA.

## Веса и локальные артефакты

Закрепленный [inventory](../harness/upstreams/spinenet.json) содержит URL, размер и SHA-256 четырех файлов. Все локальные байты повторно проверены; сумма размеров — **821 857 187 байт**. Четыре официальных URL также ответили HTTP HEAD 200 с соответствующими размерами. Хеши закреплены по локальным скачанным файлам: авторские checksums для них не заявляются, HEAD не проверяет содержимое удаленного файла.

Проверенный внешний checkout: `/Users/andrew/Projects/DorsopathyCDS/var/upstream/SpineNet`, HEAD соответствует inventory, tracked tree чистый. Веса находятся в его `spinenet/weights/`, вне нового Git-репозитория. В upstream конструктор загружает все четыре модели, включая grading, даже если затем вызвать только `detect_vb`. Каждый loader выбирает `.pt` по времени создания, поэтому L002 должен проверить точный набор файлов и хеши, а не полагаться на наличие папки. [Конструктор](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/spinenet/main.py), [пример loader](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/spinenet/models/vfr.py).

Найдено окружение `/Users/andrew/.codex/worktrees/7179/med diploma/.venv-spinosarc/bin/python`: Python 3.12.14, NumPy 1.26.4, Torch 2.5.1, Torchvision 0.20.1, pydicom 3.0.2. OpenCV headless 4.10.0.84 и Shapely 2.0.7 доступны в `/Users/andrew/Projects/DorsopathyCDS/var/spinenet-deps`. Это подтвержденная инвентаризация, а не автономное окружение нового проекта. L002 должен закрепить собственный воспроизводимый runtime и запускать модель отдельным процессом на CPU.

Окружение `.venv-data` нового проекта не подходит для неизмененного upstream: NumPy 2.5.3 не содержит `ndarray.ptp()`, используемый [при извлечении объемов](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/spinenet/utils/extract_volumes.py). Нужен NumPy 1.26.4 либо явно протестированная правка; базовый harness менять не требуется. Старые checkpoints также требуют закрепления Torch и проверки загрузки: в новых версиях изменился default `weights_only` у `torch.load`. [NumPy migration](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html), [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html).

## Проверка нашего входа без модели

Из закрепленного checkout через `importlib.util.spec_from_file_location` загружен только `spinenet/io/dicom_io.py`, без импорта `spinenet` и Torch. Вызван `load_dicoms` на двух сагиттальных сериях manifest. Подтверждено:

| Параметр | Sagittal T2/STIR | Sagittal T1 |
|---|---|---|
| Массив | `256×256×15` | `256×256×15` |
| PixelSpacing, row/column | `1.0156 / 1.0156 мм` | `1.0156 / 1.0156 мм` |
| SliceThickness | `4 мм` | `4 мм` |
| Шаг центров по нормали | примерно `5 мм` | примерно `5 мм` |
| Ориентация и порядок | единая ориентация, InstanceNumber монотонен по физической позиции | то же |
| Исходные байты | SHA-256 не изменились | SHA-256 не изменились |

Это техническая проверка загрузки 30 изображений. Она не подтверждает видимость всех уровней или точность модели. T2/STIR остается T2-like; исходные ограничения `Modality` и регистрации из I001 сохраняются.

Отчет и программа этого аудита сохраняются вне Git: `/Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/prepared/l001/`. Повтор из корня проекта:

```sh
AUDIT_ROOT=/Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320/prepared/l001
SPINENET_SRC=/Users/andrew/Projects/DorsopathyCDS/var/upstream/SpineNet
git -C "$SPINENET_SRC" rev-parse HEAD
PYTHONDONTWRITEBYTECODE=1 .venv-data/bin/python "$AUDIT_ROOT/audit_spinenet_input.py" \
  "$SPINENET_SRC" "$AUDIT_ROOT/../study.json" /tmp/rsna-spinenet-io-audit.json
```

## Требования к адаптеру L002

[`dicom_io.py`](https://github.com/rwindsor1/SpineNet/blob/5e96ac54b1c1845a37d12e40bfd16f647d40acd8/spinenet/io/dicom_io.py) сортирует по InstanceNumber, усредняет обе оси PixelSpacing в один scalar и использует SliceThickness. Поэтому стандартный loader не становится геометрическим контрактом проекта. Реализация должна:

1. Выбрать ровно одну SagT2-like серию из manifest, проверить SHA и геометрию; собрать `H×W×S` по физическим позициям, сохранив таблицу `slice_index → image_id/frame_index`. Для текущего входа in-plane spacing изотропен; неодинаковые оси нельзя молча усреднять. Геометрию неизвестного или обlique входа нужно отклонить либо преобразовать с обратным отображением.
2. Сохранить исходные PixelSpacing, IOP, IPP и **шаг центров**, отдельно от толщины. Для polygon `(x,y)` точка в DICOM LPS вычисляется как `IPP + x·PixelSpacing[1]·IOP[:3] + y·PixelSpacing[0]·IOP[3:]`. Все flips, сортировки и ресемплирование должны отражаться в обратном преобразовании. Raw DICOM не изменяются.
3. Запустить закрепленный `detect_vb` на CPU в отдельном процессе; записать версии, хеши кода и весов, параметры, exit code, время и логи. MPS и время выполнения пока не проверены.
4. Сохранить native polygons и метки во внешнем артефакте; расширить контракт геометрии с явной версией и связать его с C001. Не добавлять неизвестные поля в strict schema 1.0. Получить привязку L1–S1 и пяти дисков; отсутствующие, дублирующиеся и нестандартные метки оставить неопределенными. S2 и переходную анатомию не переименовывать в стандартные уровни.
5. Проверить overlay на исходных срезах, координаты и полноту покрытия. Согласованная последовательность меток сама по себе не доказывает правильную абсолютную нумерацию. Для первого прогона не переносить ROI на T1/AxT2 и не запускать grading/fusion.

## Резервный путь TotalSpineSeg / SpineReport

TotalSpineSeg принимает NIfTI и сам ресемплирует объем; штатный полный CLI поддерживает `cpu/cuda`. Его `--loc` требует уже сегментированный и совмещенный localizer. Произвольная вторая MRI-серия не соответствует этому входу. [Inference API](https://github.com/neuropoly/totalspineseg/blob/4502d41bcb4a12e44f4be666411461ca81b02d89/totalspineseg/inference.py).

Релиз весов [r20260730](https://github.com/neuropoly/totalspineseg/releases/tag/r20260730) содержит два ZIP примерно по 1.05 ГБ. [GitHub API](https://api.github.com/repos/neuropoly/totalspineseg/releases/tags/r20260730) публикует SHA-256: step1 — `781f81e8c94feccd3cef17d819ba5c293b506e9b1f2b106fc5e31b9dd85887b0`, step2 — `494c640ab8cb25475bd0fc4338c0931d37079af9dd19403879335c3908c06b46`. Эти архивы в L001 не скачивались. Найденный локальный runtime имеет более старую версию пакета `20260730`; он не объявляется эквивалентным проверенному HEAD `20260807`.

В `step1_levels` диски L1–L2…L5–S обозначены `21…25`; в `step2_output` соответствующие маски имеют `92,93,94,95,100`, тела L1…L5 — `41…45`, крестец — `50`. Последнее не равно отдельному телу S1; карта также допускает L6/L7. [Карта landmarks](https://github.com/neuropoly/totalspineseg/blob/4502d41bcb4a12e44f4be666411461ca81b02d89/totalspineseg/resources/labels_maps/levels_maps.json), [карта сегментации](https://github.com/neuropoly/totalspineseg/blob/4502d41bcb4a12e44f4be666411461ca81b02d89/totalspineseg/resources/labels_maps/tss_map.json).

SpineReport может понадобиться позднее для морфометрии. В проверенном commit дополнительно найден дефект CLI: parser создает `test_tss`, а resolver обращается к `test_dir`. Обычный `-t /tmp/test` воспроизвел `AttributeError` при изолированном запуске этих двух функций; полный reporting pipeline не запускался. [Источник CLI](https://github.com/ivadomed/SpineReport/blob/b07ce0fe65fc913e3162822f2ee82202e0be8808/spinereport/spinereport.py).

L001 закрывает выбор и аудит API/артефактов. Готовность нейросетевого запуска, качество локализации и сегментация остаются отдельными проверками L002 и последующих задач.
