# S002 — проверенный RSNA-комплект и CPU-окружение

19 октября 2026 года получены три обученных checkpoint из опубликованного MIT model dataset `akinosora/rsna2024-lsdc-3rd-models-pub`, **v2**. Исходники закреплены на `8a3cd3e615bf24c934dc5440e96ee52aa68c1386`. Файлы и окружение готовы для разработки S003; **МРТ этим комплектом еще не обрабатывались**.

| Роль | Опубликованный checkpoint | Размер | Результат CPU probe |
|---|---|---:|---|
| Стенозы NFN + SS, fold1 | `stage2/tkmn/side/v9/0203/exp0203_fold1_best_score.pth` | 41 158 224 B | strict load, logits `[1,2,1,3]` |
| Сагиттальные ориентиры, fold0 | `stage1/moyashii/v6/0002/exp0002_fold0_best_score.pth` | 165 772 062 B | strict load, heatmaps `[1,5,190,190]` |
| Аксиальные ориентиры, fold0 | `stage1/moyashii/v5/0003/exp0003_fold0_best_score.pth` | 72 121 783 B | strict load, heatmap `[1,1,95,95]` |

Итого **279 052 069 байт**; полный 22-ГБ ансамбль, competition dataset и дополнительный training dataset не скачивались. Точные SHA-256, URL, размеры, конфигурации и формы — в [инвентаре](../harness/upstreams/rsna-assets.json). Первичные источники: [MIT код](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub/tree/8a3cd3e615bf24c934dc5440e96ee52aa68c1386), [опубликованные веса](https://www.kaggle.com/datasets/akinosora/rsna2024-lsdc-3rd-models-pub). MIT — заявленные условия кода и набора весов; это не вывод о клинической пригодности или всех upstream data/backbone terms.

## Что именно проверено

`mri_root_impact.rsna_assets` сверяет Git revision, реальные байты всех Python-файлов с закрепленными Git blobs и выбранные config/license с SHA inventory. Git index flags не скрывают изменения. Дополнительные Python-файлы, недостающие файлы, symlink/traversal, неверные размер или SHA checkpoint отклоняются до загрузки модели. Verification не импортирует Torch или upstream.

Отдельный `mri_root_impact.rsna_probe` проверяет окружение и читает из `src/predict.py` только пять закрепленных AST class definitions: GeM, TransformerBlock, AttentionNet, KeypointNeck и KeypointNetTest. Весь `predict.py` не импортируется: его top-level CUDA, downloads и обработка исследований не выполняются; upstream `.pyc` не используются. Backbone создается с `pretrained=False`; каждый медицинский state_dict загружается через `torch.load(weights_only=True,map_location='cpu')` и `load_state_dict(strict=True)`. У classifier удален только присутствующий training-only `target_loss.ce_loss.weight`, как в авторском inference; остальные параметры совпали строго.

Далее все три модели выполняют forward на синтетических случайных тензорах нужных размеров. Проверены конечность и точная форма выхода. Нет чтения снимков или CSV разметки; отчет содержит `input_kind=synthetic_random_tensors`, `patient_images_read=false`, `synthetic_forward_performed=true`, `mri_inference_performed=false`, `root_impacts=null`. Это проверка загрузки и вычисления, не демонстрация оценки стенозов на пациенте.

Первый probe на Apple M5 с двумя CPU-потоками занял 12.38 с целиком, включая версии, source/weight verification, import, создание моделей и загрузку. Сами synthetic forward: classifier 0.24 с, sagittal 0.33 с, axial 0.14 с. Эти времена не являются длительностью обработки исследования. Постоянный результат повторной проверки — `verification/s002-001.json` во внешней папке ниже.

## Окружение и воспроизведение

Локальные исходники/веса находятся в `/Users/andrew/Datasets/mri-root-impact-ai/models/rsna2024-moyasii-v2/{source,weights}`. Рядом сохранены metadata MIT-набора, provenance загрузки и отчеты. В Git хранятся только программа, inventory и синтетические тесты; `.venv-rsna` игнорируется.

Используется отдельный Python 3.12.14, Torch 2.5.1 / torchvision 0.20.1, timm 0.9.16, NumPy 1.26.4. Полный установленный lock — [requirements-rsna.txt](../requirements-rsna.txt), `pip check` проходит. Torch отличается от CUDA-окружения авторов 2.1.2 ради поддержки локального Python/CPU; совместимость трех конкретных checkpoint проверена. Рабочие `.venv-spinenet` и `.venv-data` не изменялись.

Из корня проекта, с доступным Python 3.12:

```sh
python3.12 -m venv .venv-rsna
.venv-rsna/bin/python -m pip install -r requirements-rsna.txt
.venv-rsna/bin/python -m pip check

RSNA_MODEL_ROOT=/Users/andrew/Datasets/mri-root-impact-ai/models/rsna2024-moyasii-v2
git clone https://github.com/Moyasii/Kaggle-2024-RSNA-Pub.git "$RSNA_MODEL_ROOT/source"
git -C "$RSNA_MODEL_ROOT/source" checkout --detach 8a3cd3e615bf24c934dc5440e96ee52aa68c1386
python3 scripts/fetch_rsna_assets.py --output "$RSNA_MODEL_ROOT/weights"
python3 scripts/fetch_rsna_assets.py --output "$RSNA_MODEL_ROOT/weights" --verify-only
.venv-rsna/bin/python -m mri_root_impact.rsna_probe \
  --source "$RSNA_MODEL_ROOT/source" --weights "$RSNA_MODEL_ROOT/weights" \
  --output "$RSNA_MODEL_ROOT/verification/s002-repeat-001.json"
```

Команды clone/setup предназначены для нового каталога/окружения. Для уже подготовленной машины достаточно `--verify-only` и probe с новым output. Загрузчик использует только три versioned URL, проверяет границы HTTP-ответа, размер и заранее закрепленный SHA, сохраняет файл атомарно; существующий файл с другим содержимым не перезаписывает. `--verify-only` работает без сети. Probe требует новый внешний output и не перезаписывает прошлый отчет.

## Что остается S003

Нужно воспроизвести автоматическую локализацию, DICOM intensity conversion, dataset-v9 crops, выбор и порядок срезов, resampling и отражения стороны. Возможность создания heatmaps не подтверждает правильность ROI на наших снимках. Все три последовательности обязательны для выбранного multi-view classifier: 15 SagT1, 15 SagT2-like, 10 AxT2 crop-срезов на уровень/сторону; это не независимые T1-only и AxT2-only модели.

Сокращенный комплект не воспроизводит полный призовой ансамбль. Пересечение classifier fold1 и keypoint fold0 с нашим training-case отдельно не установлено. Нумерация RSNA/SpineNet, геометрия и сторона остаются проверяемыми ограничениями; missing/invalid ROI не становится Normal/Mild. S003 должен сохранить native результаты, source-frame geometry и C001 для двадцати боковых targets с неопределенностью. Влияние на корешок, fusion и клиническая оценка этим probe не выполнялись.
