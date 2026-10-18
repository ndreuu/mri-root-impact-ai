# L002 — геометрия локализации 1.0

`mri_root_impact.spatial.build_localization(study, geometry, native_detections, model_provenance)` возвращает два JSON-объекта: неизмененный по схеме C001 `module_result` и отдельный `localization_spatial` версии `1.0`. `validate_spatial(study, result, artifact)` проверяет их совместно без ML-библиотек и чтения изображений. Реальные результаты и координаты хранятся вне Git.

## Источник координат

`geometry` описывает один нативный вход `H×W×S`. Он содержит `series_id`, `volume_shape`, скалярный изотропный `pixel_spacing_mm`, положительный `slice_center_spacing_mm`, `coordinate_system="DICOM_LPS_mm"` и массив `slices` по порядку входного объема. Каждый элемент содержит ровно:

- `slice_index` — индекс с нуля;
- `image_ref` — `series_id`, `image_id`, `frame_index=0` из manifest;
- `image_position_patient_mm` — исходные три значения IPP;
- `image_orientation_patient` — исходные шесть значений IOP;
- `pixel_spacing_mm` — исходные `[row_spacing, column_spacing]`;
- `slice_thickness_mm` — толщина среза, отдельно от расстояния между центрами.

Все одиночные кадры выбранной сагиттальной серии представлены ровно один раз. Проверяются конечные числа, положительные размеры, изотропный in-plane spacing, ортонормальность и постоянство ориентации, равномерность физического шага, монотонный порядок и отсутствие сдвига центров в плоскости. Эти проверки подтверждают внутреннюю согласованность метаданных. Соответствие IPP/IOP/размеров реальным DICOM и хешам подтверждает `sagittal.py`, читающий исходные файлы. Один только manifest не содержит этих геометрических величин.

Вход `native_detections` сохраняет поля SpineNet `predicted_label`, `slice_nos`, `polys`, `average_polygon`. Полигон имеет четыре конечные точки `[x,y]`, где `x` — колонка, `y` — строка исходного кадра. Для каждой вершины:

```text
LPS = IPP + x × PixelSpacing[1] × IOP[:3]
          + y × PixelSpacing[0] × IOP[3:]
```

Индексы срезов должны попадать во входной объем и не повторяться внутри одной детекции. Полигон за границей кадра сохраняется без обрезки с `in_bounds=false`; такой выход не становится подтвержденной локализацией. `average_polygon` сохраняется как native summary, ему не назначается исходный кадр и LPS-ROI. Voxel masks и дисковые ROI из crop не восстанавливаются.

## Внешний артефакт

Верхний уровень содержит ровно `schema_version`, `kind`, `study_id`, `study_digest`, `synthetic`, `result_id`, `model_provenance`, `geometry`, `detections`, `coverage`, `limitations`. `study_digest` вычисляется по C001; `result_id` детерминированно зависит от manifest, геометрии, всех native детекций и модели. `model_provenance` содержит `source_commit` и `weights_sha256`; последний — закрепленный идентификатор набора весов, способ получения которого фиксирует runner.

Каждая детекция содержит `detection_id`, `assessment_id`, исходный `native_label`, `native_average_polygon_xy` и `polygons`. Каждый полигон содержит `slice_index`, явный `image_ref`, исходный `polygon_xy`, вычисленный `polygon_lps_mm` и `in_bounds`. `assessment_id` связывает детекцию с C001 observation той же сырой метки и тех же исходных кадров.

`coverage.vertebrae` явно перечисляет L1, L2, L3, L4, L5, S1 с массивом `detection_ids`, статусом и причиной. `coverage.disc_levels` перечисляет пять уровней L1–L2…L5–S1 со статусом `not_run`, причиной и `adjacent_detection_ids`. Наличие соседних тел — только сведения для будущей привязки, не детекция диска и не его источник ROI. `unsupported_detection_ids` сохраняет ссылки на нестандартные метки, например T12 или S2. Они не переименовываются в стандартные уровни.

## Статусы и пределы проверки

Все native номера позвонков остаются `indeterminate`: абсолютная нумерация не подтверждена независимым анатомическим review. Сырые observations при этом сохраняются. Отсутствующий ожидаемый позвонок получает отдельную `indeterminate` assessment без observation; отсутствие детекции не означает отсутствия структуры или патологии. Для дублей сохраняются все детекции с причиной неоднозначности. Для неподдерживаемого номера C001 target имеет `level="unknown"`, а сырая метка остается в observation и геометрическом артефакте. Все пять дисковых assessments имеют `not_run` без выдуманных координат и observations.

`anatomical_review_status="not_reviewed"` и `full_expected_numbering=false` сохраняются даже при наличии шести ожидаемых меток. `validate_spatial` заново строит канонические C001 и spatial-выходы из сохраненных native наблюдений: независимое редактирование координат LPS, ссылок на кадры, связей assessments, статусов или покрытия отклоняется. Он не определяет анатомическую правильность модели, не выполняет регистрацию между сериями и не подтверждает точность на пациентах.
