# D002 — черновик запроса воспроизводимых артефактов

Ниже подготовлен текст для возможного обращения к авторам. Он не отправлялся. Канал и адресата перед отправкой выбирает пользователь; контакты можно сверить на странице [публикации](https://link.springer.com/article/10.1007/s11547-025-01996-y).

Subject: Research reproducibility artifacts for LDH-DL-MODEL

Dear authors,

We are evaluating your LDH-DL-MODEL method for a non-commercial lumbar MRI research project. We inspected the public repository at commit `b2b9ca49aac8b4b552f0dc0ad0342e3352306ab7`. Its pretrained links appear to point to generic Ultralytics COCO models.

Could you please clarify whether the following research artifacts are available, with their permitted uses?

1. LDH-trained detection, segmentation and nine-keypoint checkpoints, including versions, checksums and dependency versions.
2. The anatomical meanings of each detection/segmentation class ID and the ordered nine-keypoint definitions, including visibility conventions.
3. DICOM-to-model preprocessing: intensity processing, image orientation, resizing/cropping, physical spacing preservation and slice selection.
4. Postprocessing code for measurements, diameter index, regional assignment and severity, including threshold boundaries and multi-region cases.
5. A permitted reproducibility example with expected intermediate outputs and final measurements, or instructions for an appropriately licensed public example.
6. Separate usage and redistribution conditions for source code, trained parameters and derived outputs.

We would preserve your method attribution, source geometry and uncertainty, and distinguish any reproduction from clinical validation.

Thank you.
