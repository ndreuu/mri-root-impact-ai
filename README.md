# MRI Root Impact AI

Отдельный локальный репозиторий Stage I: оценка признаков потенциального воздействия на нервные корешки по МРТ поясничного отдела. Границы проекта зафиксированы в [scope](docs/stage-1-scope.md).

Сейчас реализованы [контракты исследований и результатов модулей](docs/contracts.md), их JSON-валидатор, подготовка и проверка [контрольного RSNA-исследования](docs/control-study.md), рабочий harness и [первый реальный запуск локализатора SpineNet](docs/localization-run.md). Анализ грыж, стенозов и fusion пока не подключены.

L001/L002 завершены: [SpineNet выбран для исследовательской локализации](docs/localizer-audit.md), запущен на CPU; сохранены polygons и [привязка к исходным DICOM](docs/spatial-contract.md). Независимое анатомическое подтверждение номеров и дисковые ROI еще не выполнены.

D001 завершен: [аудит LDH-DL-MODEL](docs/ldh-audit.md) выявил отсутствие подтвержденных обученных lumbar-весов и измерительного конвейера в проверенном публичном комплекте. Для этой ветки заведена D002; следующий независимый шаг — S001, аудит моделей RSNA.

## Быстрый запуск

Нужны Python 3.9+ и Git. Установка пакетов и доступ к сети не требуются.

```sh
python3 scripts/harness.py doctor
python3 scripts/harness.py status
python3 scripts/harness.py check
python3 scripts/harness.py smoke
make contracts
```

При наличии Make доступны те же команды: `make doctor`, `make status`, `make check`, `make smoke`.

`doctor` проверяет окружение; `status` показывает очередь; `check` запускает проверки harness; `smoke` проверяет синтетический manifest и запись/чтение артефакта. Успешный smoke подтверждает работу harness. В артефакте явно указаны `kind=harness_smoke`, `synthetic=true`, `inference_performed=false`, `root_impacts=null`; все четыре стадии имеют статус `not_configured`.

`make contracts` проверяет синтетический bundle с локализацией, оценкой в исходной шкале, неопределенным результатом и пропущенным запуском. Эта проверка входит в `check`. Для своих файлов: `python3 -m mri_root_impact validate --study /external/study.json --result /external/module-result.json`. Изображения не открываются, входные файлы не изменяются.

Реальные входы проверяет отдельная команда `python -m mri_root_impact preflight --study /external/study.json --output /external/preflight.json`: она читает DICOM, сверяет SHA-256 и декодирует кадры. Для нее нужны дополнительные зависимости из `requirements-data.txt`; базовые проверки harness по-прежнему работают без них. Команды получения, подготовки и повторного прогона контрольного примера находятся в [инструкции](docs/control-study.md).

## Работа по шагам

1. Выбрать готовую задачу из [очереди](harness/tasks.json) и выполнить один проверяемый шаг.
2. Запустить подходящие проверки и записать фактический результат в [прогресс](docs/progress.md).
3. Синхронно обновить статус задачи и её зависимости.

Карта репозитория и правила для агентов находятся в [AGENTS.md](AGENTS.md), устройство Stage I — в [архитектуре](docs/architecture.md). Данные пациентов и веса моделей хранятся вне Git; в [fixtures](harness/fixtures/) допустимы только синтетические примеры.
