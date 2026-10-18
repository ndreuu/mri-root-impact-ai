# MRI Root Impact AI

Отдельный локальный репозиторий Stage I: оценка признаков потенциального воздействия на нервные корешки по МРТ поясничного отдела. Границы проекта зафиксированы в [scope](docs/stage-1-scope.md).

Сейчас реализованы [контракты исследований и результатов модулей](docs/contracts.md), их JSON-валидатор и рабочий harness. Модели не подключены; анализа МРТ и клинической диагностики пока нет.

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

## Работа по шагам

1. Выбрать готовую задачу из [очереди](harness/tasks.json) и выполнить один проверяемый шаг.
2. Запустить подходящие проверки и записать фактический результат в [прогресс](docs/progress.md).
3. Синхронно обновить статус задачи и её зависимости.

Карта репозитория и правила для агентов находятся в [AGENTS.md](AGENTS.md), устройство Stage I — в [архитектуре](docs/architecture.md). Данные пациентов и веса моделей хранятся вне Git; в [fixtures](harness/fixtures/) допустимы только синтетические примеры.
