# MRI Root Impact AI

Отдельный локальный репозиторий Stage I: оценка признаков потенциального воздействия на нервные корешки по МРТ поясничного отдела. Границы проекта зафиксированы в [scope](docs/stage-1-scope.md).

Сейчас подготовлен только рабочий harness: очередь задач, проверка окружения и репозитория, синтетический smoke с сохранением артефакта. Модели не подключены; анализа МРТ и клинической диагностики пока нет.

## Быстрый запуск

Нужны Python 3.9+ и Git. Установка пакетов и доступ к сети не требуются.

```sh
python3 scripts/harness.py doctor
python3 scripts/harness.py status
python3 scripts/harness.py check
python3 scripts/harness.py smoke
```

При наличии Make доступны те же команды: `make doctor`, `make status`, `make check`, `make smoke`.

`doctor` проверяет окружение; `status` показывает очередь; `check` запускает проверки harness; `smoke` проверяет синтетический manifest и запись/чтение артефакта. Успешный smoke подтверждает работу harness. В артефакте явно указаны `kind=harness_smoke`, `synthetic=true`, `inference_performed=false`, `root_impacts=null`; все четыре стадии имеют статус `not_configured`.

## Работа по шагам

1. Выбрать готовую задачу из [очереди](harness/tasks.json) и выполнить один проверяемый шаг.
2. Запустить подходящие проверки и записать фактический результат в [прогресс](docs/progress.md).
3. Синхронно обновить статус задачи и её зависимости.

Карта репозитория и правила для агентов находятся в [AGENTS.md](AGENTS.md), устройство Stage I — в [архитектуре](docs/architecture.md). Данные пациентов и веса моделей хранятся вне Git; в [fixtures](harness/fixtures/) допустимы только синтетические примеры.
