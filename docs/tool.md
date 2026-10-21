# MRI Report Tool 0.1.0

Отсечка 21 октября 2026: подготовленный набор МРТ → локализация → оценки стенозов → анатомические правила → короткий отчет F005. Один CLI запускает существующие проверяемые модули; формат рассуждения сохранен.

Рабочая поставка — **Stage I без анализа грыж**: локализация, два вида стеноза слева/справа, анатомический fusion и отчет с изображениями. LDH остается отдельной задачей; неизвестная грыжа не считается отсутствующей. Неподтвержденные нумерация и совмещение сохраняются в выводах.

## Запуск

Из корня репозитория, Python 3.9+:

```sh
python3 scripts/run_mri.py \
  --study /path/to/prepared/study.json \
  --config /path/to/config/local-models.json \
  --output /path/to/runs/new-run
```

Вход — `study.json` по [контракту исследования](contracts.md) и указанные в нем исходные DICOM с SHA-256. Для текущих моделей нужны Sagittal T1, Sagittal T2/T2-like и Axial T2. Произвольная папка снимков без описания серий пока не принимается. Подготовка RSNA-набора — [control-study.md](control-study.md).

Один раз создайте внешний конфиг локальных моделей:

```json
{
  "schema_version": "1.0",
  "mode": "inference",
  "spinenet": {
    "source": "/path/to/SpineNet",
    "python": "/path/to/.venv-spinenet/bin/python"
  },
  "rsna": {
    "source": "/path/to/rsna/source",
    "weights": "/path/to/rsna/weights",
    "python": "/path/to/.venv-rsna/bin/python"
  },
  "options": {
    "threads": 2,
    "timeout": 600,
    "allow_missing_modality": false,
    "allow_unverified_registration": false
  }
}
```

Относительные пути считаются от папки конфига. Среды и веса подготавливаются по [SpineNet](localization-run.md) и [RSNA](rsna-assets.md); тулза их не скачивает. `timeout` — лимит каждой стадии в секундах. Оба разрешающих флага по умолчанию выключены; ограничения контрольного RSNA-набора требуют явного `true`. Это сохраняет неопределенность нумерации и совмещения.

## Выход

Успех: exit code 0 и JSON в stdout с абсолютными путями `report`, `markdown_report`, `tool_run`. Ход выполнения — в stderr. Ошибка стадии: exit code 2; последующие стадии не запускаются, причина и завершенные стадии сохраняются в `tool-run.json`. Ошибка входных параметров возвращается до создания output. Существующий output не перезаписывается; новый каталог должен находиться вне Git, исходных серий, конфигов, моделей и upstream runs.

| Путь внутри output | Содержание |
|---|---|
| `fusion/report.html` | Четыре раздела: диски, фораминальный стеноз, субартикулярный стеноз, Fusion |
| `fusion/report.md` | Тот же краткий разбор в Markdown |
| `fusion/fusion.json`, `fusion/report-evidence.json` | Формальные выводы, исходные факты и привязки картинок |
| `tool-run.json` | Версия тулзы, входы и их SHA, режим, стадии, привязка к проверенному fusion-run |
| `preflight.json`, `localization/`, `stenosis/`, `crosscheck/` | Проверка DICOM и результаты стадий полного запуска |

Отчет содержит пять дисковых уровней, левую/правую оценку обеих зон стеноза с иллюстрациями и строки «находки ⇒ вывод». LDH пока не подключен: наличие, размер и сторона грыжи не определяются. Анатомическая нумерация и регистрация остаются неподтвержденными; умеренный стеноз не становится подтвержденным воздействием на корешок. Это текущее работающее MRI-ядро, Stage I целиком еще не завершен.

## Повторная сборка и проверка

Для готовых стадий используйте другой конфиг с тем же CLI:

```json
{
  "schema_version": "1.0",
  "mode": "frozen",
  "spinenet_run": "/path/to/localization-run",
  "rsna_run": "/path/to/stenosis-run",
  "crosscheck_run": "/path/to/crosscheck-run"
}
```

Этот режим повторно проверяет входные артефакты и собирает отчет без MRI inference. Одного наличия файлов недостаточно: проверяются SHA, исходные DICOM, версии кода и связи результатов.

Добавьте `--verify-only` к исходной команде, чтобы проверить сохраненный запуск: config, study, код тулзы и полный fusion с изображениями. Повторного MRI inference и записи файлов нет. Изменение входов, кода или артефактов дает ошибку; старые результаты сохраняются.

## Подготовленный локальный пример

```sh
cd /Users/andrew/Projects/mri-root-impact-ai
MRI_DATA_ROOT=/Users/andrew/Datasets/mri-root-impact-ai/rsna-3139265320
MRI_RUN_DIR="$MRI_DATA_ROOT/tool/run-$(date +%Y%m%d-%H%M%S)"
python3 scripts/run_mri.py \
  --study "$MRI_DATA_ROOT/prepared/study.json" \
  --config "$MRI_DATA_ROOT/tool-config/inference.json" \
  --output "$MRI_RUN_DIR"
```

Выходной HTML — `$MRI_RUN_DIR/fusion/report.html`; абсолютный путь печатается в поле `report`. Новый запуск получает новую папку, старые результаты сохраняются.

Контрольный полный запуск уже сохранен в `tool/t001-001`. Проверить его без моделей и записи:

```sh
python3 scripts/run_mri.py \
  --study "$MRI_DATA_ROOT/prepared/study.json" \
  --config "$MRI_DATA_ROOT/tool-config/inference.json" \
  --output "$MRI_DATA_ROOT/tool/t001-001" --verify-only
```

Конфиг `tool-config/frozen.json` повторяет исходные L002/S003/L003 без моделей и сохраняет тот же формат F005; для него также нужен новый output.
