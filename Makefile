PYTHON ?= python3

.PHONY: doctor status check smoke

doctor status check smoke:
	$(PYTHON) scripts/harness.py $@
