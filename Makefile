PYTHON ?= python3

.PHONY: doctor status check smoke contracts

doctor status check smoke:
	$(PYTHON) scripts/harness.py $@

contracts:
	$(PYTHON) -m mri_root_impact validate --study harness/fixtures/contracts/study.json \
		--result harness/fixtures/contracts/localization.json \
		--result harness/fixtures/contracts/stenosis-assessed.json \
		--result harness/fixtures/contracts/stenosis-indeterminate.json \
		--result harness/fixtures/contracts/stenosis-not-run.json
