PYTHON ?= python3

.PHONY: test table4 paper clean

test:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m pytest -q tests/test_revision_losses.py

table4:
	mkdir -p reproduced/saferlhf_stage4_two_seed
	$(PYTHON) analysis/p12_stage4_seed42_seed43_20260718/build_two_seed_table.py \
		--seed42 artifacts/saferlhf_stage4_two_seed/inputs/seed42_ranked_validation_summary.json \
		--seed43 artifacts/saferlhf_stage4_two_seed/inputs/seed43_ranked_validation_summary.json \
		--prompt-manifest artifacts/saferlhf_stage4_two_seed/inputs/fresh_default_test_1000.jsonl \
		--output-dir reproduced/saferlhf_stage4_two_seed
	diff -u artifacts/saferlhf_stage4_two_seed/TABLE4_SEED42_43.md reproduced/saferlhf_stage4_two_seed/TABLE4_SEED42_43.md

paper:
	cd ronpo_aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd ronpo_aaai && bibtex main
	cd ronpo_aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd ronpo_aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -rf reproduced .pytest_cache
	rm -f ronpo_aaai/main.aux ronpo_aaai/main.bbl ronpo_aaai/main.blg ronpo_aaai/main.log ronpo_aaai/main.out
