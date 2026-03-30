# When Accuracy Hides Path Dependence: Criterion Fidelity in LLM Judges

Ali Uyar  
Independent Researcher

This repository contains the manuscript source, experiment code, and reproducibility configs for the paper *When Accuracy Hides Path Dependence: Criterion Fidelity in LLM Judges*.

## Abstract

Large language model judges are often evaluated with ordinary accuracy or agreement, which can hide whether the model is actually following the provided criterion. This paper studies that missing property as *criterion fidelity*: a valid judge should change when criterion meaning changes and remain stable when wording changes but meaning does not. We operationalize this with criterion families that group a base criterion, meaning-preserving paraphrases, and a counterfactual criterion that should flip the gold winner. Across QA-Key and BioRubric, ordinary quality and criterion fidelity dissociate. Under standard prompting, QA-Key gaps are small but real, while BioRubric gaps are large; strict criterion-emphasis prompting is not a reliable fix and can worsen path dependence. The results position criterion fidelity as a distinct validity axis for LLM judges.

## Main Findings

- Criterion fidelity is not equivalent to ordinary base-case accuracy.
- Controlled criterion families reveal failures that flattened row-level evaluation hides.
- BioRubric is the decisive stress test: both candidates are factually true, so the rubric must do the work.
- Strict criterion-emphasis prompting is inconsistent as an intervention and can increase order-sensitive behavior.
- The strongest failures are often mediated by path dependence rather than clean semantic refusal alone.

## Manuscript

- LaTeX source: `paper/main.tex`
- Build output: `paper/build/main.pdf`
- Reader-facing PDF copy: `paper/When_Accuracy_Hides_Path_Dependence_Criterion_Fidelity_in_LLM_Judges_Ali_Uyar.pdf`
- Preferred build command: `powershell -ExecutionPolicy Bypass -File paper/build_paper.ps1`

Build the paper with:

```powershell
powershell -ExecutionPolicy Bypass -File paper/build_paper.ps1
```

This command rebuilds `paper/build/main.pdf` and refreshes the named reader-facing PDF copy automatically.

## Repository Layout

- `paper/` - manuscript source, style files, and figure assets
- `src/` - data construction, inference, and evaluation code
- `scripts/` - pipeline entrypoints for building data, running experiments, and generating paper assets
- `configs/` - YAML configs for data, prompts, models, experiments, and analysis
- `tests/` - regression tests for the public code path
- `data/` - local output location for generated datasets and audits
- `results/` - local output location for run artifacts, metrics, and figures

## Environment

- Tested with Python 3.12.
- Install dependencies with `python -m pip install -r requirements.txt`.
- Conda users can create the environment with `conda env create -f environment.yml`.
- CPU-only environments are sufficient for schema validation, linting, tests, and paper build steps.
- Full inference runs are intended for a CUDA-enabled machine with Hugging Face model access.
- The default model configs in `configs/model/` set `load_in_4bit: true`; that path expects `bitsandbytes` and is most reliable on Linux.
- Access to `meta-llama/Llama-3.1-8B-Instruct` may require prior approval and Hugging Face authentication.
- If 4-bit loading is unavailable in your environment, set `load_in_4bit` to `false` in the model config before running inference.

## Reproduction

The public pipeline is organized around the numbered scripts in `scripts/`.

```powershell
python -m pip install -r requirements.txt

python scripts/00_validate_family_schema.py --toy

python scripts/fetch_qakey_wikidata_source.py --config configs/data/qakey_wikidata_fetch.yaml
python scripts/fetch_biorubric_wikidata_source.py --config configs/data/biorubric_wikidata_fetch.yaml

python scripts/01_build_canonical_qatable.py --config configs/data/qakey_source.yaml
python scripts/02_build_canonical_facttable.py --config configs/data/biorubric_source.yaml
python scripts/03_build_qakey.py --config configs/data/qakey_pilot.yaml
python scripts/05_build_biorubric.py --config configs/data/biorubric_pilot.yaml
python scripts/07_build_full_datasets.py --config configs/data/full_build.yaml

python scripts/08_run_full_standard.py --config configs/exp/full_standard.yaml
python scripts/09_run_full_strict.py --config configs/exp/full_strict.yaml
python scripts/10_make_figures_tables.py --config configs/analysis/main.yaml
```

Validation commands:

```powershell
ruff check .
ruff format --check .
python -m pytest -q
```

## Citation

Citation metadata for GitHub and reference managers is provided in `CITATION.cff`.

## Notes on Public Artifacts

- Generated datasets, raw model outputs, metrics dumps, checkpoints, and local export bundles are not tracked in the public repository.
- Running the build and evaluation scripts will populate `data/` and `results/` locally.
- The tracked tree is intentionally limited to the manuscript, code, configs, and tests needed for a clean research release.

## License

This repository is licensed under the Apache License 2.0. See `LICENSE`.
