# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A reproduction of the synthetic constructions in Djibril Gueye and Alejandra
Quintos, *A Multivariate Generalized Cox Model for Dependent and Simultaneous
Default Times* (arXiv:2508.05022v2 [math.PR]): nine experiments covering
Lévy-subordinator drivers (Section 4.1.1), compound Poisson drivers with
synchronized jumps (Section 4.1.1), Liu's (2020) frailty/time-deformation
model (Section 4.1.2), shot-noise drivers with memory kernels (Section
4.1.3), the n-name linear factor model (Section 5.1.1), the predictable
default-atom Example 5.8, the general backward recursion of Theorem 5.1, the
independent-block factorization of Remark 5.9, and the n-name Liu model of
Section 5.1.2. See `results/RESULTS.md` for the full discussion and numeric
comparisons.

The paper is a purely mathematical construction with no dataset or empirical
application, so every experiment is fully synthetic: no market data of any
kind is used anywhere in this repository.

## Layout

```
src/    model building blocks: levy_subordinator.py, shock_rates.py,
        liu_model.py, shot_noise.py, factor_model.py, cir_block.py,
        backward_recursion.py, default_times.py, config.py (all parameters,
        grouped by experiment, each with a paper-section or "not specified
        in the paper" comment)
exp/    exp1.py .. exp9.py, one script per experiment; each calls only the
        public functions of src/ and writes its result JSON to results/
data/   generate_synthetic.py: the shared seeded RNG (src.config.RNG_SEED)
        used by every experiment; there is no market-data loader because the
        paper requires none
tests/  pytest unit tests of every src/ module (small, synthetic inputs)
results/ RESULTS.md plus the JSON file each experiment writes
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest -m "not market_data"
```

There are no `market_data`-marked tests in this repository (no market data is
used); the marker is kept for consistency with the general convention.

Experiments (each is also runnable individually as `python -m exp.expN`):

```bash
bash reproduce.sh
```

## Conventions

- All experiment parameters live only in `src/config.py`, as module-level
  `UPPER_CASE` variables grouped by experiment under a comment header; no
  classes or parameter dictionaries. Each variable's comment states its
  meaning, unit, and source (a paper section, or "not specified in the
  paper" when the paper leaves it open).
- Library modules are flat files directly under `src/`; experiment scripts
  live only in `exp/`; all pytest files live only in `tests/`.
- Experiments are run as `python -m exp.<name>` and tests as
  `python -m pytest` from the repository root; code never edits `sys.path`.
- There is no `data/data_loader.py` and no API keys because no experiment
  uses market data; `data/generate_synthetic.py` only seeds the shared RNG
  used by the model simulators in `src/`.
- Every call from `exp/` to `src/` uses the exact function signatures and
  return-value keys defined in `src/`; do not add keyword arguments that a
  `src/` function does not accept.
- After rerunning an experiment, its command overwrites the corresponding
  file in `results/`. The numbers and verdicts in `results/RESULTS.md` are
  not regenerated automatically: check them against the new result files and
  update them by hand.
