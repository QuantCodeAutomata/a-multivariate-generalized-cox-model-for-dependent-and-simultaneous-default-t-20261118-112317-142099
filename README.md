# A Multivariate Generalized Cox Model for Dependent and Simultaneous Default Times

Reproduction of the synthetic constructions of Djibril Gueye and Alejandra
Quintos, *A Multivariate Generalized Cox Model for Dependent and Simultaneous
Default Times*, arXiv:2508.05022v2 [math.PR].

The paper builds default times `τ^i = inf{t : K^i_t ≥ Θ^i}` from increasing,
possibly jumping, F-adapted drivers `K^i` crossed against independent unit
exponential thresholds `Θ^i`, and derives the conditional and unconditional
joint survival law of `(τ^1,...,τ^n)` for several classes of drivers: Lévy
subordinators, compound Poisson processes, Liu's (2020) frailty/time-
deformation model, shot-noise processes with age-dependent memory kernels, a
linear multivariate factor model, and an independent-block factorization of
continuous deterioration and shock-driven default. This repository
implements, as nine self-contained experiments, the explicit formulas and
qualitative claims attached to each construction (Marshall-Olkin shock rates,
F_t-conditional survival, simultaneous-default probabilities, failure or
success of Lemma 3.4, predictable default atoms) and checks each formula
against an independent Monte Carlo simulation of the underlying model.

All nine experiments are fully synthetic: the paper is a mathematical
construction, not an empirical study, so no market data or other external
dataset is used anywhere in this repository.

## Repository layout

```
src/    model building blocks (Levy subordinators, shock rates, Liu's model,
        shot-noise drivers, factor model, CIR block, backward recursion,
        default-time simulation) and src/config.py (all experiment parameters)
exp/    one script per experiment, exp1.py .. exp9.py
data/   data/generate_synthetic.py: the shared seeded RNG used by every
        experiment (see its docstring for why no other "data" is needed)
tests/  pytest unit tests of every src/ module
results/ results/RESULTS.md and the JSON file each experiment writes
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or environment variables are required: every experiment is a
closed-form / Monte Carlo computation with no external data dependency.

## Running the tests

```bash
python -m pytest -m "not market_data"
```

(there are no `market_data`-marked tests in this repository, since no market
data is used; the flag is included for consistency with the general test
convention.)

## Running the experiments

Each experiment writes its results JSON into `results/` and prints a short
summary; see `results/RESULTS.md` for the full discussion of each file.

```bash
python -m exp.exp1   # Bivariate Levy subordinator drivers, Marshall-Olkin recovery
python -m exp.exp2   # Compound Poisson drivers, synchronized common jumps
python -m exp.exp3   # Liu's (2020) competing-risks model with frailty and time deformation
python -m exp.exp4   # Shot-noise drivers with age-dependent memory kernels
python -m exp.exp5   # n-name linear factor model, Marshall-Olkin shock rates
python -m exp.exp6   # Example 5.8: predictable default atoms, non-monotone Moebius coordinate
python -m exp.exp7   # General backward recursion (Theorem 5.1) for n-name joint survival
python -m exp.exp8   # Independent-block factorization (Remark 5.9)
python -m exp.exp9   # Multivariate Liu model for n names (Section 5.1.2)
```

Or run the whole suite with:

```bash
bash reproduce.sh
```

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for every experiment's closed-
form values, Monte Carlo checks, verdicts, and the "Deviations from the
paper" / "Not reproduced" sections.
