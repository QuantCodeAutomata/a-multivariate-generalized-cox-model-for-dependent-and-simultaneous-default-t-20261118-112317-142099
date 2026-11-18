"""Module-level parameters for every experiment, grouped by experiment.

All experiments in this repository are fully synthetic (Section 4 and
Section 5 of the paper give explicit mathematical constructions, not an
empirical application), so no market-data parameters are needed. Every
parameter below is either read off a paper equation/example or, where the
paper leaves a numeric choice open (e.g. a specific Levy measure or a
specific frailty distribution), set to a simple illustrative value; such
choices are flagged "not specified in the paper".
"""
import numpy as np

RNG_SEED = 20261118  # fixes reproducibility of all Monte Carlo checks below

# ---------------------------------------------------------------------------
# exp_1: Bivariate Levy subordinator drivers, Marshall-Olkin law recovery
# (Section 4.1.1, eq. 15-16). Levy measure not specified in the paper: a
# bivariate compound Poisson mixture of idiosyncratic and common exponential
# jumps is used, satisfying the paper's integrability condition trivially
# (finite total intensity).
# ---------------------------------------------------------------------------
EXP1_Z1 = 1.3          # scaling constant z1 (not specified in the paper)
EXP1_Z2 = 0.9           # scaling constant z2 (not specified in the paper)
EXP1_LAMBDA_TOTAL = 2.0  # total Poisson jump intensity of the mixture (not specified)
EXP1_P_IDIO1 = 0.3      # P(jump is idiosyncratic to name 1) (not specified)
EXP1_P_IDIO2 = 0.3      # P(jump is idiosyncratic to name 2) (not specified)
EXP1_P_COMMON = 0.4     # P(jump is common to both names), ensures nu((0,inf)^2)>0
EXP1_MARK_RATE_IDIO1 = 1.0   # rate of Exp mark on the idiosyncratic-1 jumps
EXP1_MARK_RATE_IDIO2 = 1.2   # rate of Exp mark on the idiosyncratic-2 jumps
EXP1_MARK_RATE_COMMON1 = 0.8  # rate of Exp mark on name 1 at a common jump
EXP1_MARK_RATE_COMMON2 = 1.5  # rate of Exp mark on name 2 at a common jump
EXP1_T = 5.0            # time horizon T (not specified in the paper)
EXP1_T1 = 1.6           # maturity t1 used for the conditional/unconditional checks
EXP1_T2 = 2.4           # maturity t2 used for the conditional/unconditional checks
EXP1_N_PATHS = 200_000  # Monte Carlo paths for the simultaneous-default check
EXP1_FT_CONDITIONING_TIME = 0.5  # conditioning time t <= min(t1, t2) for eq. (16)

# ---------------------------------------------------------------------------
# exp_2: Compound Poisson drivers with synchronized common jumps only
# (Section 4.1.1, "Compound Poisson drivers"). Mark distributions not
# specified in the paper: Exponential marks are used (closed-form Laplace
# transform phi_i(z) = rate_i / (rate_i + z)).
# ---------------------------------------------------------------------------
EXP2_LAMBDA = 3.0        # Poisson intensity lambda of the common clock (not specified)
EXP2_MARK_RATE1 = 0.7    # rate of Exp mark xi_i (name 1) (not specified)
EXP2_MARK_RATE2 = 2.0    # rate of Exp mark zeta_i (name 2) (not specified)
EXP2_Z1 = 1.5            # scaling constant z1 (not specified)
EXP2_Z2 = 0.6            # scaling constant z2 (not specified)
EXP2_T = 4.0             # time horizon
EXP2_N_JUMPS_FOR_CLASSIFICATION = 500_000  # common-jump events simulated

# ---------------------------------------------------------------------------
# exp_3: Liu's (2020) competing-risks model with covariates, frailty and
# time deformation (Section 4.1.2). Frailty law not specified in the paper:
# a Gamma(alpha, alpha) frailty is used, with E[theta] = 1 and known Laplace
# transform L_theta(u) = (1 + u/alpha)^{-alpha}.
# ---------------------------------------------------------------------------
EXP3_PHI1_BETA = np.array([0.8, -0.3])  # covariate coefficients for phi_1(X)=exp(beta^T X)
EXP3_PHI2_BETA = np.array([0.5, 0.4])   # covariate coefficients for phi_2(X)
EXP3_X_MEAN = np.array([0.0, 0.0])      # covariate mean (not specified)
EXP3_X_STD = np.array([0.5, 0.5])       # covariate std (not specified)
EXP3_FRAILTY_ALPHA = 4.0        # Gamma(alpha, alpha) shape=rate, so E[theta]=1
EXP3_LEVY_LAMBDA_TOTAL = 2.0     # intensity of the underlying bivariate subordinator
EXP3_P_IDIO1 = 0.3
EXP3_P_IDIO2 = 0.3
EXP3_P_COMMON = 0.4
EXP3_MARK_RATE_IDIO1 = 1.0
EXP3_MARK_RATE_IDIO2 = 1.2
EXP3_MARK_RATE_COMMON1 = 0.8
EXP3_MARK_RATE_COMMON2 = 1.5
EXP3_DELTA0_ALPHA = 1.3   # Weibull-type continuous deformation delta0(t)=t^alpha
EXP3_JUMP_DATE = 1.0      # predictable jump date t0 of the discontinuous delta0
EXP3_JUMP_SIZE = 0.8      # size of the jump of delta0 at t0
EXP3_T1 = 1.2
EXP3_T2 = 2.1
EXP3_N_PATHS = 200_000    # Monte Carlo paths

# ---------------------------------------------------------------------------
# exp_4: Shot-noise drivers with age-dependent memory kernels (Section
# 4.1.3). Kernel shapes, intensity and mark law not specified in the paper;
# a nonhomogeneous Poisson arrival process with multiplicative kernels
# G^j(t,x) = x * b^j(t) is used, as suggested in the paper's own
# "Nonhomogeneous Poisson arrivals" paragraph.
# ---------------------------------------------------------------------------
EXP4_LAMBDA_N_RATE = 1.5     # constant part of the nonhomogeneous intensity lambda^N(s) (not specified)
EXP4_MARK_RATE = 1.0         # rate of the Exp(rate) mark distribution rho
EXP4_B1_IMMEDIATE = 0.6      # b^1(0): immediate impact coefficient for name 1
EXP4_B1_SLOPE = 0.5          # slope of b^1(t) = b1_immediate + b1_slope * t (memory)
EXP4_B2_IMMEDIATE = 0.4      # b^2(0): immediate impact coefficient for name 2
EXP4_B2_SLOPE = 0.3          # slope of b^2(t)
EXP4_T1 = 1.5
EXP4_T2 = 2.3
EXP4_N_PATHS = 150_000       # Monte Carlo paths (shot-noise simulation is costlier)
EXP4_T_GRID_POINTS = 400     # quadrature nodes for the time integrals in (23),(27),(28)

# ---------------------------------------------------------------------------
# exp_5: Multivariate n-name Levy subordinator, linear factor structure
# (Section 5.1.1). Number of names, factors and loadings not specified.
# ---------------------------------------------------------------------------
EXP5_N_NAMES = 5          # n: moderate so that 2^n - 1 = 31 subsets are enumerable
EXP5_N_FACTORS = 3        # m factors (not specified)
EXP5_LOADINGS = np.array([  # a_{ik} >= 0, rows=names, cols=factors (not specified)
    [1.0, 0.0, 0.3],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.4],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.5],
])
EXP5_FACTOR_INTENSITIES = np.array([1.2, 0.9, 0.7])  # lambda_k (not specified)
EXP5_FACTOR_MARK_RATES = np.array([1.0, 1.3, 0.8])   # rate of Exp mark per factor
EXP5_T = 3.0
EXP5_MATURITIES = np.array([1.1, 1.4, 1.0, 1.7, 1.3])  # t_i used in the joint survival check
EXP5_N_PATHS = 150_000

# ---------------------------------------------------------------------------
# exp_6: Example 5.8 (predictable default atoms, non-monotone Moebius
# coordinates). Model is fully specified by the paper: Y ~ Bernoulli(1/2),
# jump date t0 = 1.
# ---------------------------------------------------------------------------
EXP6_JUMP_DATE = 1.0
EXP6_N_PATHS = 300_000

# ---------------------------------------------------------------------------
# exp_7: General backward recursion (Theorem 5.1) for n names. Case 1 uses
# the Levy subordinator model of exp_5 (Lemma 3.4 holds). Case 2 uses a
# Liu-type model with a random Gamma frailty for n names (Lemma 3.4 fails).
# ---------------------------------------------------------------------------
EXP7_N_NAMES = 3
EXP7_MATURITIES = np.array([1.2, 0.9, 1.6])
EXP7_T_COND = 0.4            # conditioning time t <= min(maturities)
EXP7_LEVY_LOADINGS = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # (not specified)
EXP7_LEVY_FACTOR_INTENSITIES = np.array([1.0, 1.4])
EXP7_LEVY_FACTOR_MARK_RATES = np.array([1.1, 0.9])
EXP7_LIU_PHI_RATES = np.array([0.8, 1.1, 0.6])   # phi_i(X) covariate effects (not specified)
EXP7_LIU_FRAILTY_ALPHA = 3.0
EXP7_LIU_DELTA0_ALPHA = 1.0   # identity time deformation delta0(t) = t
EXP7_LIU_LEVY_LAMBDA_TOTAL = 1.5
EXP7_LIU_P_COMMON = 1.0       # fully synchronized jumps across all n names
EXP7_LIU_MARK_RATES = np.array([1.0, 1.2, 0.9])
EXP7_N_QUADRATURE_NODES = 4000  # Gauss-Laguerre-type quadrature nodes to integrate out the frailty
EXP7_N_PATHS = 150_000

# ---------------------------------------------------------------------------
# exp_8: Independent-block factorization (Remark 5.9): X^j integrated CIR
# intensities, K-tilde^j linear-factor Levy subordinator (as in exp_5).
# CIR parameters and loadings not specified in the paper.
# ---------------------------------------------------------------------------
EXP8_N_NAMES = 3
EXP8_CIR_KAPPA = np.array([1.0, 1.2, 0.9])   # mean-reversion speed per name
EXP8_CIR_THETA = np.array([0.02, 0.03, 0.025])  # long-run mean intensity
EXP8_CIR_SIGMA = np.array([0.1, 0.12, 0.09])   # volatility (Feller condition checked in code)
EXP8_CIR_LAMBDA0 = np.array([0.02, 0.03, 0.025])  # initial intensity
EXP8_SHOCK_LOADINGS = np.array([[1.0, 0.0], [1.0, 0.4], [0.0, 1.0]])  # a_{ik} (not specified)
EXP8_SHOCK_FACTOR_INTENSITIES = np.array([0.8, 0.6])
EXP8_SHOCK_FACTOR_MARK_RATES = np.array([1.0, 1.1])
EXP8_MATURITIES = np.array([1.5, 2.0, 1.8])
EXP8_N_PATHS = 100_000
EXP8_N_CIR_STEPS = 500        # Euler/exact steps per unit time for the CIR integrated-intensity simulation

# ---------------------------------------------------------------------------
# exp_9: Multivariate Liu model for n names with frailty and time
# deformation (Section 5.1.2). Reuses exp_5's loadings (reinterpreted as the
# covariate-driven aggregate psi built from the same bivariate-type mixture
# subordinator per factor) with a Gamma frailty.
# ---------------------------------------------------------------------------
EXP9_N_NAMES = 4
EXP9_PHI_RATES = np.array([0.7, 1.0, 0.5, 0.9])  # phi_i(X) (not specified)
EXP9_FRAILTY_ALPHA = 3.5
EXP9_LEVY_LAMBDA_TOTAL = 1.8
EXP9_P_COMMON = 1.0        # single common-jump Lévy driver shared by all n names
EXP9_MARK_RATES = np.array([1.0, 0.9, 1.2, 1.1])
EXP9_DELTA0_CONTINUOUS_ALPHA = 1.0   # delta0(t) = t (continuous, Weibull alpha=1)
EXP9_DELTA0_JUMP_DATE = 1.0          # discontinuity date for the jump test
EXP9_DELTA0_JUMP_SIZE = 0.6
EXP9_MATURITIES = np.array([1.1, 1.4, 0.9, 1.7])
EXP9_N_PATHS = 150_000
