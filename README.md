# KTND

**Koopman-Thermodynamic Neural Decomposition. A neural network learns a non-reversible Koopman
operator from daily market returns, then reads irreversibility, entropy production, and regime
timescales straight off its spectrum.**

![The 15 learned Koopman eigenvalues of the 11-ETF model in the complex plane, all inside the unit circle, most of them in conjugate pairs off the real axis](outputs/figures/fig1_eigenvalue_spectrum.png)

Textbook finance assumes prices move like a reversible random walk: play the tape backwards and
it looks statistically the same. In physics that assumption is called detailed balance, and
markets are widely thought to break it: they tend to fall fast and recover slowly. The usual way to show this is one
scalar asymmetry statistic. KTND learns the market's transfer operator instead, with separate
encoders for the present and the future so the operator is free to be non-symmetric. A reversible
system would give real eigenvalues. Complex ones mean the dynamics rotate through state space in
a preferred direction, and the size of that rotation is a measurable departure from equilibrium,
broken down mode by mode.

Manuscript draft: **`paper/main.tex`** (written for Physical Review E; some of its numbers come
from earlier runs than the result files, see Honesty).

```
daily log returns: SPY (1993 to 2026) or 11 cross-asset ETFs (2007 to 2026)
    ->  delay embedding (5 lags for SPY, 2 per ETF), z-scored on training-period statistics
    ->  dual-lobe VAMPnet: independent encoders for x(t) and x(t + 5 trading days)
    ->  Koopman matrix K: complex eigenvalues, left and right eigenfunctions
    ->  spectral gap, per-mode entropy production, irreversibility field I(x)
    ->  surrogate tests, NBER regime detection
```

Irreversibility, read off a spectrum.

## The core idea

1. **Two encoders, not one.** A standard VAMPnet shares weights between time t and t + τ, which
   forces a self-adjoint operator with real eigenvalues, so it cannot see irreversibility at all.
   KTND gives the two lobes independent weights.
2. **Complex eigenvalues are probability currents.** Each eigenvalue
   λ_k = exp(-(γ_k + iω_k)τ) splits into a decay rate γ_k and a rotation ω_k. The per-mode
   entropy production Ṡ_k = ω_k² A_k / γ_k is zero for every real mode, so only rotating modes
   carry irreversibility.
3. **Irreversibility has a location.** The field I(x) = Σ_k σ_k |u_k(x) - v_k(x)|² compares the
   left and right eigenfunctions point by point. It vanishes exactly where the projected dynamics
   satisfy detailed balance.
4. **The spectral gap is a timescale.** Δ = |Re ln λ_2| / τ bounds how long a regime persists
   (at least 1/Δ trading days).

## What it does (measured on daily ETF returns, seed 42 unless noted)

| Measurement | SPY, K = 5 modes | 11 ETFs, K = 15 modes |
|---|---|---|
| Complex Koopman modes (\|Im λ\| > 0.05) | 2 of 5 | **10 of 15** |
| Complex modes, mean of 5 seeds | 1.6 ± 0.9 | 9.6 ± 1.7 |
| Spectral gap Δ (persistence bound 1/Δ) | 0.196 (5.1 days) | 0.168 (6.0 days) |
| Entropy production, k-NN estimator | 0.26 | 0.31 |
| Surrogate test, Cohen's d (200 surrogates) | **31.0** (IAAFT, mean irreversibility) | 7.1 (PCA-projected permutation, asymmetry) |
| NBER recession F1, mean of 5 seeds | 0.345 ± 0.004 | **0.511 ± 0.013** |

On SPY, no IAAFT surrogate (same power spectrum and amplitude distribution, temporal asymmetry
destroyed) came close to the observed mean irreversibility: Cohen's d of 31.0. For recession
detection, a Gaussian HMM fit to the top eigenfunctions of the 11-ETF model reaches F1 0.52 on
seed 42, against 0.37 for VIX thresholds, 0.34 for an LSTM autoencoder, 0.21 for DMD and for
PCA with k-means, and 0.03 for a 3-state HMM on returns.

Results are in `outputs/results/`: `analysis_results_{univariate,multiasset}.json`,
`multi_seed_summary.json`, `statistical_tests*.json`, `baseline_comparison.csv`,
`last_minute.json`, and `acceptance_fixes.json`.

## How it works

**Data.** Daily closes from Yahoo Finance, committed in `data/prices.csv` (SPY from 1993, plus
the 11 cross-asset ETFs and 8 sector ETFs) and `data/vix.csv`. The multiasset set is SPY, QQQ,
IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, and VNQ, which starts in 2007 when HYG lists. Log returns
are standardized with training-period statistics only. Splits are chronological: test is 2020
onward.

**Model.** `src/model/vampnet.py`: two MLP lobes (Linear, BatchNorm, ELU, dropout 0.1) with
hidden sizes [64, 64, 32] for SPY and [128, 128, 64] for the ETFs. Whitened covariances give the
Koopman matrix, its SVD gives the VAMP-2 score, and its eigendecomposition gives the spectrum and
the left and right eigenfunctions.

**Training.** Lag τ = 5 trading days. Loss = negative VAMP-2 plus an orthogonality penalty
(weight 0.005), an entropy-consistency term (0.1), and a spectral penalty (1e-5). Adam at 3e-4,
weight decay 1e-5, cosine schedule, up to 800 epochs with patience 80. Five seeds (0, 1, 2, 3, 42).

**Validation.** On synthetic data first: a non-reversible double well and a Brownian gyrator with
known analytical entropy production, both in `tests/test_synthetic.py`. On market data: IAAFT
surrogates, Chapman-Kolmogorov consistency, block-bootstrap eigenvalue intervals, Ljung-Box,
Kolmogorov-Smirnov stability of eigenfunctions between train and test, bidirectional Granger
tests against VIX, a model-free time-reversal asymmetry statistic, and 5-fold walk-forward
cross-validation.

**Baselines.** Gaussian HMM, DMD, PCA with k-means, GARCH(1,1), VIX thresholds, and an LSTM
autoencoder, all scored against NBER recession dates with the label-to-recession mapping learned
on pre-2018 data.

## Run it

```bash
pip install -r requirements.txt
python data/download.py                        # optional: refresh data/ from Yahoo Finance
python experiments/run_all.py                  # train both modes, baselines, rolling, statistical tests, CV, figures
python experiments/run_all.py --modes univariate --resume-from 5
```

Stages also run on their own:

```bash
python experiments/run_main.py --config config/default.yaml --mode multiasset
python experiments/run_robustness.py --mode univariate      # needs a trained checkpoint
python experiments/run_cv.py --mode multiasset
python experiments/run_entropy_calibration.py               # Brownian gyrator benchmark
python experiments/run_ablations.py
python experiments/run_figures.py
pytest tests/
```

`./run.sh --fast` runs the univariate pipeline only. `KTND_Finance_Colab.ipynb` runs the same
stages on a Colab GPU.

## Layout

```
src/
  model/             dual-lobe VAMPnet, Koopman analysis, entropy estimators, irreversibility field, losses
  analysis/          spectral, regime (HMM and NBER), rolling windows, Chapman-Kolmogorov, statistical tests
  data/              preprocessing, delay embedding, loaders, IAAFT surrogates
  baselines/         HMM, DMD, PCA, GARCH, LSTM autoencoder, VIX thresholds
  constants.py       date splits, tickers, NBER dates
experiments/         run_all orchestrator, training, baselines, CV, statistical tests, calibration, figures
config/              default, univariate, multiasset, and 13 ablation configs
data/                prices.csv, vix.csv, download.py
outputs/figures/     paper and supplemental figures (PDF and PNG)
outputs/results/     every result file quoted above
tests/               143 test functions, including synthetic double-well and gyrator checks
paper/               manuscript draft and bibliography
```

## Honesty

* The irreversibility evidence is mixed. On SPY the mean irreversibility field separates cleanly
  from IAAFT surrogates (d = 31.0), but the detailed-balance statistic tested alongside it does
  not (d = -1.32, combined p = 0.9), and the SPY bootstrap finds no eigenvalue phase whose 95%
  interval excludes zero. On the 11 ETFs the surrogate test falls back to a PCA-projected
  permutation.
* Entropy production numbers are relative, not absolute. On the Brownian gyrator, where the true
  rate is known, only the Frobenius estimator tracks it (r = 0.94), and it reads roughly a
  thousand times too small. On SPY the four estimators disagree by orders of magnitude.
* The Chapman-Kolmogorov test fails for SPY at τ = 5, and the multiasset run reports all-zero
  errors, which points to a bug rather than a pass.
* Regime results are thin. Recession F1 is scored over the full sample, so only the 2020
  recession is out of sample, and accuracy is below always predicting expansion. The rolling
  crisis signal in `last_minute.json` uses centered windows that see 250 trading days ahead, so
  it is not a forecast and is not quoted here.
* The manuscript quotes numbers from earlier runs (8 of 10 complex modes, d = 33.0, 1,000
  surrogates) that the current result files do not reproduce. The result files are the reference.
* Trained checkpoints are not committed, so the statistical-test and rolling stages need a
  training run first.

MIT licensed.
