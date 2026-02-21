# TimeGAN — Synthetic Financial Time Series

> Generating realistic multivariate financial time series with Time-series Generative Adversarial Networks, applied to six cross-asset equity and macro instruments.

---

## Motivation

High-quality financial data is expensive, regulated, and scarce. Firms operating under strict data governance constraints — or researchers working with short historical windows — often cannot access enough samples to train robust quantitative models. Synthetic data generation offers a principled solution: if a generative model successfully captures the statistical structure of real financial returns, synthetic samples can augment training sets, stress-test risk models, or serve as realistic simulation environments.

Financial time series present a uniquely demanding test for generative models. Unlike natural images or text, financial returns exhibit fat tails, volatility clustering (ARCH effects), and regime-dependent cross-asset correlations that shift dramatically across macroeconomic cycles. A model that merely reproduces the marginal distribution of each asset independently — without capturing these temporal dynamics and multivariate dependencies — is of little practical value. TimeGAN was chosen precisely because its architecture is designed to address this: the supervised training phase explicitly forces the Generator to learn stepwise conditional distributions rather than treating each timestep independently.

---

## Architecture

TimeGAN consists of five GRU-based components trained in three sequential phases:

```
                         ┌─────────────────────────────────────────────┐
                         │              TRAINING PHASES                 │
                         └─────────────────────────────────────────────┘

 PHASE 1 — Autoencoder Pre-training
 ────────────────────────────────────────────────────────────
  Real data X ──► [ Embedder H ] ──► Latent H ──► [ Recovery R ] ──► X̃
                                         │
                               Minimise MSE(X, X̃)

 PHASE 2 — Supervised Pre-training (TimeGAN's key innovation)
 ────────────────────────────────────────────────────────────
  Real data X ──► [ Embedder H ] ──► H_t ──► [ Supervisor S ] ──► Ĥ_{t+1}
                                                      │
                                          Minimise MSE(H_{t+1}, Ĥ_{t+1})

 PHASE 3 — Joint Adversarial Training
 ────────────────────────────────────────────────────────────
  Noise Z ──► [ Generator E ] ──► Ê ──► [ Supervisor S ] ──► Ĥ ──► [ Recovery R ] ──► X̂  (synthetic output)
                                    │                   │
                                    └───► [ Discriminator D ] ◄─── [ Embedder H ] ◄── Real X
                                                  │
                                     Adversarial + Supervised + Moment losses

 INFERENCE
 ────────────────────────────────────────────────────────────
  Noise Z ──► [ Generator E ] ──► [ Supervisor S ] ──► [ Recovery R ] ──► Synthetic X̂
```

**Why three phases?**
The Embedder/Recovery pre-training ensures the Generator is trained against a meaningful latent representation rather than raw noise. The Supervisor pre-training teaches temporal dynamics before adversarial pressure is introduced. Joint training then refines all components simultaneously with four coordinated loss terms: adversarial (×2 paths), supervised temporal, and moment-matching.

---

## Assets & Rationale

| Ticker | Asset Class | Role in the dataset |
|--------|-------------|---------------------|
| SPY | US Equity Index | Broad market factor; baseline for correlation analysis |
| GLD | Gold | Safe-haven; near-zero equity correlation in normal regimes, negative in crises |
| TLT | Long-term US Treasuries | Regime-sensitive: negative SPY correlation pre-2022, near-zero post rate hikes |
| XLE | Energy Sector | Commodity-driven; partially decorrelated from broad market |
| MSFT | Large-cap Technology | High SPY correlation (~0.85) but distinct idiosyncratic volatility |
| JPM | Financials | Rate-sensitive; interesting counterpoint to TLT dynamics |

**Data period:** 2010-01-01 → 2024-12-31 (3,773 trading days)

Four distinct macroeconomic regimes are present in this window:
1. **2010–2019** — Post-GFC bull market, compressed volatility, falling correlations
2. **Mar–Sep 2020** — COVID crash and recovery, explosive volatility spike
3. **2022** — Aggressive Fed rate hike cycle, historically unusual SPY/TLT positive correlation
4. **2023–2024** — AI-driven rally, renewed risk appetite

---

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Discriminative AUC | 0.8657 | A simple GRU classifier separates real from synthetic with high confidence — the synthetic data is visually plausible but statistically detectable |
| TROR MAE (baseline) | 0.02762 | Upper bound: model trained and tested on real data |
| TSTR MAE | 0.01349 | Model trained on synthetic, tested on real |
| TSTR / TROR ratio | 0.49 | Below 1.0, but misleading — see note below |
| Mean \|Δ Correlation\| | 0.0560 | Moderate error in cross-asset correlations; broad structure preserved but relationships are imprecise |

**Honest assessment:** The results are mixed. The discriminative AUC of 0.87 indicates that the synthetic data is readily distinguishable from real data by even a shallow classifier — a clear sign that TimeGAN has not fully reproduced the statistical texture of the original series. The correlation structure is partially preserved (mean error of 0.056 on a [−1, 1] scale), which suggests the model captured some cross-asset relationships but not with high fidelity.

The TSTR/TROR ratio of 0.49 is not a positive result despite appearing below 1.0. A ratio well below 1.0 reflects that the synthetic data is *smoother and more predictable* than real financial data — the Generator has learned a simplified version of the distribution that lacks the volatility clustering, fat tails, and regime shifts that make real returns hard to forecast. Training a model on this synthetic data and testing it on real data yields artificially low error because the synthetic test targets are easier, not because the synthetic training data is richer.

This is a known limitation of TimeGAN on financial data at this scale (3,700 windows, 6 assets, 10k training steps). The project is best understood as a working implementation of the architecture rather than a production-ready data augmentation tool.

---

## Project Structure

```
timegan-financial/
├── 01_eda.ipynb                  # Data download, EDA, preprocessing, window creation
├── 02_timegan_training.ipynb     # TimeGAN architecture, 3-phase training, generation
├── 03_evaluation.ipynb           # Discriminative score, TSTR, moments, ACF, correlations
├── README.md
├── data/                         # Created by notebook 01
│   ├── windows.npy               # (n_windows, 24, 6) — sliding windows, normalised
│   ├── scaled_data.npy           # (T, 6) — full MinMax-scaled price series
│   ├── prices_raw.npy            # (T, 6) — original adjusted close prices
│   ├── scaler.pkl                # Fitted MinMaxScaler for inverse-transforming
│   ├── tickers.json
│   ├── dates.json
│   ├── prices.csv
│   └── log_returns.csv
└── output/                       # Created by notebook 02
    ├── generated_data.npy        # (n_windows, 24, 6) — synthetic windows, normalised
    ├── generated_data_rescaled.npy  # Inverse-transformed to original price scale
    ├── training_losses.pkl       # All phase loss histories
    └── *.png                     # Loss curves and sanity check plots
```

---

## How to Run

All notebooks are designed to run on **Google Colab (Free Tier, T4 GPU)**.

**Step 1 — Open each notebook in Colab**
Upload `01_eda.ipynb`, `02_timegan_training.ipynb`, and `03_evaluation.ipynb` to your Google Drive, or open them directly from GitHub via the Colab badge.

**Step 2 — Enable GPU**
`Runtime → Change runtime type → T4 GPU`

**Step 3 — Run notebook 01**
This downloads data, runs EDA, and saves all artefacts to `data/`. No GPU needed.

**Step 4 — Run notebook 02**
Trains TimeGAN for 30,000 total steps across three phases (~30–60 min on T4). Saves synthetic data to `output/`.

**Step 5 — Run notebook 03**
Loads real and synthetic windows and runs all evaluation metrics. Saves plots to `output/`.

> **Note on file persistence in Colab:** Colab's `/content/` directory is ephemeral. Either mount Google Drive at the start of each session or re-run upstream notebooks to regenerate artefacts.

---

## Requirements

```
tensorflow >= 2.12
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
scipy
yfinance
```

Install in Colab:
```bash
pip install yfinance statsmodels -q
```
All other dependencies are pre-installed in the Colab environment.

---

## References

- Yoon, J., Jarrett, D., & van der Schaar, M. (2019). **Time-series Generative Adversarial Networks.** *Advances in Neural Information Processing Systems (NeurIPS 2019)*. https://arxiv.org/abs/2005.00139

- Jansen, S. (2020). **Machine Learning for Algorithmic Trading** (2nd ed.), Chapter 21: Generative Adversarial Networks. Packt Publishing. https://github.com/stefan-jansen/machine-learning-for-trading

