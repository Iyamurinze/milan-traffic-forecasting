# Forecasting Mobile Network Traffic in Milan

A comparative study of three sequential models for **one-step-ahead** forecasting of mobile
Internet traffic, using the Telecom Italia Big Data Challenge grid for the city of Milan.

> **Research question.** How do different sequential models compare for one-step-ahead mobile
> network traffic forecasting, and how does their performance vary across geographical areas with
> different traffic characteristics?

Everything runs in **Google Colab**. The three notebooks are self-contained and pass data to each
other through Google Drive.

---

## Quick start

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**.
2. Upload and run the notebooks **in order**:

| # | Notebook | Covers | Runtime | Time |
|---|---|---|---|---|
| 1 | `colab/01_data_pipeline.ipynb` | Download, memory benchmark, streaming ETL → Parquet | CPU | ~30 min |
| 2 | `colab/02_eda.ipynb` | Distribution, five areas, MSTL, ACF/PACF, stationarity, anomalies | CPU | ~5 min |
| 3 | `colab/03_models.ipynb` | SARIMA + LSTM + TCN, tuning, seed variance, final evaluation | **T4 GPU** | ~40 min |

**Before notebook 3:** Runtime → Change runtime type → **T4 GPU**.

Each notebook mounts Drive and reads/writes `MyDrive/milan_traffic_project`, so notebook 1 only
needs running once.

Dependencies are installed by each notebook's first cell (`pyarrow`, `statsmodels`, `tqdm`,
`psutil`); `requirements.txt` lists them for anyone running outside Colab.

---

## The data

| | |
|---|---|
| Source | Harvard Dataverse, [`doi:10.7910/DVN/EGZHFV`](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV) |
| Coverage | 1 Nov 2013 – 1 Jan 2014 (62 days) |
| Grid | 10,000 areas (100 × 100), ~235 m per side |
| Resolution | 10 minutes → 144 intervals/day, 8,928 total |
| Raw size | 62 TSV files, **19.38 GiB**, ~4.84 M rows/day |
| Licence | ODbL 1.0 — cite Barlacchi *et al.*, *Sci Data* **2**, 150055 (2015) |

Schema (headerless, tab-separated, empty field = no activity):

```
square_id  time_ms  country_code  sms_in  sms_out  call_in  call_out  internet
```

**The files sit behind a Dataverse guestbook**, so a plain `GET` returns HTTP 400 and
`?gbrecs=true` does not help. Notebook 1 handles this by POSTing a guestbook response and
downloading the signed URL it returns — **no account or API token is required**, only an e-mail
address. Change `GUESTBOOK_EMAIL` in notebook 1 to your own.

---

## Data handling and memory management

The raw feed is keyed by `(square_id, time_ms, country_code)` across ~250 country codes, which is
nuisance structure for this problem. Collapsing it turns 4,842,625 rows per day into **1,439,982
populated cells out of a possible 1,440,000** — the grid is 99.999% dense, so a matrix is the right
representation and a row store the wrong one.

Measured on one real day-file (notebook 1 reproduces this):

| Strategy | Resident / day | Peak RSS | Projected, 62 days |
|---|---|---|---|
| Naive `read_csv`, 8 cols, inferred dtypes | 295.6 MB | +590 MB | **17.9 GB** |
| + column pruning (3 cols) | 110.8 MB | +126 MB | 6.7 GB |
| + dtype downcast (`uint16` / `float32`) | 64.7 MB | +395 MB | 3.9 GB |
| + chunked, country collapsed | 27.5 MB | +225 MB | **flat in day count** |

Two findings matter more than the headline ratio:

- **Downcasting alone makes the transient peak worse** (+126 → +395 MB) even as resident size falls.
  The parser materialises columns in its inferred types and casts afterwards, so both briefly
  coexist. Specifying dtypes is not a substitute for chunking.
- **Chunk size buys memory, not speed** — 18 MB peak at 250,000 rows vs 606 MB at 2,000,000, with
  throughput flat to within 5%. Hence the 250,000 default.

The raw corpus stays on Colab's local disk and is never written to Drive: 19.4 GiB exceeds a free
Drive quota, and the FUSE mount is far slower than local disk. Only the compact Parquet result
(~400 MB) is persisted.

**Trade-off.** Aggregating across country codes at ingestion is irreversible. Acceptable here
because the research question concerns only total Internet traffic per area, but recovering that
breakdown would mean re-downloading 19.4 GiB.

---

## Models

| Model | Family | Why it is here |
|---|---|---|
| **SARIMA** | Linear statistical | The literature's reference point; interpretable; built for the seasonal-plus-trend structure the EDA finds |
| **LSTM** | Recurrent neural | The default in mobile-traffic forecasting; gating spans a long window without vanishing gradients |
| **TCN** | Convolutional neural | Reaches comparable context by dilated causal convolution — parallel in time, a different inductive bias |
| *Persistence* | *Reference* | *`x̂(t+1) = x(t)`. Lag-1 autocorrelation exceeds 0.95, so this is a strong forecast that every model must beat* |
| *Seasonal naive* | *Reference* | *`x̂(t+1) = x(t+1−144)`. Measures how much is pure calendar repetition* |

A seasonal period of *s* = 144 makes the textbook SARIMA state space intractable — 146 state
dimensions with a cubic Kalman update, and a single fit that does not finish in ten minutes.
Notebook 3 applies the seasonal difference explicitly instead, reducing the state to two, and
compares that against harmonic (Fourier) regressors.

---

## Forecasting protocol

At time *t* a model receives the observed history up to and including *t* and predicts *x(t+1)* —
**rolling one-step-ahead with observed history**, not recursive multi-step, so errors never compound.

Splits are strictly chronological:

| Split | Period | Purpose |
|---|---|---|
| Train | 1 Nov – 8 Dec | Parameter estimation |
| Validation | 9 – 15 Dec | Early stopping |
| **Test** | **16 – 22 Dec** | **Reported once, informs no decision** |

Hyperparameters are chosen on a *tuning split shifted one week earlier*, so the reported week plays
no part in model selection.

Notebook 3 asserts the protocol rather than assuming it: for every model it perturbs the series
after a cut point and checks that earlier forecasts are unchanged, and that later ones do change.
A model that peeks at the future scores beautifully and forecasts nothing.

---

## Repository layout

```
colab/
  01_data_pipeline.ipynb   Download, memory benchmark, streaming ETL -> Parquet on Drive
  02_eda.ipynb             Exploratory and time-series analysis
  03_models.ipynb          The three models, tuning, seed variance, final evaluation
reports/
  report.md                Report source
  report.pdf               The submitted report (20 pages)
  assets/                  Figures extracted from the executed notebooks
data/
  manifest.json            The 62 source files with sizes and MD5 checksums
```

---

## Notes on validity

- The notebook ETL was checked against an independent pandas aggregation of a real day-file:
  per-square daily totals agree to a relative difference of ~1e-9 (float32 storage precision).
- Notebooks 2 and 3 were executed end-to-end before publication, with reduced training budgets, to
  confirm every cell runs and every artefact is produced.
- Notebook 3 measures **seed-to-seed variance** of a fixed architecture and compares it against each
  hyperparameter sweep's spread, so tuning conclusions that fall inside the noise band are reported
  as such rather than claimed as findings.

## Use of AI

See the *Use of AI* statement in the report.

## References

See the report's reference list (`reports/report.pdf`, section 9).
