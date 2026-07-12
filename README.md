# Stock Return Prediction and Trading Backvalidation

This project builds and evaluates machine-learning models for predicting next-day stock returns for a selected universe of technology stocks. The goal is not only to minimize prediction error, but also to test whether the predictions can support a simple trading simulation under an expanding-window backvalidation setup.

The project currently uses daily OHLCV stock data, engineered technical features, several baseline models, classical ML models, and a fully connected neural network. It also contains early work for adding hourly price data and fundamental company data.

> **Important:** This project is for research and experimentation only. It is not financial advice and should not be used for real trading without much stronger validation, transaction-cost modeling, slippage modeling, and risk controls.

---

## Quick Start

### Requirements

- Python 3.11
- pip
- Git

### Setup

Move into the project folder:

```powershell
cd "C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

### Environment Variables

The project uses the Twelve Data API for market data. The API key should not be written directly in the code.

Create a local `.env` file for your private values. This file should stay on your computer and should not be uploaded with real secrets.

Use `.env.example` as the public template:

```text
TWELVE_DATA_API_KEY=your-twelve-data-api-key
```

For the current code, set the environment variable in PowerShell before running code that calls Twelve Data:

```powershell
$env:TWELVE_DATA_API_KEY = "your-real-api-key"
```

The `.env` file is useful as a local reference, but Python does not load it automatically unless the project later adds a package such as `python-dotenv`.

### Run The Website

```powershell
python main.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Run Tests

```powershell
pytest
```

### API Endpoints

The Flask app also exposes a small JSON API for developer tooling and future UI work:

```text
GET /api/status
GET /api/models
GET /api/datasets
```

#### `GET /api/status`

Checks that the API is alive and that the API routes are registered correctly.

Example response:

```json
{
  "service": "stock-return-prediction",
  "status": "ok",
  "version": "1.0"
}
```

#### `GET /api/models`

Lists the model flags supported by the project. This is useful for future UI pages or tools because they can discover model options without reading the Python source code.

Example response:

```json
{
  "models": [
    {
      "flag": 0,
      "name": "Dumb baseline"
    },
    {
      "flag": 1,
      "name": "XGBoost"
    }
  ]
}
```

#### `GET /api/datasets`

Lists local data artifacts without loading large files into memory. This is useful for developer tooling because it helps developers see which data files exist.

Example response:

```json
{
  "data_dir": "path/to/project/data",
  "file_count": 3,
  "files": [
    {
      "name": "tech_universe.csv",
      "type": "csv",
      "size_bytes": 1234
    }
  ]
}
```

### Notes

- The project uses Twelve Data for stock-market data.
- Some data is cached locally inside the `data/` folder.
- This project is for research and learning, not real trading.

---

## Project Goal

The main goal is to answer this question:

> Can historical stock-price features predict the next-day return well enough to beat simple baselines and produce useful trading decisions in a backtest?

The main prediction target is:

```text
next_day_return = close(t+1) / close(t) - 1
```

The project evaluates models by:

1. **RMSE** — how close the predicted return is to the real next-day return.
2. **Right direction accuracy** — whether the model predicts the correct sign of the next-day return.
3. **Right direction above threshold** — whether the model is correct only when it makes a sufficiently strong prediction.
4. **Trading simulation** — whether predictions produce profit/loss under a simple buy/sell rule.

---

## Where to Put This README

Put this file in the **root folder of the project**, not inside the `models/` folder and not inside the `data/` folder.

Recommended location:

```text
algo_trading_stocks/
├── README.md                  <- put it here
├── requirements.txt
├── runtime_config.py
├── data_preparing.py
├── data_preparing_hourly_prices.py
├── fundamentals_data_preparing.py
├── evaluation_and_simulation.py
├── data/
│   ├── tech_universe.csv
│   ├── *.pkl
│   └── ...
└── models/
    ├── model_base.py
    ├── base_line_models.py
    ├── ml_models.py
    └── neural_network_models.py
```

`README.md` is **not a Python file**. It is a Markdown documentation file. GitHub, GitLab, VS Code, PyCharm, and many other tools automatically display it as the main project explanation page.

---

## Main Files

### `requirements.txt`

Lists the Python packages needed for the project.

Main dependencies include:

```text
pandas
matplotlib
numpy
scikit-learn
requests
twelvedata
xgboost
tensorflow
```

### `runtime_config.py`

Asks where the code is running:

```text
1 = local machine
2 = Google Colab
```

Many file paths depend on this value. This is useful because the project is edited locally but may be run in Google Colab.

### `data_preparing.py`

Handles the main daily stock-data pipeline:

- Loads the selected technology-stock universe from `tech_universe.csv`.
- Fetches stock data from Twelve Data.
- Saves and loads data using pickle files.
- Creates return-based features.
- Creates rolling and relative features, such as:
  - previous-day return,
  - multi-day returns,
  - relative volume,
  - SMA features,
  - SMA gap percentages.
- Aligns dates across stocks.
- Splits the data into train/validation/test sets.
- Defines the `next_day_return` target.

### `data_preparing_hourly_prices.py`

Contains early work for collecting and extending hourly price data. The idea is to increase the amount of training data and later merge hourly features into the main daily dataset.

### `fundamentals_data_preparing.py`

Contains early work for collecting fundamental/company-profile fields such as sector and industry. This is currently limited because the relevant Twelve Data endpoint may not be available on the free plan.

### `model_base.py`

Defines the abstract model interface. Every model must implement:

```python
fit(x_train, y_train)
predict(x_test, train_set=None, val_set=None)
```

This makes the evaluation code work with different model types through the same interface.

### `base_line_models.py`

Contains simple baseline models:

- `DumbModel` — always predicts 0 return.
- `PreviousDayReturnModel` — predicts tomorrow's return using the previous daily return.
- `RollingAvgModel` — predicts tomorrow's return using a rolling average of previous returns.

These models are important because a machine-learning model is only useful if it beats simple baselines.

### `ml_models.py`

Contains classical machine-learning models:

- `XGBoostModel`
- `LinearRegressionModel`

### `neural_network_models.py`

Contains the fully connected neural network model:

- `FullyConnectedNeuralNetwork`

The current neural network is a feed-forward dense network using TensorFlow/Keras. It predicts a continuous next-day return, so the default loss is MSE.

### `evaluation_and_simulation.py`

This is the main evaluation and backtesting file. It contains:

- Model factory logic with model flags.
- Expanding-window evaluation.
- RMSE evaluation.
- Directional-accuracy evaluation.
- Threshold-based directional evaluation.
- A simple portfolio/trading simulation.
- Model comparison utilities.

---

## Model Flags

The evaluation code uses integer flags to select models:

| Flag | Model |
|---:|---|
| 0 | Dumb model |
| 1 | XGBoost model |
| 2 | Previous daily return baseline |
| 3 | Rolling average baseline |
| 4 | Linear regression |
| 5 | Fully connected neural network |

---

## Validation Types

The code currently supports these evaluation types:

| Flag | Validation Type | Meaning |
|---:|---|---|
| 1 | RMSE | Lower is better |
| 2 | Right direction | Higher is better |
| 3 | Right direction over threshold | Higher is better, but only counts confident predictions |

The trading simulation is implemented separately through `run_model_simulation_backvalidation()` and `model_simulation()`.

---

## Expanding-Window Evaluation

The project uses an expanding-window style evaluation:

1. Train on historical data from the initial start date up to the current validation date.
2. Predict the next validation day.
3. Store the prediction score.
4. Move one trading day forward.
5. Repeat until the end of the train/validation period.

This is the correct general direction for stock prediction because the model must not train on future data.

The code also correctly notes an important leakage risk:

> Scaling must be fit only on the current training window, not on the full dataset.

That is why MinMax scaling is done inside the evaluation loop for models that need scaling, such as linear regression and the neural network.

---

## Trading Simulation Logic

The simulation uses the model predictions to make buy/sell decisions.

Current simplified logic:

- Buy stocks whose predicted return is above `min_y_pred_to_buy`.
- Sell held stocks whose predicted return is below `max_y_pred_to_sell`.
- Use a maximum daily spending cap.
- Track cash, positions, number of buys, number of sells, and portfolio value.

Example parameters used in the code:

```python
initial_cash = 10_000.0
min_y_pred_to_buy = 0.02
max_y_pred_to_sell = -0.02
transaction_fee = 0.0
max_spending_for_a_day = 3_000.0
```

These rules are still basic. A serious backtest should later include transaction costs, bid/ask spread, slippage, position sizing, exposure limits, and benchmark comparison.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The main evaluation file is:

```bash
python evaluation_and_simulation.py
```

When the code starts, it may ask:

```text
please enter where the code runs.
enter 1 for locally, 2 for Google colab:
```

Choose:

```text
1
```

for local execution, or:

```text
2
```

for Google Colab.

---

## Data Files

The project expects data files under a `data/` folder. The exact paths currently depend on whether the code is running locally or in Google Colab.

Expected data includes:

```text
data/tech_universe.csv
data/five_thousand_days_data.pkl
data/five_thousand_days_data_experiment.pkl
data/experiment_train_and_validation_data.pkl
data/experiment_test_data.pkl
```

The pickle files are used so the API does not need to be called every time.

---

## Important Current Limitations

1. **API key should not be hard-coded.**  
   The Twelve Data API key is currently written directly in `data_preparing.py`. It should be moved to an environment variable or a private config file that is not committed to Git.

2. **Some paths are hard-coded.**  
   Several paths currently point to a specific local Windows folder or a Google Colab folder. This should eventually be replaced with `pathlib.Path` and a single project-root configuration.

3. **The neural-network expanding-window logic is not finished.**  
   The current code includes a TODO for changing the FCNN training style so that the model trains once, keeps its weights, predicts the next day, then updates on the new day.

4. **Backtest assumptions are simplified.**  
   The simulation currently does not fully model real execution costs, liquidity, spread, slippage, taxes, or realistic order timing.

5. **Feature engineering needs more validation.**  
   Every feature must be checked carefully to make sure it only uses information available before the prediction date.

6. **No final holdout-test evaluation yet.**  
   The project has a test split, but most experimentation appears to happen on the train/validation data. Final model selection should be followed by one clean evaluation on the untouched test set.

---

## Recommended Next Steps

### 1. Fix project structure

Make sure the import paths and folder structure match. Since the code imports from `models.*`, the model files should be inside a folder named `models`:

```text
models/model_base.py
models/base_line_models.py
models/ml_models.py
models/neural_network_models.py
```

Also add:

```text
models/__init__.py
```

### 2. Move secrets out of code

Replace the hard-coded API key with:

```python
import os
Twelve_data_API_key = os.getenv("TWELVE_DATA_API_KEY")
```

Then set the key outside the code.

### 3. Add a clean `main.py`

Right now, `evaluation_and_simulation.py` contains both functions and executable experiment logic. It would be cleaner to keep reusable functions in modules and run experiments from a separate file, for example:

```text
main.py
```

### 4. Improve the FCNN expanding-window method

For the neural network, the more realistic online-learning-style approach is:

1. Train once on the initial window.
2. Predict day `t+1`.
3. After the real value for `t+1` is known, update the same model weights.
4. Predict day `t+2`.
5. Continue forward.

This is different from retraining a new neural network from scratch every day and may be faster. It also better matches the idea of a model that learns through time.

### 5. Add experiment logging

Save each run's parameters and results to a CSV or JSON file:

```text
experiments/results.csv
```

Useful fields:

- model name,
- feature columns,
- train start date,
- validation range,
- RMSE,
- direction accuracy,
- threshold,
- profit/loss,
- number of trades,
- final portfolio value.

---

## Suggested Folder Structure

```text
algo_trading_stocks/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── runtime_config.py
├── data_preparing.py
├── data_preparing_hourly_prices.py
├── fundamentals_data_preparing.py
├── evaluation_and_simulation.py
├── data/
│   ├── tech_universe.csv
│   ├── five_thousand_days_data.pkl
│   ├── five_thousand_days_data_experiment.pkl
│   ├── experiment_train_and_validation_data.pkl
│   └── experiment_test_data.pkl
├── models/
│   ├── __init__.py
│   ├── model_base.py
│   ├── base_line_models.py
│   ├── ml_models.py
│   └── neural_network_models.py
└── experiments/
    └── results.csv
```

---

## Status

The project is currently in an active research/prototype stage. The core pipeline exists, including data preparation, model abstraction, multiple models, expanding-window validation, and a basic trading simulation. The next serious engineering step is to clean the project structure, remove hard-coded paths/secrets, and finish the neural-network expanding-window update logic.
