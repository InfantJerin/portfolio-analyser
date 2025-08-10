# portfolio-analyser
Project to analyze the portfolio. 
# Portfolio Prediction with Sentiment Analysis

A comprehensive portfolio forecasting system that combines traditional financial analysis with machine learning-based sentiment prediction to forecast portfolio performance and risk metrics.

## Features

- **Sentiment-driven ML predictions**: Uses LightGBM/XGBoost to predict asset returns based on sentiment data
- **PyPortfolioOpt integration**: Leverages proven portfolio optimization algorithms
- **Monte Carlo simulation**: Simulates thousands of possible portfolio outcomes
- **Bootstrap analysis**: Uses historical data blocks for scenario generation
- **Comprehensive risk analysis**: VaR, CVaR, drawdown analysis, and more
- **Scenario analysis**: Bullish/Neutral/Bearish market regime projections

## Installation

1. Create and activate the virtual environment:
```bash
source env/bin/activate  # On macOS/Linux
# or
env\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Sample Data

Run the complete pipeline with the provided sample data:

```bash
python main.py
```

### Custom Data

Run with your own data files:

```bash
python main.py \
  --portfolio data/your_portfolio.csv \
  --prices data/your_prices.csv \
  --sentiment data/your_sentiment.csv \
  --config config/your_config.json \
  --output results/
```

### Data Format Requirements

#### Portfolio Data (`portfolio.csv`)
```csv
ticker,weight,name,sector
AAPL,0.25,Apple Inc,Technology
GOOGL,0.20,Alphabet Inc,Technology
...
```

#### Price Data (`prices.csv`)
```csv
date,AAPL,GOOGL,MSFT,AMZN,TSLA,JPM,JNJ,PG
2024-01-02,185.64,140.93,376.04,151.94,248.86,168.15,157.25,155.48
2024-01-03,184.25,141.80,375.12,149.93,244.92,169.12,156.78,154.85
...
```

#### Sentiment Data (`sentiment.csv`)
```csv
date,market_sentiment,AAPL_sentiment,GOOGL_sentiment,MSFT_sentiment,...
2024-01-02,0.15,0.25,0.18,-0.05,0.12,0.32,-0.08,0.05,0.10
2024-01-03,-0.12,-0.08,-0.15,-0.22,-0.18,-0.25,-0.12,-0.05,-0.08
...
```

## System Architecture

### Core Components

1. **Data Ingestion** (`src/data_ingestion.py`)
   - Loads and validates portfolio, price, and sentiment data
   - Aligns data by date and handles missing values
   - Prepares data in PyPortfolioOpt compatible format

2. **Sentiment Predictor** (`src/sentiment_predictor.py`)
   - Trains ML models to predict returns from sentiment
   - Feature engineering with technical indicators
   - Time series cross-validation for robust model training

3. **Portfolio Analyzer** (`src/portfolio_analyzer.py`)
   - PyPortfolioOpt integration for risk and return calculations
   - Sentiment-adjusted expected returns
   - VaR/CVaR risk metrics calculation

4. **Monte Carlo Simulation** (`src/monte_carlo.py`)
   - Simulates thousands of portfolio return paths
   - Bootstrap resampling from historical data
   - Comprehensive risk metric calculation

5. **Scenario Analysis** (`src/scenario_analysis.py`)
   - Bullish/Neutral/Bearish scenario projections
   - Stress testing under extreme conditions
   - Historical regime analysis

### Pipeline Flow

```
Data Loading → ML Training → Risk Analysis → Simulations → Scenario Analysis → Results
```

## Output

The system generates:

1. **JSON Results** (`portfolio_prediction_results.json`)
   - Complete analysis results with all metrics
   - Sentiment predictions and model performance
   - Monte Carlo and bootstrap simulation results

2. **Scenario Summary** (`scenario_summary.csv`)
   - Comparison of performance across market scenarios
   - Expected returns, volatility, and Sharpe ratios

3. **Console Output**
   - Real-time pipeline progress
   - Key metrics and summary statistics
   - Model training performance

## Configuration

Customize the analysis through `config/config.json`:

```json
{
    "forecast_horizon": 20,
    "model_params": {
        "lightgbm": {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31
        }
    },
    "simulation_params": {
        "monte_carlo_iterations": 10000,
        "bootstrap_blocks": 1000
    },
    "scenario_thresholds": {
        "bullish": 0.2,
        "bearish": -0.2
    }
}
```

## Example Results

```
Key Results:
• Expected Return (Sentiment-Adjusted): 8.45%
• Portfolio Volatility: 15.23%
• Sharpe Ratio: 0.423
• VaR (95%): -2.34%
• Probability of Loss: 23.4%

Scenario Comparison:
• Bearish: 3.21%
• Neutral: 6.78%
• Bullish: 12.34%
```

## Requirements

- Python 3.8+
- PyPortfolioOpt 1.5.5+
- pandas, numpy, scikit-learn
- LightGBM for ML models
- matplotlib, plotly for visualizations

## Notes

- Ensure portfolio weights sum to 1.0
- Sentiment scores should be normalized (-1 to 1 or 0 to 100)
- Minimum 1-2 years of historical data recommended
- All dates should be in YYYY-MM-DD format