# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sentiment-driven portfolio forecasting system that predicts asset returns and portfolio risk metrics using historical price data and sentiment scores. The system implements machine learning models (LightGBM/XGBoost) to forecast returns and provides risk analysis through Monte Carlo simulations and bootstrapping.

## Environment Setup

The project uses a Python virtual environment located in `env/`. To activate:
```bash
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows
```

Python version: 3.13.0 (managed via pyenv)

## Core Architecture

### Data Pipeline
- **Input**: Portfolio weights, historical prices, sentiment scores
- **Processing**: Feature engineering from sentiment and price history
- **ML Models**: Train LightGBM/XGBoost for next-day return prediction
- **Forecasting**: Generate mean returns and uncertainty measures

### Risk Analysis Components
- **Portfolio Metrics**: Expected return, volatility, VaR/CVaR calculation
- **Monte Carlo**: Simulate thousands of return paths using forecasted parameters
- **Bootstrapping**: Resample historical return blocks adjusted for current sentiment
- **Scenario Analysis**: Bullish/Neutral/Bearish regime projections

### Key Workflows
1. **Data Ingestion**: Upload portfolio, price history, sentiment data
2. **Model Training**: Train sentiment�returns models for each asset
3. **Forecasting**: Apply current sentiment to predict future returns
4. **Risk Calculation**: Combine forecasts with covariance for portfolio metrics
5. **Simulation**: Run Monte Carlo and bootstrap scenarios
6. **Output**: Present forecasts, risk metrics, and scenario analysis

## Expected File Structure

When implemented, the project should include:
- `data/` - Input data files (portfolio, prices, sentiment)
- `src/` or main modules for:
  - Data preprocessing and feature engineering
  - ML model training and prediction
  - Portfolio risk calculations
  - Monte Carlo simulation engine
  - Scenario analysis logic
- `notebooks/` - Jupyter notebooks for analysis
- `requirements.txt` - Python dependencies
- `config/` - Configuration files for models and parameters

## Development Notes

- Focus on financial time series analysis and ML forecasting
- Implement proper data validation for financial data inputs
- Consider performance optimization for Monte Carlo simulations
- Ensure risk calculations follow standard financial formulas
- Handle missing data and edge cases in sentiment/price data


Running the app
 source env/bin/activate
  python main.py
  2. Run Test Suite:
  ./test_run.sh
  3. Custom Data:
  python main.py --portfolio your_portfolio.csv --prices your_prices.csv --sentiment your_sentiment.csv
  4. Check Help:
  python main.py --help