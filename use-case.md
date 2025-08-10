User Scenario: Sentiment-Driven Portfolio Forecasting
Who’s using it?
An analyst in the investment team. They already have:

A list of assets in their portfolio (with weights).

Historical daily prices for those assets.

Historical daily sentiment scores for those assets or for the overall market.

What the analyst wants to do

Upload their portfolio, price history, and sentiment data.

Click “Run Forecast” (or run a command).

See:

Predicted returns for each asset for the chosen horizon (e.g., next 20 trading days).

Expected portfolio return, volatility, and risk metrics like VaR/CVaR.

Monte Carlo simulation results showing possible future portfolio values.

Bootstrapped scenario results based on historical data blocks.

“Bullish / Neutral / Bearish” scenario projections based on sentiment levels.

How it works behind the scenes (high-level)

Sentiment → Asset Returns

The system uses historical sentiment and price data to train a model (e.g., LightGBM/XGBoost) that predicts next-day returns for each asset.

It applies the current sentiment data to produce forecasted mean returns and uncertainty (std deviation) for each asset.

Portfolio Risk Forecasting

The system combines the forecasted returns with the covariance between assets to calculate expected portfolio return, volatility, and risk measures.

Simulations

Monte Carlo: Runs thousands of simulated return paths using forecasted means and covariances to estimate the range of possible portfolio values.

Bootstrapping: Resamples blocks of historical returns to create alternative scenarios, adjusted based on current sentiment.

Scenario Analysis

Defines three regimes:

Bullish: sentiment is high, returns are shifted upward.

Neutral: sentiment is average, returns unchanged.

Bearish: sentiment is low, returns are shifted downward.

Shows projected portfolio outcomes for each regime.

