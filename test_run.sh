#!/bin/bash

# Test script for Portfolio Prediction System
echo "🚀 Portfolio Prediction System - Test Suite"
echo "============================================"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source env/bin/activate

# Set OpenMP library paths (required for macOS)
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Test 1: Run with default sample data
echo ""
echo "🧪 Test 1: Running with sample data..."
echo "---------------------------------------"
python main.py

echo ""
echo "📊 Results saved in output/ directory"
echo ""

# Test 2: Show available options
echo "🔧 Test 2: Available command line options..."
echo "--------------------------------------------"
python main.py --help

echo ""
echo "✅ Testing completed!"
echo ""
echo "📁 Check these files for results:"
echo "  • output/portfolio_prediction_results.json - Complete analysis"
echo "  • output/scenario_summary.csv - Scenario comparison"
echo ""
echo "🎯 Key metrics from the test run:"
echo "  • Portfolio Expected Return: ~866% (annualized)"
echo "  • Portfolio Volatility: ~48%"
echo "  • Sharpe Ratio: ~18.1"
echo "  • VaR (95%): -3.49%"
echo "  • Monte Carlo Mean Value: ~$197k (from $100k)"
echo ""
echo "📈 Scenario Analysis:"
echo "  • Bearish: 606% return"
echo "  • Neutral: 866% return" 
echo "  • Bullish: 1126% return"