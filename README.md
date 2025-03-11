# Fantasy Premier League (FPL) Prediction Model

This project is a Python-based prediction model for Fantasy Premier League (FPL). It helps you select the optimal team for each game week by predicting player points based on **form**, **total points**, and **fixture difficulty**. The model uses linear programming to optimize team selection within FPL constraints.

---

## **Features**
- **Form Calculation**: Uses a rolling average of recent performance to predict player form.
- **Fixture Difficulty**: Incorporates FPL's Fixture Difficulty Rating (FDR) to adjust predictions.
- **Weighted Predictions**: Balances the influence of form and total points using customizable weights.
- **Optimization**: Selects the best team within FPL constraints (budget, squad size, position limits).

---

## **Requirements**
- Python 3.7+
- Required libraries: `pandas`, `numpy`, `requests`, `pulp`

Install the required libraries using:
```bash
pip install pandas numpy requests pulp