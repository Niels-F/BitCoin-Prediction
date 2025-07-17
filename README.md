# 🧠 Bitcoin Price Prediction with Echo State Network (ESN)

This project is a simple proof-of-concept (PoC) to explore the use of Echo State Networks (ESNs) for time series prediction — specifically, predicting Bitcoin prices based on historical data. The notebook is self-contained and designed for quick experimentation using a custom RC (Reservoir Computing) implementation.

---

## 📘 Contents

- `bitcoin_prediction.ipynb`: Jupyter Notebook that performs data loading, preprocessing, ESN training, and evaluation.
- `RC_ESN.py`: Custom implementation of the Echo State Network class used in the notebook.

---

> 💡 Optional: If you're using VS Code, register the kernel:

```bash
python -m ipykernel install --user --name=esn_env --display-name "Python (ESN)"
```

---

## ▶️ How to Run

1. Clone this repository (or copy the files to your working folder).
2. Ensure `RC_ESN.py` is in the same folder as `bitcoin_prediction.ipynb`.
3. Open the notebook in Jupyter Lab or VS Code.
4. Run the notebook cells step by step.

---

## 📊 Description

The notebook performs the following:

* Downloads historical Bitcoin data via [yfinance](https://pypi.org/project/yfinance/)
* Normalizes and prepares the time series data
* Initializes and trains a custom Echo State Network (`ESN` from `RC_ESN.py`)
* Predicts Bitcoin price movements
* Evaluates performance (e.g., MAE)

---

## 🧪 Notes

* This is **not a trading model**, just an educational demonstration of ESNs.
* You can improve performance using hyperparameter tuning (e.g., grid search or MCMC).
* 

---

## 📄 License

Apache License
