
This project implements a **time series forecasting model** to predict stock closing prices using **Recurrent Neural Networks (RNNs)**, specifically **LSTM (Long Short-Term Memory)**.

We use historical stock data from **Yahoo Finance** via the `yfinance` API and train models using:

* 🔹 Manual PyTorch implementation
* 🔹 High-level Deeplay framework

---

## 📊 Dataset

* Source: Yahoo Finance (via `yfinance`)
* Stock: **AAPL (Apple Inc.)**
* Period: **2008 – 2024**
* Feature used: `Close` price

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Extract closing prices
* Normalize data using training mean and standard deviation
* Convert time series into sequences using a sliding window

```python
sequence_length = 60
```

Each sample:

* Input → last 60 days
* Target → next day price

---

### 2. Sequence Creation

```python
def create_sequence(df, sequence_length=60):
```

Transforms:

```
[ x₁, x₂, ..., x₆₀ ] → predict x₆₁
```

---

### 3. Model Architecture (Manual PyTorch)

```python
rnn = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
fc = nn.Linear(32, 1)
```

#### 🔁 Forward Pass

1. Input → LSTM
2. Extract last hidden state
3. Pass through Linear layer
4. Output → predicted price

---

### 4. Training Setup

* Loss Function: **MAE (L1 Loss)**
* Optimizer: **Adam**
* Epochs: 100
* Batch Size: 32

---

### 5. Deeplay Implementation

Same architecture implemented using:

```python
dl.RecurrentModel(
    in_features=1,
    hidden_features=[32],
    out_features=1,
    rnn_type='LSTM'
)



## 🧠 Key Learnings

* RNN/LSTM can model sequential financial data
* Proper data preprocessing is critical
* Last hidden state summarizes sequence information
* Deeplay simplifies deep learning workflows

---


## 🔮 Future Improvements

* Use **GRU / Transformer models**
* Add technical indicators (RSI, MACD)
* Use proper **time-based splitting**
* Multi-step forecasting
* Hyperparameter tuning

---

## 👨‍💻 Author

Gihara Jayasinghe

---
