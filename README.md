# **RUL Prediction using LSTM and Hyperparameter Optimization**

This project implements Remaining Useful Life (RUL) prediction for NASA CMAPSS Turbofan Jet Engines using Long Short-Term Memory (LSTM) neural networks based off engine variables such as cycle times, operational settings, and sensor readings. The solution utilizes sliding windows for time-series data processing and Optuna for hyperparameter optimization across multiple datasets.


---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Dependencies](#dependencies)
4. [Data Source](#data-source)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)

---

## **Project Overview**

This project:
- Processes CMAPSS time-series datasets using sliding windows with padding and masking.
- Builds LSTM models for RUL prediction.
- Performs hyperparameter optimization using **Optuna** for window size, batch size, learning rate, and epochs.
- Ensures robust model evaluation with RMSE metrics.

---

## **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/rul-prediction-lstm.git
   cd rul-prediction-lstm

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   or, using conda,

   conda create --name rul-prediction python=3.8 -y
   conda activate rul-prediction

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt

## **Dependencies**
The following dependencies are required for this project:
	•	Python (3.12.2)
	•	TensorFlow (>=2.9.0)
	•	NumPy
	•	Pandas
	•	Scikit-learn
	•	Optuna
	•	Matplotlib