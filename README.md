# **RUL Prediction using LSTM and Hyperparameter Optimization**

This project implements Remaining Useful Life (RUL) prediction for NASA CMAPSS Turbofan Jet Engines using Long Short-Term Memory (LSTM) neural networks based off engine variables such as cycle times, operational settings, and sensor readings. The solution utilizes sliding windows for time-series data processing and Optuna for hyperparameter optimization across multiple datasets.

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
   git clone https://github.com/your_username/JetEngineRUL.git
   cd JetEngineRUL

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   or, using conda,

   conda create --name JetEngineRUL python=3.12.2 -y
   conda activate JetEngineRUL

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt

    or, using conda,

    conda install requirements.txt

4. **Download Engine Data**:
   https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data

## **Dependencies**
The following dependencies are required for this project:
 - Python (3.12.2)
 - TensorFlow (>=2.9.0)
 - NumPy
 - Pandas
 - Scikit-learn
 - Optuna
 - Matplotlib

## **File Structure**
```bash
   JetEngineRUL/
     ├── data/
     │   ├── raw/
     │   │   └── CMAPSSData/                     # Contains the C-MAPSS dataset
     │   └── processed/                          # Contains processed dataset
     ├── notebooks/
     │   ├── 01_eda.ipynb                        # Exploratory Data Analysis
     │   ├── 02_preprocessing.ipynb              # Data Preprocessing
     │   └── 03_modeling_and_evaluation.ipynb    # Modeling and Evaluation
     ├── results/
     │   ├── figures/                            # Figures from results
     │   ├── metrics/                            # Performance Metrics
     │   └── models/                             # Saved Models
     ├── requirements.txt                        # Python dependencies
     ├── LICENSE                                 # MIT License
     └── README.md                               # Project description and instructions
