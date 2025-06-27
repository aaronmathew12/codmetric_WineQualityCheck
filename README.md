# 🍷 Wine Quality Prediction

This project uses machine learning to predict the quality of red wine based on various physicochemical properties. The dataset is sourced from the UCI Machine Learning Repository.

## 📊 Dataset

- **Name:** Wine Quality Data Set
- **Source:** [UCI Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **File Used:** `winequality-red.csv`
- **Features:** Fixed acidity, Volatile acidity, Citric acid, Residual sugar, Chlorides, Free sulfur dioxide, Total sulfur dioxide, Density, pH, Sulphates, Alcohol
- **Target:** Wine Quality (Score between 0 and 10)

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn (optional for visualization)

## 📌 Project Structure
wine-quality/
├── code.py # Main script for loading, training, and evaluating models
├── winequality-red.csv # Dataset (should be placed in the same directory)
└── README.md # Project documentation

🔍 What It Does

1. Loads the dataset and performs initial exploration.
2. Handles missing values and normalizes features.
3. Splits the data into training and testing sets.
4. Trains two machine learning models:
   - Logistic Regression
   - Decision Tree Classifier
5. Evaluates both models using:
   - Accuracy
   - Confusion Matrix
   - Classification Report

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/wine-quality-prediction.git
   cd wine-quality-prediction

2. Install dependencies
    ```bash
   pip install pandas scikit-learn
3. Run the script
   ```bash
   python code.py

## 📈 Future Improvements
Add visualizations (histograms, heatmaps)

Try more advanced models like Random Forest or XGBoost

Turn it into a Jupyter Notebook for easier exploration
