# 🧠 Brain Stroke Prediction Using Machine Learning

A robust machine learning pipeline to predict the likelihood of a **stroke** based on patient health records. This project incorporates **data preprocessing**, **feature encoding**, **balancing with SMOTE**, and **active learning** using **uncertainty sampling** during k-fold cross-validation to enhance prediction accuracy on imbalanced datasets.

---

## 📌 Project Overview

Brain strokes can lead to severe disability or death if not detected early. This project applies machine learning techniques to predict the probability of stroke occurrence using medical indicators such as BMI, glucose level, hypertension, heart disease, and other demographic and health features.

---

## 🧰 Technologies Used

* **Python 3.8+**
* **Pandas, NumPy** – data processing
* **Seaborn, Matplotlib** – visualization
* **Scikit-learn** – modeling and evaluation
* **XGBoost** – alternative classifier (optional)
* **Imbalanced-learn (SMOTE)** – dataset balancing
* **Active Learning** – uncertainty-based sampling
* **Jupyter / Google Colab** – development environment

---

## 📂 Dataset

* **Source:** [`brain_stroke.csv`](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* **Features include:**

  * `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`
* **Target variable:** `stroke` (0 = No Stroke, 1 = Stroke)

---

## 📈 Project Workflow

### 1. 📥 Data Loading & Inspection

* Read the dataset into a DataFrame
* Handled missing values and data types
* Identified categorical columns for encoding

### 2. 🔄 Data Preprocessing

* One-hot encoded categorical features
* Applied **MinMaxScaler** to `avg_glucose_level` and `bmi`
* Separated features (X) and target (y)

### 3. ⚖️ Dataset Balancing with SMOTE

* Applied **Synthetic Minority Oversampling Technique (SMOTE)** to address class imbalance
* Ensured the minority stroke class was adequately represented in training data

### 4. 🧠 Model Training with Active Learning

* Used **Random Forest Classifier** with **K-Fold Cross-Validation** (`k=5`)
* Incorporated **uncertainty sampling**:

  * Identified and added the most uncertain predictions into the training set iteratively
  * Improved learning on ambiguous cases

### 5. 📊 Evaluation Metrics

Tracked metrics across all folds:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve & AUC

---

## 🧪 Results

### ✅ Average Metrics (across folds):

| Metric    | Value (approx.) |
| --------- | --------------- |
| Accuracy  | \~High          |
| Precision | \~High          |
| Recall    | \~High          |
| F1 Score  | \~High          |

### 📌 Confusion Matrix

* Visualized the average confusion matrix from all k-folds

### 📈 ROC Curve

* Plotted ROC curve for each fold
* Overlayed with **mean ROC** and **AUC**

---

## 📁 Project Structure

```
brain_stroke_prediction/
│
├── brain_stroke.ipynb             # Main analysis and model training notebook
├── brain_stroke.csv               # Dataset (from Google Drive / Kaggle)
├── README.md                      # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/brain-stroke-prediction.git
   cd brain-stroke-prediction
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```

4. Open the notebook:

   ```bash
   jupyter notebook brain_stroke.ipynb
   ```

---

## ✅ Features

* ⚖️ Handles imbalanced medical data using SMOTE
* 🧪 Applies k-fold cross-validation for robustness
* 🧠 Integrates active learning for smarter training
* 📊 Evaluates with precision, recall, F1-score, and ROC/AUC
* 📈 Visualizes confusion matrix and model performance

---

## 🧠 Future Enhancements

* Deploy the model via Flask/Django for web-based predictions
* Extend to multi-model ensemble learning
* Integrate feature importance & SHAP explainability
* Real-time stroke prediction system for hospitals

---

## 📬 Contact

**Author:** Percy Owoeye
**GitHub:** [@Percy-O](https://github.com/percy-o)
**Email:** [owoeyepercyolawale@gmail.com](mailto:owoeyepercyolawale@gmail.com)
