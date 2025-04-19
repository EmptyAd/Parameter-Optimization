# 🔍 SVM Parameter Optimization with Randomized Search

This project focuses on tuning hyperparameters for an SVM (Support Vector Machine) model using `RandomizedSearchCV` from scikit-learn. The goal is to maximize classification accuracy across multiple randomized train-test splits.

---

## 📊 Dataset

Used a multi-class classification dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). Features were scaled and used directly for SVM classification.

---

## ⚙️ Methodology

- **Model**: Support Vector Classifier (`SVC`)
- **Train-Test Split**: 70-30 ratio
- **Repetition**: 10 different random seeds (`random_state = 0 to 9`)
- **Search Strategy**: `RandomizedSearchCV`
  - **Hyperparameter Space**:
    - `C`: 10 values in log space (0.01 to 100)
    - `kernel`: `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'`
    - `gamma`: `'scale'`, `'auto'`
  - **Iterations**: 20 per run
  - **CV Folds**: 3
  - **Scoring Metric**: Accuracy

---

## 🧪 Results



---

## 📈 Fitness Plot (Convergence of Best Accuracy)

The plot below shows the accuracy trend across all 10 randomized samples. Each point represents the best cross-validation accuracy found in that iteration.

> ![Fitness Plot](path_to_your_image.png)

- **X-axis**: Sample iteration (S1 to S10)
- **Y-axis**: Accuracy
- **Insight**: Model consistently reaches ~97% accuracy using `rbf` kernel and appropriate `C`.

---

## ✅ Key Observations

- Highest accuracy achieved: **97.85%**
- Most successful kernel: `'rbf'`
- `'auto'` gamma consistently outperformed `'scale'`
- Ideal `C` values ranged between **1.6 to 100**

---

## 📁 Project Structure

