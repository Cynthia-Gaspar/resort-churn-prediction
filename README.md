# 🏖️ Resort Guest Churn Prediction

## 📌 Executive Summary

A luxury all-inclusive resort chain is losing guests — either they cancel their loyalty membership or simply don't return. This project builds a classification model to predict which guests are likely to churn based on booking behavior, on-site spending patterns, and loyalty data.

The final solution combines **HistGradientBoosting** with seed bagging, threshold optimization, and a LightGBM blend. The model achieves a **Kaggle leaderboard F1 of 0.850**, with a CV score of 0.8519 (a gap of less than 0.002) indicating a well-calibrated validation process with minimal overfitting.

## 📊 Data

| Parameter | Details |
|---|---|
| Source | Kaggle competition dataset — *Steve's Luxury Resorts* |
| Training set | 6,954 guests |
| Test set | 1,739 guests |
| Target | `Churned` (True = cancelled membership or did not return) |
| Features | Booking info · Room assignment · Spending by category · Loyalty points · Survey score · Email engagement |

## 📈 Key Results

| Model | CV F1 | LB F1 |
|---|---|---|
| Logistic Regression | 0.7198 | — |
| Random Forest | 0.8183 | — |
| HistGradientBoosting (default) | 0.8436 | — |
| HistGradientBoosting (tuned) | 0.8480 | — |
| **HistGB + Seed Bagging + LightGBM Blend** | **0.8519** | **0.850** |

The CV-to-LB gap of 0.0019 confirms the validation strategy was sound, that means that what the model learned on training data transferred cleanly to unseen data. The optimal decision threshold shifted to **0.465** (vs. the default 0.5), reflecting the near-balanced churn rate (~50/50). Each optimization step contributed measurably: hyperparameter tuning (+0.0044), seed bagging (+0.0027), and blending (+0.0012).

**What this means for business:** Guests on the all-inclusive package churn at 81.9% vs 32.8% for those without it, a gap of nearly 50 percentage points. European guests churn at 66.2%, nearly double the rate of Americas guests (42.5%). VIP guests, by contrast, churn at only 37.1%, confirming that loyalty program engagement is protective. The model's value goes beyond these individual signals: it combines all available guest information to rank every individual by their probability of churning, enabling the resort to prioritize retention campaigns before guests are lost.

## 🛠️ Technologies

Python · pandas · numpy · matplotlib · seaborn · scikit-learn · LightGBM

`HistGradientBoostingClassifier` · `LGBMClassifier` · `ColumnTransformer` · `Pipeline` ·
`StratifiedKFold` · `RandomizedSearchCV` · Seed Bagging · Threshold Optimization ·
Out-of-Fold Blending · Log Transformation · OneHotEncoding

## 🧭 Reflections and Learnings

- **EDA surfaced the most actionable insight:** The strongest predictor in the dataset was discovered during the Exploratory Data Analysis. Guests on the all-inclusive package churn at 81.9%, nearly double the rate of those without it. Exploratory analysis identified where the churn signal was concentrated before any model was built.

- **CV score and LB score can be made to agree:** The 0.0019 gap between cross-validation and leaderboard F1 came from using stratified k-fold consistently, applying all preprocessing inside the pipeline to prevent leakage, and validating on out-of-fold predictions rather than the training set.

- **Threshold tuning is underused and high-impact:** The default 0.5 threshold assumes symmetric costs for false positives and false negatives. Optimizing the threshold directly on F1 over the OOF predictions delivered one of the largest single gains in this project.

- **Seed bagging reduces variance without adding complexity.** Training the same model architecture across multiple random seeds and averaging the probability outputs is a low-cost way to stabilize predictions.

- **Blending works best when models are diverse.** The optimal blend was 70% HistGB / 30% LightGBM. The two models are different enough in their gradient computation to produce complementary errors, but similar enough in architecture that the blend remains coherent.

- **Log transformation on skewed spending data:** All five spending categories plus `spend_total` are heavily right-skewed. Applying `log1p` compresses the long tail so extreme values don't dominate. That said, `HistGradientBoosting` is a tree-based model, naturally more robust to scale than linear models. The transformation is a sound preprocessing choice, but its impact is moderate on the
final model.

---

*This project was completed as part of Machine Learning and AI a component of the Master of Management Analytics curriculum at Queen's University.*

## 👩‍💻 About the Author

Hi! I'm Cynthia Gaspar, a data scientist and analytics professional with a Master of Management Analytics from Queen's University. I'm passionate about understanding real-world problems through data. Let's connect!

🔗 [LinkedIn](https://www.linkedin.com/in/cynthia-gaspar)
