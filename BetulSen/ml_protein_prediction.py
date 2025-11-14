# ============================================================
# Predicting Protein Levels From Gene Expression (Fake Dataset)
# Simulated Multi-Gene → Protein Expression Regression Model
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. YAPAY GEN EKSPRESYONU VERİSİ ÜRET
# ------------------------------------------------------------

np.random.seed(7)

num_samples = 150       # 150 hücre örneği
num_genes = 300         # 300 gene ait ekspresyon profili

samples = [f"Sample_{i}" for i in range(num_samples)]
genes = [f"Gene_{j}" for j in range(num_genes)]

# Gen ekspresyonu matrisi (0–20 arası rastgele değer)
expr = pd.DataFrame(
    np.random.rand(num_samples, num_genes) * 20,
    index=samples,
    columns=genes
)

# ------------------------------------------------------------
# 2. PROTEİN SEVİYESİ SİMÜLE ET
# ------------------------------------------------------------
# Bir protein seviyesinin 300 genin bazılarıyla biyolojik ilişkisi olduğunu varsayıyoruz.

# Gizli biyolojik ilişki: 10 gen proteini etkiliyor (bilinmiyor)
important_genes = np.random.choice(genes, size=10, replace=False)

# Protein seviyesi = seçilmiş genlerin ortalaması + gürültü
protein_level = (
    expr[important_genes].mean(axis=1)
    + np.random.normal(loc=0, scale=0.5, size=num_samples)
)

protein_df = pd.DataFrame({
    "Sample": samples,
    "Protein_Level": protein_level.values
}).set_index("Sample")

print("Expression matrix shape:", expr.shape)
print("Protein level shape:", protein_df.shape)

# ------------------------------------------------------------
# 3. MAKİNE ÖĞRENMESİ VERİ SETİ HAZIRLA
# ------------------------------------------------------------

X = expr
y = protein_df["Protein_Level"]

# Eğitim / test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 4. MODEL EĞİTİMİ (Gradient Boosting → güçlü bir regresör)
# ------------------------------------------------------------

gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    "n_estimators": [150, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4]
}

grid = GridSearchCV(
    gbr,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

# ------------------------------------------------------------
# 5. MODEL PERFORMANSI
# ------------------------------------------------------------

y_pred = best_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("Best Parameters:", grid.best_params_)
print(f"Test MSE: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

# ------------------------------------------------------------
# 6. ÖRNEK PROTEİN TAHMİNİ
# ------------------------------------------------------------
example = pd.DataFrame(X_test.iloc[:5])
example_scaled = scaler.transform(example)

example_preds = best_model.predict(example_scaled)

print("\n====== Example Predictions ======")
for i, pred in enumerate(example_preds, 1):
    print(f"Sample {i}: Predicted Protein Level = {pred:.3f}")
