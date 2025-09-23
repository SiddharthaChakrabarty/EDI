# svr_stack_fixed_trees.py
"""
Stacking pipeline with corrected handling of tree-model targets.

- Preprocessing: one-hot -> scale -> polynomial -> scale
- Target: log1p(y) -> StandardScaler => 'scaled log1p' used for SVR/Ridge and meta
- RandomForest / GradientBoosting: trained on raw y (no scaling), but their raw preds
  are converted to 'scaled log1p' before stacking (so meta sees consistent units).
- ModifiedSVR: trained on scaled log1p target (same as Ridge).
- Final predictions are inverse-transformed back to original (kg/ha) before evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# -------------------------
# Load & preprocess
# -------------------------
df = pd.read_csv("crop_yield_dataset.csv")
df = df.drop(columns=['Latitude', 'Longitude'])
df = pd.get_dummies(df, columns=['State', 'CropType'], drop_first=True)

feature_cols = [c for c in df.columns if c != 'Yield(kg_ha)']
X_raw = df[feature_cols].astype(float).values
y_raw = df['Yield(kg_ha)'].values.astype(float)

# Train/test split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
)

# Feature scaling -> polynomial -> final scaling
scaler_X1 = StandardScaler()
X_train_s1 = scaler_X1.fit_transform(X_train_raw)
X_test_s1 = scaler_X1.transform(X_test_raw)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_s1)
X_test_poly = poly.transform(X_test_s1)

scaler_X2 = StandardScaler()
X_train_all = scaler_X2.fit_transform(X_train_poly)
X_test_all = scaler_X2.transform(X_test_poly)

# (Optional) feature selection - here we keep all features for simplicity,
# but you can select top-K as before if desired:
X_train_sel = X_train_all
X_test_sel = X_test_all

# -------------------------
# Target transform pipeline (same convention used throughout for meta features)
# We'll use log1p(y) then standardize => "scaled_logy"
# -------------------------
y_train_log = np.log1p(y_train_raw)   # raw -> log1p
y_test_log = np.log1p(y_test_raw)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1,1)).ravel()
y_test_scaled = scaler_y.transform(y_test_log.reshape(-1,1)).ravel()

# Helper transforms (raw <-> scaled_logy)
def raw_to_scaled_logy(y_raw_array):
    """raw -> log1p -> standardize"""
    return scaler_y.transform(np.log1p(y_raw_array).reshape(-1,1)).ravel()

def scaled_logy_to_raw(y_scaled_array):
    """scaled_logy -> inverse standardize -> expm1 -> raw"""
    inv = scaler_y.inverse_transform(y_scaled_array.reshape(-1,1)).ravel()
    return np.expm1(inv)

# -------------------------
# Compact ModifiedSVR (primal) — same gradient math used previously
# -------------------------
class ModifiedSVR:
    def __init__(self, C=1.0, base_epsilon=0.05, delta=0.02, lambda1=0.0, lr=0.005,
                 n_iter=3000, batch_size=16, random_state=0, early_stopping=False, stop_patience=500):
        self.C = C
        self.base_epsilon = base_epsilon
        self.delta = delta
        self.lambda1 = lambda1
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)
        self.w = None
        self.b = 0.0
        self.early_stopping = early_stopping
        self.stop_patience = stop_patience

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        if self.w is None:
            self.w = np.zeros(n_features)
        # Adam state
        m = np.zeros_like(self.w); v = np.zeros_like(self.w)
        mb = 0.0; vb = 0.0
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        best_val_loss = np.inf; best_wb = None; patience = 0

        for it in range(1, self.n_iter + 1):
            idx = self.rng.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
            Xb = X[idx]; yb = y[idx]
            preds = Xb.dot(self.w) + self.b
            resid = yb - preds
            t = np.abs(resid) - self.base_epsilon
            dL_df = np.zeros_like(resid)
            for i in range(len(resid)):
                if t[i] <= 0:
                    dL_df[i] = 0.0
                elif t[i] <= self.delta:
                    dL_df[i] = - (t[i] / self.delta) * np.sign(resid[i])
                else:
                    dL_df[i] = -1.0 * np.sign(resid[i])
            grad_w = self.w.copy()
            if self.lambda1 != 0.0:
                grad_w += self.lambda1 * (self.w / (np.sqrt(self.w**2 + 1e-8)))
            grad_w += self.C * (dL_df[:,None] * Xb).sum(axis=0)
            grad_b = self.C * dL_df.sum()

            # Adam updates
            m = beta1*m + (1-beta1)*grad_w
            v = beta2*v + (1-beta2)*(grad_w**2)
            m_hat = m / (1 - beta1**it)
            v_hat = v / (1 - beta2**it)
            self.w -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            mb = beta1*mb + (1-beta1)*grad_b
            vb = beta2*vb + (1-beta2)*(grad_b**2)
            mb_hat = mb / (1 - beta1**it)
            vb_hat = vb / (1 - beta2**it)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps_adam)

            # optional val monitoring
            if (it % 200 == 0 or it == 1) and (X_val is not None and y_val is not None):
                vpred = X_val.dot(self.w) + self.b
                vr = y_val - vpred
                vt = np.abs(vr) - self.base_epsilon
                vloss_terms = np.where(vt <= 0, 0.0, np.where(vt <= self.delta, vt**2/(2*self.delta), vt - self.delta/2))
                val_loss = 0.5*np.sum(self.w**2) + self.lambda1*np.sum(np.sqrt(self.w**2 + 1e-8)) + self.C*np.sum(vloss_terms)
                if val_loss < best_val_loss - 1e-9:
                    best_val_loss = val_loss
                    best_wb = (self.w.copy(), self.b)
                    patience = 0
                else:
                    patience += 200
                if self.early_stopping and patience >= self.stop_patience:
                    break

        if best_wb is not None:
            self.w, self.b = best_wb

    def predict(self, X):
        return X.dot(self.w) + self.b

# -------------------------
# Stacking: OOF predictions (make sure all base model outputs are in scaled_logy space)
# We will create OOF predictions for:
# 0: Ridge (trained on scaled_logy)
# 1: RandomForest (trained on raw y; convert predictions to scaled_logy)
# 2: GradientBoosting (trained on raw y; convert -> scaled_logy)
# 3: ModifiedSVR (trained on scaled_logy)
# -------------------------
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
n_train = X_train_sel.shape[0]
n_test = X_test_sel.shape[0]
oof_train = np.zeros((n_train, 4))
oof_test = np.zeros((n_test, 4))

fold = 0
for train_idx, val_idx in kf.split(X_train_sel):
    print(f"Stack fold {fold+1}/{n_folds} ...")
    X_tr, X_val = X_train_sel[train_idx], X_train_sel[val_idx]
    # Note: y_train_scaled is scaled log1p(y). For trees we still keep raw labels.
    y_tr_scaled = y_train_scaled[train_idx]
    y_val_scaled = y_train_scaled[val_idx]
    y_tr_raw = y_train_raw[train_idx]
    y_val_raw = y_train_raw[val_idx]

    # 0) Ridge on scaled logy
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_tr, y_tr_scaled)
    oof_train[val_idx, 0] = ridge.predict(X_val)
    oof_test[:, 0] += ridge.predict(X_test_sel) / n_folds

    # 1) RandomForest trained on RAW y (important)
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_tr, y_tr_raw)  # raw target
    rf_val_raw = rf.predict(X_val)
    rf_test_raw = rf.predict(X_test_sel)
    # convert rf predictions to scaled_logy space
    rf_val_scaled = raw_to_scaled_logy(rf_val_raw)
    rf_test_scaled = raw_to_scaled_logy(rf_test_raw)
    oof_train[val_idx, 1] = rf_val_scaled
    oof_test[:, 1] += rf_test_scaled / n_folds

    # 2) GradientBoosting trained on RAW y
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    gb.fit(X_tr, y_tr_raw)
    gb_val_raw = gb.predict(X_val)
    gb_test_raw = gb.predict(X_test_sel)
    gb_val_scaled = raw_to_scaled_logy(gb_val_raw)
    gb_test_scaled = raw_to_scaled_logy(gb_test_raw)
    oof_train[val_idx, 2] = gb_val_scaled
    oof_test[:, 2] += gb_test_scaled / n_folds

    # 3) ModifiedSVR trained on scaled_logy
    # warm-start with ridge weights
    svr = ModifiedSVR(C=5.0, base_epsilon=0.05, delta=0.02, lr=0.005, n_iter=3000, batch_size=max(8, int(X_tr.shape[0]/4)), random_state=42, early_stopping=True, stop_patience=400)
    svr.w = ridge.coef_.copy()
    svr.b = ridge.intercept_.copy()
    svr.fit(X_tr, y_tr_scaled, X_val=X_val, y_val=y_val_scaled)
    oof_train[val_idx, 3] = svr.predict(X_val)
    oof_test[:, 3] += svr.predict(X_test_sel) / n_folds

    fold += 1

# OOFs built (all columns are in scaled_logy units). Train meta-learner (Ridge) on oof_train -> y_train_scaled
meta = Ridge(alpha=1.0, random_state=42)
meta.fit(oof_train, y_train_scaled)

# Predict on test: meta.predict(oof_test) returns scaled_logy predictions -> inverse to raw
stack_test_scaled = meta.predict(oof_test)
stack_test_preds_raw = scaled_logy_to_raw(stack_test_scaled)

# Also train final base models on full training set (to inspect individually)
# Ridge final (scaled target)
ridge_full = Ridge(alpha=1.0, random_state=42)
ridge_full.fit(X_train_sel, y_train_scaled)
ridge_test_scaled = ridge_full.predict(X_test_sel)
ridge_test_raw = scaled_logy_to_raw(ridge_test_scaled)

# RF final (raw target)
rf_full = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf_full.fit(X_train_sel, y_train_raw)
rf_test_raw = rf_full.predict(X_test_sel)

# GB final (raw)
gb_full = GradientBoostingRegressor(n_estimators=400, max_depth=4, random_state=42)
gb_full.fit(X_train_sel, y_train_raw)
gb_test_raw = gb_full.predict(X_test_sel)

# SVR final (scaled target) - warm start with ridge_full coeffs
svr_final = ModifiedSVR(C=5.0, base_epsilon=0.05, delta=0.02, lr=0.005, n_iter=6000, batch_size=max(8,int(X_train_sel.shape[0]/4)), random_state=42, early_stopping=True, stop_patience=800)
svr_final.w = ridge_full.coef_.copy()
svr_final.b = ridge_full.intercept_.copy()
svr_final.fit(X_train_sel, y_train_scaled, X_val=X_train_sel, y_val=y_train_scaled)
svr_test_scaled = svr_final.predict(X_test_sel)
svr_test_raw = scaled_logy_to_raw(svr_test_scaled)

# -------------------------
# Evaluation helpers
# -------------------------
def print_metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0
    print(f"{label}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}, MAPE={mape:.2f}%")

print("\n--- Final test set results ---")
print_metrics(y_test_raw, svr_test_raw, "ModifiedSVR")
print_metrics(y_test_raw, ridge_test_raw, "Ridge")
print_metrics(y_test_raw, rf_test_raw, "RandomForest (raw)")
print_metrics(y_test_raw, gb_test_raw, "GradientBoosting (raw)")
print_metrics(y_test_raw, stack_test_preds_raw, "Stacked Ensemble (meta Ridge)")

# Save predictions
out = pd.DataFrame({
    "Actual": y_test_raw,
    "SVR": svr_test_raw,
    "Ridge": ridge_test_raw,
    "RF": rf_test_raw,
    "GB": gb_test_raw,
    "Stack": stack_test_preds_raw
})
out.to_csv("stacked_fixedtrees_preds.csv", index=False)
print("\nSaved stacked_fixedtrees_preds.csv")


# -------------------------
# Extended evaluation: accuracy-like metrics
# -------------------------
def accuracy_like(y_true, y_pred):
    """1 - (MAE / mean(actual))"""
    return 1.0 - (np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-8))

def fraction_within_tolerance(y_true, y_pred, tol_percent=5.0):
    """Fraction of predictions within ±tol_percent of actual"""
    tol = tol_percent / 100.0
    frac = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8) <= tol)
    return frac * 100.0

models = {
    "ModifiedSVR": svr_test_raw,
    "Ridge": ridge_test_raw,
    "RandomForest": rf_test_raw,
    "GradientBoosting": gb_test_raw,
    "Stacked Ensemble": stack_test_preds_raw
}

print("\n--- Test set metrics including accuracy ---")
for name, preds in models.items():
    rmse_val = math.sqrt(mean_squared_error(y_test_raw, preds))
    mae_val = mean_absolute_error(y_test_raw, preds)
    r2_val = r2_score(y_test_raw, preds)
    mape_val = np.mean(np.abs((y_test_raw - preds)/(y_test_raw + 1e-8))) * 100
    acc_val = accuracy_like(y_test_raw, preds)
    frac_5 = fraction_within_tolerance(y_test_raw, preds, 5.0)
    frac_10 = fraction_within_tolerance(y_test_raw, preds, 10.0)
    print(f"\n{name}:")
    print(f" RMSE : {rmse_val:.3f} kg/ha")
    print(f" MAE  : {mae_val:.3f} kg/ha")
    print(f" R2   : {r2_val:.3f}")
    print(f" MAPE : {mape_val:.2f} %")
    print(f" Accuracy-like (1 - MAE/mean) : {acc_val*100:.2f} %")
    print(f" Fraction within ±5%  : {frac_5:.2f} %")
    print(f" Fraction within ±10% : {frac_10:.2f} %")
