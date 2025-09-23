# full_pipeline_save.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("crop_yield_dataset.csv")
df = df.drop(columns=['Latitude', 'Longitude'])

# Ensure all dummy columns are present
model_columns = ["Year", "LandSize(ha)", "FertilizerUsage(kg_ha)", "PesticideUsage(kg_ha)",
                 "AvgTemperature(C)", "AnnualRainfall(mm)", "State_Haryana", "State_Maharashtra",
                 "State_Punjab", "State_Uttar Pradesh", "State_West Bengal",
                 "CropType_Maize", "CropType_Rice", "CropType_Wheat"]

df = pd.get_dummies(df, columns=['State', 'CropType'], drop_first=True)
for col in model_columns:
    if col not in df.columns:
        df[col] = 0

X_raw = df[model_columns].astype(float).values
y_raw = df['Yield(kg_ha)'].astype(float).values

# -------------------------
# Train/test split
# -------------------------
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
)

# -------------------------
# Feature scaling -> polynomial -> final scaling
# -------------------------
scaler_X1 = StandardScaler()
X_train_s1 = scaler_X1.fit_transform(X_train_raw)
X_test_s1 = scaler_X1.transform(X_test_raw)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_s1)
X_test_poly = poly.transform(X_test_s1)

scaler_X2 = StandardScaler()
X_train_all = scaler_X2.fit_transform(X_train_poly)
X_test_all = scaler_X2.transform(X_test_poly)

X_train_sel = X_train_all
X_test_sel = X_test_all

# -------------------------
# Target transform pipeline
# -------------------------
y_train_log = np.log1p(y_train_raw)
y_test_log = np.log1p(y_test_raw)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1,1)).ravel()
y_test_scaled = scaler_y.transform(y_test_log.reshape(-1,1)).ravel()

def raw_to_scaled_logy(y_raw_array):
    return scaler_y.transform(np.log1p(y_raw_array).reshape(-1,1)).ravel()

def scaled_logy_to_raw(y_scaled_array):
    inv = scaler_y.inverse_transform(y_scaled_array.reshape(-1,1)).ravel()
    return np.expm1(inv)

# -------------------------
# ModifiedSVR
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
            dL_df = np.where(t <= 0, 0.0, np.where(t <= self.delta, -(t/self.delta)*np.sign(resid), -np.sign(resid)))
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

            # validation
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
# Stacking (OOF predictions)
# -------------------------
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
n_train = X_train_sel.shape[0]
n_test = X_test_sel.shape[0]
oof_train = np.zeros((n_train, 4))
oof_test = np.zeros((n_test, 4))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_sel), 1):
    print(f"Stack fold {fold}/{n_folds} ...")
    X_tr, X_val = X_train_sel[train_idx], X_train_sel[val_idx]
    y_tr_scaled, y_val_scaled = y_train_scaled[train_idx], y_train_scaled[val_idx]
    y_tr_raw, y_val_raw = y_train_raw[train_idx], y_train_raw[val_idx]

    # Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_tr, y_tr_scaled)
    oof_train[val_idx,0] = ridge.predict(X_val)
    oof_test[:,0] += ridge.predict(X_test_sel)/n_folds

    # RandomForest
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_tr, y_tr_raw)
    oof_train[val_idx,1] = raw_to_scaled_logy(rf.predict(X_val))
    oof_test[:,1] += raw_to_scaled_logy(rf.predict(X_test_sel))/n_folds

    # GradientBoosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    gb.fit(X_tr, y_tr_raw)
    oof_train[val_idx,2] = raw_to_scaled_logy(gb.predict(X_val))
    oof_test[:,2] += raw_to_scaled_logy(gb.predict(X_test_sel))/n_folds

    # ModifiedSVR
    svr = ModifiedSVR(C=5.0, base_epsilon=0.05, delta=0.02, lr=0.005, n_iter=3000,
                      batch_size=max(8, int(X_tr.shape[0]/4)), random_state=42,
                      early_stopping=True, stop_patience=400)
    svr.w = ridge.coef_.copy()
    svr.b = ridge.intercept_.copy()
    svr.fit(X_tr, y_tr_scaled, X_val=X_val, y_val=y_val_scaled)
    oof_train[val_idx,3] = svr.predict(X_val)
    oof_test[:,3] += svr.predict(X_test_sel)/n_folds

# Meta-learner
meta = Ridge(alpha=1.0, random_state=42)
meta.fit(oof_train, y_train_scaled)

# -------------------------
# Save full stacking pipeline objects
# -------------------------
stacked_pipeline_objects = {
    "model_columns": model_columns,
    "scaler_X1": scaler_X1,
    "poly": poly,
    "scaler_X2": scaler_X2,
    "scaler_y": scaler_y,
    "meta_model": meta,
    "ridge": ridge,
    "rf": rf,
    "gb": gb,
    "svr": svr
}
joblib.dump(stacked_pipeline_objects, "full_stacked_pipeline.pkl")
print("Saved full_stacked_pipeline.pkl with all base models, meta-model, and preprocessing.")
