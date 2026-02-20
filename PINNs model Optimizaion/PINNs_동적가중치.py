import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import optuna
from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import CosineDecay

# ---------------------------------------------------------
# 0. 설정 및 데이터 로드 (기존 로직 유지)
# ---------------------------------------------------------
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

file_path = '원소열처리326ai.xlsx'
df = pd.read_excel(file_path)

X = df.iloc[:, 1:22].values 
y_target = df.iloc[:, 22:26].values

try:
    TS_raw = df['TS'].values.reshape(-1, 1)
    HB_raw = df['HB'].values.reshape(-1, 1)
    E_raw  = df['E'].values.reshape(-1, 1)
except KeyError:
    HB_raw = df.iloc[:, 19].values.reshape(-1, 1)
    TS_raw = df.iloc[:, 20].values.reshape(-1, 1)
    E_raw  = df.iloc[:, 21].values.reshape(-1, 1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_target_scaled = scaler_y.fit_transform(y_target)

y_mean = tf.constant(scaler_y.mean_, dtype=tf.float32)
y_std = tf.constant(scaler_y.scale_, dtype=tf.float32)
y_combined = np.hstack([y_target_scaled, TS_raw, HB_raw, E_raw])

global_physics_weight = tf.Variable(1e-5, dtype=tf.float32)

class DynamicPhysicsWeight(Callback):
    def __init__(self, initial_pw=1e-5, final_pw=5e-4, total_epochs=500):
        super().__init__()
        self.initial_pw, self.final_pw, self.total_epochs = initial_pw, final_pw, total_epochs
    def on_epoch_begin(self, epoch, logs=None):
        new_pw = self.initial_pw + (self.final_pw - self.initial_pw) * (epoch / self.total_epochs)
        global_physics_weight.assign(new_pw)

# ---------------------------------------------------------
# 1. Custom Loss & Model (가중치 가변형)
# ---------------------------------------------------------
def pinn_loss_formulas(y_mean, y_std, 
                       w_sigma, w_b, w_ef, w_c):
    def loss(y_true, y_pred):
        y_actual_scaled = y_true[:, :4] 
        ts = y_true[:, 4:5]
        hb = y_true[:, 5:6]
        e_mod = y_true[:, 6:7]
        e_mod_mpa = e_mod * 1000.0 + 1e-6
        ratio = ts / (hb + 1e-6)

        y_pred_original = (y_pred * y_std) + y_mean
        pred_sigma_f = y_pred_original[:, 0:1]
        pred_b       = y_pred_original[:, 1:2]
        pred_e_f     = y_pred_original[:, 2:3]
        pred_c       = y_pred_original[:, 3:4]

        phy_sigma_f = tf.zeros_like(pred_sigma_f)
        phy_b       = tf.zeros_like(pred_b)
        phy_e_f     = tf.zeros_like(pred_e_f)
        phy_c       = tf.zeros_like(pred_c)

        # --- 경험식 조건문 ---
        # [Group 1] Su < 802
        mask_g1 = (ts < 802)
        cond_1_1 = mask_g1 & (ratio > 3.66)
        phy_sigma_f = tf.where(cond_1_1, 1.22 * ts + 553.29, phy_sigma_f)
        phy_b       = tf.where(cond_1_1, -0.132, phy_b)
        ef_val_1_1  = (1.12 * tf.square(ts) - 1377.0 * ts + 499788.0) / e_mod_mpa
        phy_e_f     = tf.where(cond_1_1, ef_val_1_1, phy_e_f)
        phy_c       = tf.where(cond_1_1, -0.543, phy_c)

        cond_1_2 = mask_g1 & (ratio <= 3.66)
        phy_sigma_f = tf.where(cond_1_2, 0.94 * ts + 460.38, phy_sigma_f)
        phy_b       = tf.where(cond_1_2, -0.160, phy_b)
        ef_val_1_2  = (-0.06 * tf.square(ts) + 154.0 * ts + 19790.0) / e_mod_mpa
        phy_e_f     = tf.where(cond_1_2, ef_val_1_2, phy_e_f)
        phy_c       = tf.where(cond_1_2, -0.496, phy_c)

        # [Group 2] 802 <= Su < 1238
        mask_g2 = (ts >= 802) & (ts < 1238)
        cond_2_1 = mask_g2 & (ratio > 3.66)
        phy_sigma_f = tf.where(cond_2_1, 1.95 * ts - 515.52, phy_sigma_f)
        phy_b       = tf.where(cond_2_1, -0.134, phy_b)
        ef_val_2_1  = (-2.002 * tf.square(ts) + 4071.0 * ts - 1927507.0) / e_mod_mpa
        phy_e_f     = tf.where(cond_2_1, ef_val_2_1, phy_e_f)
        phy_c       = tf.where(cond_2_1, -0.510, phy_c)

        cond_2_2 = mask_g2 & (ratio <= 3.66)
        phy_sigma_f = tf.where(cond_2_2, 1.09 * ts + 261.82, phy_sigma_f)
        phy_b       = tf.where(cond_2_2, -0.092, phy_b)
        ef_val_2_2  = (-0.4712 * tf.square(ts) + 881.0 * ts - 288495.0) / e_mod_mpa
        phy_e_f     = tf.where(cond_2_2, ef_val_2_2, phy_e_f)
        phy_c       = tf.where(cond_2_2, -0.536, phy_c)

        # [Group 3] Su >= 1238
        mask_g3 = (ts >= 1238)
        cond_3_1 = mask_g3 & (ratio > 3.66)
        phy_sigma_f = tf.where(cond_3_1, 1.11 * ts + 444.14, phy_sigma_f)
        phy_b       = tf.where(cond_3_1, -0.101, phy_b)
        ef_val_3_1  = (0.1242 * tf.square(ts) - 557.0 * ts + 684976.0) / e_mod_mpa
        phy_e_f     = tf.where(cond_3_1, ef_val_3_1, phy_e_f)
        phy_c       = tf.where(cond_3_1, -0.633, phy_c)

        cond_3_2 = mask_g3 & (ratio <= 3.66)
        phy_sigma_f = tf.where(cond_3_2, 1.30 * ts + 74.56, phy_sigma_f)
        phy_b       = tf.where(cond_3_2, -0.118, phy_b)
        ef_val_3_2  = (-0.06 * tf.square(ts) + 136.0 * ts + 53575.0) / e_mod_mpa
        phy_e_f     = tf.where(cond_3_2, ef_val_3_2, phy_e_f)
        phy_c       = tf.where(cond_3_2, -0.578, phy_c)

        # --- Loss 계산 (4개 개별 가중치 적용) ---
        loss_data = tf.reduce_mean(tf.square(y_actual_scaled - y_pred))
        
        loss_phy_sigma = tf.reduce_mean(tf.square(pred_sigma_f - phy_sigma_f))
        loss_phy_b     = tf.reduce_mean(tf.square(pred_b - phy_b))
        loss_phy_ef    = tf.reduce_mean(tf.square(pred_e_f - phy_e_f))
        loss_phy_c     = tf.reduce_mean(tf.square(pred_c - phy_c))
        
        # [수정 포인트] 개별 가중치 곱하기
        loss_physics = ((w_sigma * loss_phy_sigma) + 
                       (w_b     * loss_phy_b) + 
                       (w_ef    * loss_phy_ef) + 
                       (w_c     * loss_phy_c))
                       
        return loss_data + (global_physics_weight * loss_physics)

    return loss

def create_model(input_dim, weights):
    inputs = Input(shape=(input_dim,))
    x = Dense(512)(inputs); x = LeakyReLU(alpha=0.01)(x)
    x = Dense(512)(x); x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.1)(x)
    x = Dense(256)(x); x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.05)(x)
    x = Dense(128)(x); x = LeakyReLU(alpha=0.01)(x)
    x = Dense(64, activation='relu')(x); x = LeakyReLU(alpha=0.01)(x)
    x = Dense(16)(x); x = LeakyReLU(alpha=0.01)(x)
    outputs = Dense(4, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    lr_sch = CosineDecay(3e-4, 7000, alpha=1e-6)
    model.compile(optimizer=Adam(learning_rate=lr_sch, clipnorm=1.0),
                  loss=pinn_loss_formulas(y_mean, y_std, **weights)
                 )
    return model

# ---------------------------------------------------------
# 2. Optuna 탐색 단계 (3-Fold)
# ---------------------------------------------------------
# def objective(trial):
#     w_params = {
#         'w_sigma': trial.suggest_float('w_sigma', 0.1, 1.2),
#         'w_ef': trial.suggest_float('w_ef', 0.1, 1.5),
#         'w_b': trial.suggest_float('w_b', 0.01, 0.5),
#         'w_c': trial.suggest_float('w_c', 0.01, 0.5)
#     }
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     errors = []
#     for t_idx, v_idx in kf.split(X_scaled):
#         model = create_model(X_scaled.shape[1], w_params)
#         model.fit(X_scaled[t_idx], y_combined[t_idx], epochs=300, batch_size=16, verbose=0)
        
#         pred = scaler_y.inverse_transform(model.predict(X_scaled[v_idx]))
        
#         # --- [추가] 여기서도 음수 보정을 수행해야 정확한 성능 평가가 됩니다 ---
#         ts_v = TS_raw[v_idx].flatten()
#         hb_v = HB_raw[v_idx].flatten()
#         ratio_v = ts_v / (hb_v + 1e-6)
#         ef_p = pred[:, 2]
#         neg_v = ef_p < 0
        
#         # ... (생략된 그룹별 중앙값 보정 코드들을 여기에 다 넣으세요) ...
#         mask_g1_v = (ts_v < 802)
#         pred[mask_g1_v & (ratio_v > 3.66) & neg_v, 2] = 0.416
#         pred[mask_g1_v & (ratio_v <= 3.66) & neg_v, 2] = 0.4085
        
#         # Group 2
#         mask_g2_v = (ts_v >= 802) & (ts_v < 1238)
#         pred[mask_g2_v & (ratio_v > 3.66) & neg_v, 2] = 0.4599
#         pred[mask_g2_v & (ratio_v <= 3.66) & neg_v, 2] = 0.6665
        
#         # Group 3
#         mask_g3_v = (ts_v >= 1238)
#         pred[mask_g3_v & (ratio_v > 3.66) & neg_v, 2] = 0.72
#         pred[mask_g3_v & (ratio_v <= 3.66) & neg_v, 2] = 0.595

#         # 보정된 pred로 에러 계산
#         true_val = scaler_y.inverse_transform(y_combined[v_idx, :4])
#         row_mae = np.mean(np.abs(true_val - pred), axis=1)
#         errors.append(np.sum(row_mae > 1.0))
        
#     return np.mean(errors)

def objective(trial):
    w_params = {
        'w_sigma': trial.suggest_float('w_sigma', 0.45, 0.75),
        'w_ef': trial.suggest_float('w_ef', 0.4, 0.6),
        'w_b': trial.suggest_float('w_b', 0.05, 0.25),
        'w_c': trial.suggest_float('w_c', 0.25, 0.45)
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = [] # 여기에 값이 쌓여야 합니다.
    
    for t_idx, v_idx in kf.split(X_scaled):
        try:
            model = create_model(X_scaled.shape[1], w_params)
            # 학습 (verbose=0으로 조용히)
            model.fit(X_scaled[t_idx], y_combined[t_idx], epochs=200, batch_size=16, verbose=0)
            
            # 예측
            raw_pred = model.predict(X_scaled[v_idx], verbose=0)
            pred = scaler_y.inverse_transform(raw_pred)
            
            # --- [음수 보정 로직] ---
            ts_v = TS_raw[v_idx].flatten()
            hb_v = HB_raw[v_idx].flatten()
            ratio_v = ts_v / (hb_v + 1e-6)
            neg_v = pred[:, 2] < 0
            
            mask_g1_v = (ts_v < 802)
            mask_g2_v = (ts_v >= 802) & (ts_v < 1238)
            mask_g3_v = (ts_v >= 1238)

            pred[mask_g1_v & (ratio_v > 3.66) & neg_v, 2] = 0.416
            pred[mask_g1_v & (ratio_v <= 3.66) & neg_v, 2] = 0.4085
            pred[mask_g2_v & (ratio_v > 3.66) & neg_v, 2] = 0.4599
            pred[mask_g2_v & (ratio_v <= 3.66) & neg_v, 2] = 0.6665
            pred[mask_g3_v & (ratio_v > 3.66) & neg_v, 2] = 0.72
            pred[mask_g3_v & (ratio_v <= 3.66) & neg_v, 2] = 0.595

            # --- [스코어 계산] ---
            true_val_raw = scaler_y.inverse_transform(y_combined[v_idx, :4])
            
            # log10 안전하게 계산
            log_true = np.log10(np.maximum(true_val_raw, 1.0))
            log_pred = np.log10(np.maximum(pred, 1.0))
            
            row_log_mae = np.mean(np.abs(log_true - log_pred), axis=1)

            count_over_1_0 = np.sum(row_log_mae > 1.0)
            count_over_1_5 = np.sum(row_log_mae > 1.5)
            
            severe_indices = row_log_mae > 1.0
            severe_penalty = np.sum(row_log_mae[severe_indices]) if np.any(severe_indices) else 0.0

            fold_score = (
                (float(count_over_1_5) * 10.0) + 
                (float(count_over_1_0) * 40.0) + 
                (float(severe_penalty) * 2.0) + 
                (float(np.mean(row_log_mae)) * 10.0)
            )
            
            # [가장 중요] 리스트에 추가!!
            fold_scores.append(fold_score)
            
        except Exception as e:
            print(f"폴드 학습 중 에러 발생: {e}")
            fold_scores.append(1e5) # 에러 나면 큰 벌점 주고 다음 폴드로

    # 만약 리스트가 비어있다면 (최악의 경우)
    if not fold_scores:
        return 1e6
        
    return np.mean(fold_scores)

# [수정] 옵튜나 스터디 생성 부분
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=20,  # 처음 20번은 무지성(?)으로 넓게 던져봄 (탐색)
        multivariate=True     # 4개 변수 간의 상관관계를 고려해서 좁혀 들어감 (중요!)
    )
)
study.optimize(objective, n_trials=20) # 시도 횟수

# ---------------------------------------------------------
# 3. 최적 가중치로 최종 5-Fold 학습 & 결과 저장
# ---------------------------------------------------------
best_weights = study.best_params
print(f"최적 가중치 적용: {best_weights}")

# ... (이후 5-Fold 학습 및 엑셀 저장 로직 실행) ...
# ---------------------------------------------------------
# 4. K-Fold Cross Validation
# ---------------------------------------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

final_preds_sum = np.zeros((X_scaled.shape[0], 4))

print("="*50)
print("🚀 [FINAL SELECTION] Optuna Best Weights")
for param_name, param_value in best_weights.items():
    print(f" - {param_name}: {param_value:.6f}")
print("="*50)

print(f"[{n_splits}-Fold] K-Fold 교차 검증 학습을 시작합니다... (가중치 개별 적용)")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_combined)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    X_train_k, X_val_k = X_scaled[train_idx], X_scaled[val_idx]
    y_train_k, y_val_k = y_combined[train_idx], y_combined[val_idx]
    
    model = create_model(X_scaled.shape[1], best_weights)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=0)
    
    dynamic_pw_callback = DynamicPhysicsWeight(
    initial_pw=1e-5, 
    final_pw=5e-4, 
    total_epochs=500 # 전체 에포크 수에 맞춰 조절
    )

    history = model.fit(
        X_train_k, y_train_k,
        validation_data=(X_val_k, y_val_k),
        epochs=500,
        batch_size=16,
        callbacks=[early_stop, dynamic_pw_callback],
        verbose=0
    )
    
    pred_k = model.predict(X_scaled)
    final_preds_sum += pred_k
    
    best_val_loss = min(history.history['val_loss'])
    print(f"Fold {fold+1} 완료 | Best Val Loss: {best_val_loss:.6f}")

print("\n모든 Fold 학습 완료.")

# ---------------------------------------------------------
# 5. 결과 저장 (변수명 통일 수정 완료)
# ---------------------------------------------------------
# (A) 5개 모델 예측값 평균 계산 (앙상블)
y_pred_ensemble_scaled = final_preds_sum / n_splits

# [수정] 변수명을 y_pred_ensemble_original로 통일
y_pred_ensemble_original = scaler_y.inverse_transform(y_pred_ensemble_scaled)

# (B) e_f 음수값 보정 로직 적용
ef_pred = y_pred_ensemble_original[:, 2]

# 그룹핑을 위한 TS, Ratio 정보 준비 (전체 데이터 기준)
ts_all = TS_raw.flatten()
hb_all = HB_raw.flatten()
ratio_all = ts_all / (hb_all + 1e-6)
neg_mask = ef_pred < 0

# --- 그룹별 중앙값 대체 ---
mask_g1 = (ts_all < 802)
y_pred_ensemble_original[mask_g1 & (ratio_all > 3.66) & neg_mask, 2] = 0.416
y_pred_ensemble_original[mask_g1 & (ratio_all <= 3.66) & neg_mask, 2] = 0.4085

mask_g2 = (ts_all >= 802) & (ts_all < 1238)
y_pred_ensemble_original[mask_g2 & (ratio_all > 3.66) & neg_mask, 2] = 0.4599
y_pred_ensemble_original[mask_g2 & (ratio_all <= 3.66) & neg_mask, 2] = 0.6665

mask_g3 = (ts_all >= 1238)
y_pred_ensemble_original[mask_g3 & (ratio_all > 3.66) & neg_mask, 2] = 0.72
y_pred_ensemble_original[mask_g3 & (ratio_all <= 3.66) & neg_mask, 2] = 0.595

# (C) 엑셀 저장
result_df = df.copy()
target_names = ['sigma_f', 'b', 'e_f', 'c']

for i, name in enumerate(target_names):
    result_df[f'{name}_Pred'] = y_pred_ensemble_original[:, i]

output_filename = 'PINN_Prediction_Weighted_16.xlsx'
result_df.to_excel(output_filename, index=False)

print(f"작업 완료! 결과 파일: {output_filename}")