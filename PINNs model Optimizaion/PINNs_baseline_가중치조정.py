import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Lambda, Add, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import CosineDecay
import optuna
# ---------------------------------------------------------
# 0. 난수(Random Seed) 고정
# ---------------------------------------------------------
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
print(f"Random Seed가 {seed_value}로 고정되었습니다.")

# ---------------------------------------------------------
# 1. 데이터 불러오기 및 전처리
# ---------------------------------------------------------
file_path = '원소열처리326ai.xlsx' 
if not os.path.exists(file_path):
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

df = pd.read_excel(file_path)

if df.empty:
    raise ValueError("데이터를 불러오지 못했습니다.")

# [설정] 열처리 데이터 포함 (Index 1~21)
X = df.iloc[:, 1:22].values 
y_target = df.iloc[:, 22:26].values

# 물리 변수 추출
try:
    TS_raw = df['TS'].values.reshape(-1, 1)
    HB_raw = df['HB'].values.reshape(-1, 1)
    E_raw  = df['E'].values.reshape(-1, 1)
except KeyError:
    print("헤더 이름 참조 실패. 인덱스로 할당합니다.")
    HB_raw = df.iloc[:, 19].values.reshape(-1, 1)
    TS_raw = df.iloc[:, 20].values.reshape(-1, 1)
    E_raw  = df.iloc[:, 21].values.reshape(-1, 1)

# 스케일링
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_target_scaled = scaler_y.fit_transform(y_target)

y_mean = tf.constant(scaler_y.mean_, dtype=tf.float32)
y_std = tf.constant(scaler_y.scale_, dtype=tf.float32)

y_combined = np.hstack([y_target_scaled, TS_raw, HB_raw, E_raw])

class DynamicPhysicsWeight(Callback):
    def __init__(self, initial_pw=1e-5, final_pw=1e-4, total_epochs=500):
        super(DynamicPhysicsWeight, self).__init__()
        self.initial_pw = initial_pw
        self.final_pw = final_pw
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_pw = self.initial_pw + (self.final_pw - self.initial_pw) * (epoch / self.total_epochs)
        global_physics_weight.assign(new_pw)
        if epoch % 10 == 0:
            print(f" - 현재 Physics Weight: {new_pw:.2e}")

global_physics_weight = tf.Variable(1e-5, dtype=tf.float32)

# ---------------------------------------------------------
# 2. PINN Custom Loss Function (4개 가중치 개별 조절)
# ---------------------------------------------------------
def pinn_loss_formulas(y_mean, y_std, 
                       w_sigma=1.0, w_b=1.0, w_ef=1.0, w_c=1.0):
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

def custom_mae(y_true, y_pred):
    y_true_target = y_true[:, :4]
    return tf.reduce_mean(tf.abs(y_true_target - y_pred))

# ---------------------------------------------------------
# 3. 모델 생성 함수
# ---------------------------------------------------------

def create_model(input_dim): # 8 Wide-Backbone
   
    weight_sigma = 0.41
    weight_b     = 0.49
    weight_ef    = 1.06
    weight_c     = 0.11

    l2_val = 0
    inputs = Input(shape=(input_dim,))

    # --- [Block 1: 512층] ---
    x = Dense(512, kernel_regularizer=l2(l2_val))(inputs) # activation 삭제
    x = LeakyReLU(alpha=0.01)(x) # 별도 층으로 추가
    x = Dense(512, kernel_regularizer=l2(l2_val))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.1)(x) 
    
    # --- [Block 2: 256층] ---
    x = Dense(256, kernel_regularizer=l2(l2_val))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.05)(x)
    
    # --- [Block 3: 정밀층] ---
    x = Dense(128, kernel_regularizer=l2(l2_val))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_val))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(16)(x) # 16노드 층
    x = LeakyReLU(alpha=0.01)(x)
    
    # 2. 출력층 (여기는 절대 건드리지 마세요! linear 유지)
    outputs = Dense(4, activation='linear')(x)

    lr_schedule = CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=7000, # 학습 끝까지 서서히 줄어들게 설정
        alpha=1e-6 # 최종 학습률의 하한선 (0까지 도달 가능)
    )
    model = Model(inputs=inputs, outputs=outputs)

    # optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
    # model.compile(
    #     optimizer=optimizer, 
    #     loss=pinn_loss_formulas(y_mean, y_std), 
    #     metrics=[custom_mae]
    # )
    # return model

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule, clipnorm=1.0), 
        loss=pinn_loss_formulas(y_mean, y_std, 
                                          w_sigma=weight_sigma, 
                                          w_b=weight_b, 
                                          w_ef=weight_ef, 
                                          w_c=weight_c), 
        metrics=[custom_mae]
    )
    return model

# ---------------------------------------------------------
# 4. K-Fold Cross Validation
# ---------------------------------------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

final_preds_sum = np.zeros((X_scaled.shape[0], 4))

print(f"[{n_splits}-Fold] K-Fold 교차 검증 학습을 시작합니다... (가중치 개별 적용)")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_combined)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    X_train_k, X_val_k = X_scaled[train_idx], X_scaled[val_idx]
    y_train_k, y_val_k = y_combined[train_idx], y_combined[val_idx]
    
    model = create_model(X_scaled.shape[1])
    
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

output_filename = 'PINN_Prediction_Weighted_5.xlsx'
result_df.to_excel(output_filename, index=False)

print(f"작업 완료! 결과 파일: {output_filename}")