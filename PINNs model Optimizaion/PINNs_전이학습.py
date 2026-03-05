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
import gc
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

CURRENT_TRY = 0
PREVIOUS_TRY = None  # 불러올 이전 Try 번호
LOAD_PREVIOUS = False

weight_sigma = 0.461813
weight_ef    = 0.458375
weight_b     = 0.186726
weight_c     = 0.267191

save_path = f"TL1_best_model_try{CURRENT_TRY}.weights.h5"
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
    def __init__(self, initial_pw=1e-5, final_pw=5e-3, total_epochs=500):
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

def create_model(input_dim, freeze_count=0, is_try_zero=False): # 8 Wide-Backbone
    # --- 학습률 결정 로직 ---
    if is_try_zero:
        current_lr = 3e-4  # Try 0: 처음 배울 땐 에너지를 높게! (대각선을 그리기 위해)
    elif freeze_count >= 4:
        current_lr = 5e-5  # Try 1: 앞단 잠그고 뒷단만 배울 때
    elif freeze_count >= 2:
        current_lr = 1e-5  # Try 2~3: 조금씩 더 풀 때
    else:
        current_lr = 5e-6  # 전이학습 마지막 정밀 조정
    # -----------------------

    l2_val = 0
    inputs = Input(shape=(input_dim,))

    # --- [Block 1: 512층] ---
    x = Dense(512, kernel_regularizer=l2(l2_val))(inputs) # activation 삭제
    x = LeakyReLU(alpha=0.01)(x)
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
    outputs = Dense(4, activation='linear')(x)

    lr_schedule = CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=7000, # 학습 끝까지 서서히 줄어들게 설정
        alpha=1e-6 # 최종 학습률의 하한선 (0까지 도달 가능)
    )
    model = Model(inputs=inputs, outputs=outputs)

    if freeze_count > 0:
        # 모델의 레이어 중 앞부분 freeze_count만큼 반복하며 얼림
        # 보통 Dense층과 LeakyReLU층이 쌍으로 있으므로 i*2 혹은 i*3으로 조절해야 하지만
        # 직관적으로 앞의 'n개' 레이어를 지정합니다.
        for layer in model.layers[:freeze_count]:
            layer.trainable = False
        print(f" >>> [동결 완료] 앞부분 {freeze_count}개 레이어를 고정했습니다.")

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

# --- Fold 루프 내부 수정 ---
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_combined)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    X_train_k, X_val_k = X_scaled[train_idx], X_scaled[val_idx]
    y_train_k, y_val_k = y_combined[train_idx], y_combined[val_idx]
    
    # [1] 먼저 동결 범위를 결정
    if LOAD_PREVIOUS:
        if PREVIOUS_TRY < 2: freeze_until = 4 
        elif PREVIOUS_TRY < 4: freeze_until = 2
        else: freeze_until = 0
        try_zero_flag = False
    else:
        freeze_until = 0 
        try_zero_flag = True # 처음 시작할 때만 True

    # [2] 결정된 동결 범위를 넣어서 모델 생성 (학습률도 자동으로 결정됨)
    model = create_model(X_scaled.shape[1], freeze_count=freeze_until, is_try_zero=try_zero_flag)
    
    # 2. 전이학습 모드일 경우 가중치 자동 로드
    if LOAD_PREVIOUS:
        fold_load_path = f'TL1_best_model_try{PREVIOUS_TRY}_fold{fold+1}.weights.h5'
        if os.path.exists(fold_load_path):
            model.load_weights(fold_load_path)
            print(f"[전이학습] Try {PREVIOUS_TRY}의 Fold {fold+1} 가중치를 성공적으로 로드했습니다.")
        else:
            print(f"⚠️ [주의] 로드할 파일을 찾을 수 없습니다: {fold_load_path}. 처음부터 학습합니다.")

    # 3. 학습 세팅
    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=0)
    dynamic_pw_callback = DynamicPhysicsWeight(initial_pw=1e-5, final_pw=5e-4, total_epochs=1000)

    # (Stage 1) 메인 학습
    # LOAD_PREVIOUS가 True라면, 이미 알고 있는 상태에서 '복습'하는 단계가 됩니다.
    print(f"Stage 1: {'전이 학습 시작' if LOAD_PREVIOUS else '일반 학습 시작'}...")
    history = model.fit(
        X_train_k, y_train_k,
        validation_data=(X_val_k, y_val_k),
        epochs=500, batch_size=16,
        callbacks=[early_stop, dynamic_pw_callback],
        verbose=0
    )

    # # (Stage 2) 초미세 조정 (Fine-tuning)
    # print(f"Stage 2: Fine-tuning 시작 (LR: 1e-5)")
    # model.compile(
    #     optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), 
    #     loss=pinn_loss_formulas(y_mean, y_std, 
    #                              w_sigma=weight_sigma, 
    #                              w_b=weight_b, 
    #                              w_ef=weight_ef, 
    #                              w_c=weight_c), 
    #     metrics=[custom_mae]
    # )
    # global_physics_weight.assign(5e-4) # 물리 규제치를 적절히 높은 상태로 유지

    # model.fit(
    #     X_train_k, y_train_k,
    #     validation_data=(X_val_k, y_val_k),
    #     callbacks=[early_stop],
    #     epochs=200, 
    #     batch_size=16,
    #     verbose=0
    # )

    
    # 3. 예측 및 누적
    pred_k = model.predict(X_scaled)
    final_preds_sum += pred_k
    
    # 각 폴드별 가중치 저장 (다음 Try에서 불러오기 위함)
    model.save_weights(f'TL1_best_model_try{CURRENT_TRY}_fold{fold+1}.weights.h5')
    
    best_val_loss = min(history.history['val_loss'])
    print(f"Fold {fold+1} 완료 | Best Val Loss: {best_val_loss:.6f}")
    K.clear_session()
    gc.collect()

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

output_filename = f'PINN_Result_Transfer_Learning1_{CURRENT_TRY}.xlsx'
result_df.to_excel(output_filename, index=False)

print(f"작업 완료! 결과 파일: {output_filename}")

# ---------------------------------------------------------
# 6. 시각화 (그래프 5개 도출)
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 10))

# (1) Loss Curve (마지막 Fold의 History 활용)
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Model Loss Progress')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# (2~5) Fatigue Parameters: Predicted vs Actual
target_names = ['sigma_f', 'b', 'e_f', 'c']
y_actual_original = scaler_y.inverse_transform(y_target_scaled)

for i, name in enumerate(target_names):
    ax = fig.add_subplot(2, 3, i+2)
    
    # 데이터 플롯
    act = y_actual_original[:, i]
    pred = y_pred_ensemble_original[:, i]
    
    ax.scatter(pred, act, alpha=0.5, edgecolors='k', c='royalblue')
    
    # Perfect Prediction Line (y=x)
    min_val = min(min(act), min(pred))
    max_val = max(max(act), max(pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_title(f'{name}: Predicted vs Actual')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Actual Value')
    ax.grid(True)

plt.tight_layout()
# 그래프 자동 저장
plt.savefig(f'1Result_Transfer_Learning_{CURRENT_TRY}.png', dpi=300)
# 최종 가중치 저장 경로 자동 생성
plt.show()