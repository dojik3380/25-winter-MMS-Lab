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
# ---------------------------------------------------------
# 0. 난수(Random Seed) 고정 [가장 먼저 실행]
# ---------------------------------------------------------
seed_value = 42

# 1. Python 및 OS 레벨 난수 고정
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)

# 2. NumPy 난수 고정
np.random.seed(seed_value)

# 3. TensorFlow 난수 고정
tf.random.set_seed(seed_value)

print(f"Random Seed가 {seed_value}로 고정되었습니다.")

# ---------------------------------------------------------
# 1. 데이터 불러오기 및 전처리
# ---------------------------------------------------------
file_path = '원소열처리326ai.xlsx' 
if not os.path.exists(file_path):
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

df = pd.read_excel(file_path)

if df.empty:
    raise ValueError("데이터를 불러오지 못했습니다. 파일 경로를 확인해주세요.")

# 1-1. 데이터 분리
X = df.iloc[:, 1:22].values 
y_target = df.iloc[:, 22:26].values

# 1-2. 물리 변수 추출 (TS, HB, E)
try:
    TS_raw = df['TS'].values.reshape(-1, 1)
    HB_raw = df['HB'].values.reshape(-1, 1)
    E_raw  = df['E'].values.reshape(-1, 1)
except KeyError:
    print("헤더 이름 참조 실패. 인덱스로 강제 할당합니다.")
    HB_raw = df.iloc[:, 19].values.reshape(-1, 1)
    TS_raw = df.iloc[:, 20].values.reshape(-1, 1)
    E_raw  = df.iloc[:, 21].values.reshape(-1, 1)

# 1-3. 스케일링
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
# 2. PINN Custom Loss Function (경험식 적용)
# ---------------------------------------------------------
def pinn_loss_formulas(y_mean, y_std):
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

        loss_data = tf.reduce_mean(tf.square(y_actual_scaled - y_pred))
        loss_phy_sigma = tf.reduce_mean(tf.square(pred_sigma_f - phy_sigma_f))
        loss_phy_b     = tf.reduce_mean(tf.square(pred_b - phy_b))
        loss_phy_ef    = tf.reduce_mean(tf.square(pred_e_f - phy_e_f))
        loss_phy_c     = tf.reduce_mean(tf.square(pred_c - phy_c))
        
        loss_physics = loss_phy_sigma + loss_phy_b + loss_phy_ef + loss_phy_c
        # return loss_data + (physics_weight * loss_physics)
        return loss_data + (global_physics_weight * loss_physics)

    return loss

def custom_mae(y_true, y_pred):
    y_true_target = y_true[:, :4]
    return tf.reduce_mean(tf.abs(y_true_target - y_pred))

# ---------------------------------------------------------
# 3. 모델 생성 함수 (K-Fold를 위해 함수화)
# ---------------------------------------------------------
# def create_model(input_dim):
#     l2_val = 0
#     model = Sequential([
#         Input(shape=(input_dim,)),
#         Dense(128, activation='relu', kernel_regularizer=l2(l2_val)),
#         Dropout(0.2),
#         Dense(64, activation='relu', kernel_regularizer=l2(l2_val)),
#         Dropout(0.1),
#         Dense(32, activation='relu', kernel_regularizer=l2(l2_val)),
#         Dense(16),
#         Dense(4, activation='linear')
#     ])
#     # optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
#     # model.compile(
#     #     optimizer=optimizer, 
#     #     loss=pinn_loss_formulas(y_mean, y_std, physics_weight=1e-5), 
#     #     metrics=[custom_mae]
#     # )
#     # return model

#     # lr_schedule = CosineDecay(
#     #     initial_learning_rate=1e-3,
#     #     decay_steps=1000, # 학습 끝까지 서서히 줄어들게 설정
#     #     alpha=0.0 # 최종 학습률의 하한선 (0까지 도달 가능)
#     # )

#     model.compile(optimizer=Adam(learning_rate=1e-3), # 컴파일 시 learning_rate에 위에서 만든 스케줄러를 넣습니다.
#                   loss=pinn_loss_formulas(y_mean, y_std), 
#                   metrics=[custom_mae])
#     return model

# def create_model(input_dim): # Batch Norm & LeakyReLU
#     model = Sequential([
#         Input(shape=(input_dim,)),
        
#         # --- Block 1 ---
#         Dense(128), 
#         # BatchNormalization(),
#         # Activation('relu'),
#         Activation('swish'),   
#         # LeakyReLU(alpha=0.01),
#         Dropout(0.2),
#         # --- Block 2 ---
#         Dense(64),
#         # BatchNormalization(),
#         # Activation('relu'),
#         Activation('swish'), 
#         # LeakyReLU(alpha=0.01),
#         Dropout(0.1),  
#         # --- Block 3 ---
#         Dense(32),
#         Activation('swish'), 
#         # BatchNormalization(),
#         # Activation('relu'),
#         # LeakyReLU(alpha=0.01),
#         # --- 출력층 ---
#         Dense(4, activation='linear')
#     ])

#     model.compile(
#         optimizer=Adam(learning_rate=1e-3), 
#         loss=pinn_loss_formulas(y_mean, y_std, physics_weight=1e-5), 
#         metrics=[custom_mae]
#     )
#     return model

# def create_model(input_dim): # 5 NiN
#     # 1. 입력층
#     inputs = Input(shape=(input_dim,))
    
#     # --- [Block 1: NiN(MLPConv) 핵심부] ---
#     # 128 노드를 거친 후, 다시 128 노드를 붙여서 내부 특징을 한 번 더 추출합니다.
#     x = Dense(128, activation='relu')(inputs)
#     x = Dense(128, activation='relu')(x)  # 이 줄이 NiN의 핵심입니다!
#     x = Dropout(0.2)(x)
#     # --- [Block 2] ---
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     # --- [Block 3] ---
#     x = Dense(32, activation='relu')(x)
#     # 2. 출력층 (도진님 원래 코드처럼 다시 linear로 원복!)
#     outputs = Dense(4, activation='linear')(x)
#     model = Model(inputs=inputs, outputs=outputs)

#     # 컴파일 설정 (기존과 동일)
#     model.compile(
#         optimizer=Adam(learning_rate=1e-3), 
#         loss=pinn_loss_formulas(y_mean, y_std, physics_weight=1e-5), 
#         metrics=[custom_mae]
#     )
#     return model

# def create_model(input_dim): # 6 ResNet
#     # 1. 입력층
#     inputs = Input(shape=(input_dim,))
    
#     # --- [Block 1: Residual Block] ---
#     # 메인 경로 (데이터가 가공되는 길)
#     x_path = Dense(128, activation='relu')(inputs)
#     x_path = Dropout(0.2)(x_path)
#     x_path = Dense(128)(x_path) # 마지막엔 활성화를 잠시 끕니다 (더한 후 할 예정)
    
#     # 지름길 (입력 피처 21개를 128개로 확장해서 더할 준비)
#     shortcut = Dense(128)(inputs) 
    
#     # [핵심] 가공된 값과 원본(지름길)을 합칩니다.
#     x = Add()([x_path, shortcut]) 
#     x = Activation('relu')(x) # 합친 후 ReLU 적용
    
#     # --- [Block 2 & 3: 일반 Dense] ---
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(32, activation='relu')(x)
    
#     # 2. 출력층 (순수하게 linear로 유지)
#     outputs = Dense(4, activation='linear')(x)
    
#     model = Model(inputs=inputs, outputs=outputs)

#     model.compile(
#         optimizer=Adam(learning_rate=1e-3), 
#         loss=pinn_loss_formulas(y_mean, y_std, physics_weight=1e-5), 
#         metrics=[custom_mae]
#     )
#     return model

# def create_model(input_dim): # 7 DenseBlock
#     # 1. 입력층
#     inputs = Input(shape=(input_dim,))
    
#     # --- [Block 1: DenseNet 컨셉] ---
#     # 1층: 128노드
#     x1 = Dense(128, activation='relu')(inputs)
#     x1 = Dropout(0.2)(x1)
    
#     # 2층: 입력(21) + 1층출력(128)을 합쳐서 64노드로 전달
#     # 이 '합치기'가 데이터 손실을 막아줍니다.
#     concat1 = Concatenate()([inputs, x1]) 
#     x2 = Dense(64, activation='relu')(concat1)
#     x2 = Dropout(0.1)(x2)
    
#     # 3층: 입력(21) + 1층출력(128) + 2층출력(64)을 모두 합쳐서 32노드로 전달
#     concat2 = Concatenate()([inputs, x1, x2])
#     x3 = Dense(32, activation='relu')(concat2)
    
#     # 2. 출력층 (원칙대로 linear 유지)
#     outputs = Dense(4, activation='linear')(x3)
    
#     model = Model(inputs=inputs, outputs=outputs)

#     model.compile(
#         optimizer=Adam(learning_rate=1e-3), 
#         loss=pinn_loss_formulas(y_mean, y_std, physics_weight=1e-5), 
#         metrics=[custom_mae]
#     )
#     return model

def create_model(input_dim): # 8 Wide-Backbone
    # 1. 입력층
    l2_val = 0
    inputs = Input(shape=(input_dim,))
    
    # # --- [Block 1: Wide Layer] ---
    # # 128이 아닌 256으로 너비를 두 배 키웁니다.
    # x = Dense(512, activation='relu', kernel_regularizer=l2(l2_val))(inputs)
    # x = Dense(512, activation='relu', kernel_regularizer=l2(l2_val))(x)
    # x = Dropout(0.2)(x) 
    
    # # --- [Block 2] ---
    # # 병목 현상을 막기 위해 서서히 줄여나갑니다.
    # x = Dense(256, activation='relu', kernel_regularizer=l2(l2_val))(x)
    # x = Dropout(0.1)(x)
    # # --- [Block 3] ---
    # x = Dense(128, activation='relu', kernel_regularizer=l2(l2_val))(x)
    # # x = Dense(64, activation='relu', kernel_regularizer=l2(l2_val))(x)
    # # x = Dense(32, activation='relu', kernel_regularizer=l2(l2_val))(x)
    # x = Dense(16, activation='relu')(x)
    # # 2. 출력층 (원칙 고수: linear 유지)
    # outputs = Dense(4, activation='linear')(x)

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
        loss=pinn_loss_formulas(y_mean, y_std), 
        metrics=[custom_mae]
    )
    return model

# def create_model(input_dim): # softplus
#     # --- [입력층 및 백본] ---
#     inputs = Input(shape=(input_dim,))
    
#     # 도진님의 Baseline 구조 유지
#     x = Dense(128, activation='relu')(inputs)
#     x = Dropout(0.2)(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(32, activation='relu')(x)
    
#     # --- [출력층 분리 로직] ---
#     # 1. 공통 층에서 4개의 원천 값을 뽑음
#     raw_out = Dense(4, activation='linear')(x)
    
#     # 2. 4개의 타겟을 쪼갬 (sigma_f', b, e_f', c 순서라고 가정)
#     # 도진님 데이터 순서가 sigma_f, b, ef, c 라면 아래와 같이 나눕니다.
#     sig_b = Lambda(lambda x: x[:, 0:2])(raw_out)  # sigma_f', b
#     ef_raw = Lambda(lambda x: x[:, 2:3])(raw_out) # e_f' (아직 음수 가능)
#     c_raw = Lambda(lambda x: x[:, 3:4])(raw_out)  # c
    
#     # 3. e_f'에만 Softplus 적용 (음수 방지 핵심!)
#     ef_fixed = Lambda(lambda x: tf.keras.activations.softplus(x))(ef_raw)
    
#     # 4. 다시 합치기
#     outputs = Concatenate()([sig_b, ef_fixed, c_raw])
    
#     model = Model(inputs=inputs, outputs=outputs)

#     model.compile(
#         optimizer=Adam(learning_rate=1e-3), 
#         loss=pinn_loss_formulas(y_mean, y_std, physics_weight=1e-5), 
#         metrics=[custom_mae]
#     )
#     return model
# ---------------------------------------------------------
# 4. K-Fold Cross Validation 학습 및 예측
# ---------------------------------------------------------
n_splits = 5  # Fold 개수 설정
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

# 전체 데이터에 대한 예측값 누적용 배열
final_preds_sum = np.zeros((X_scaled.shape[0], 4))

print(f"[{n_splits}-Fold] K-Fold 교차 검증 학습을 시작합니다...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_combined)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    # 데이터 분할
    X_train_k, X_val_k = X_scaled[train_idx], X_scaled[val_idx]
    y_train_k, y_val_k = y_combined[train_idx], y_combined[val_idx]
    
    # 모델 생성 (매 Fold마다 초기화)
    model = create_model(X_scaled.shape[1])
    
    # 학습 설정
    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    dynamic_pw_callback = DynamicPhysicsWeight(
    initial_pw=1e-5, 
    final_pw=5e-4, 
    total_epochs=500 # 전체 에포크 수에 맞춰 조절
    )

    # 학습 수행
    history = model.fit(
        X_train_k, y_train_k,
        validation_data=(X_val_k, y_val_k),
        epochs=500,
        batch_size=16,
        # callbacks=[early_stop],
        callbacks=[early_stop, dynamic_pw_callback],
        verbose=0  # Fold별 로그 과다 출력 방지
    )
    
    # 현재 모델로 전체 데이터 예측하여 누적
    pred_k = model.predict(X_scaled)
    final_preds_sum += pred_k
    
    # 성능 출력
    best_val_loss = min(history.history['val_loss'])
    print(f"Fold {fold+1} 완료 | Best Val Loss: {best_val_loss:.6f}")

print("\n모든 Fold 학습 완료.")

# ---------------------------------------------------------
# 5. 예측, 후처리(음수 보정) 및 결과 저장
# ---------------------------------------------------------
# (A) 5개 모델 예측값 평균 계산 (앙상블)
y_pred_ensemble_scaled = final_preds_sum / n_splits
y_pred_all_original = scaler_y.inverse_transform(y_pred_ensemble_scaled)

# (B) e_f 음수값 보정 로직 적용 (평균값에 대해 적용)
ef_pred = y_pred_all_original[:, 2]

# 그룹핑을 위한 TS, Ratio 정보 준비 (전체 데이터 기준)
ts_all = TS_raw.flatten()
hb_all = HB_raw.flatten()
ratio_all = ts_all / (hb_all + 1e-6)

# e_f가 음수인 데이터 찾기 (Mask)
neg_mask = ef_pred < 0

# --- 그룹별 중앙값 대체 ---
# [Group 1] TS < 802
mask_g1 = (ts_all < 802)
# 1-1: Ratio > 3.66 -> 0.416
mask_1_1 = mask_g1 & (ratio_all > 3.66)
y_pred_all_original[mask_1_1 & neg_mask, 2] = 0.4160

# 1-2: Ratio <= 3.66 -> 0.4085
mask_1_2 = mask_g1 & (ratio_all <= 3.66)
y_pred_all_original[mask_1_2 & neg_mask, 2] = 0.4085

# [Group 2] 802 <= TS < 1238
mask_g2 = (ts_all >= 802) & (ts_all < 1238)
# 2-1: Ratio > 3.66 -> 0.4599
mask_2_1 = mask_g2 & (ratio_all > 3.66)
y_pred_all_original[mask_2_1 & neg_mask, 2] = 0.4599

# 2-2: Ratio <= 3.66 -> 0.6665
mask_2_2 = mask_g2 & (ratio_all <= 3.66)
y_pred_all_original[mask_2_2 & neg_mask, 2] = 0.6665

# [Group 3] TS >= 1238
mask_g3 = (ts_all >= 1238)
# 3-1: Ratio > 3.66 -> 0.72
mask_3_1 = mask_g3 & (ratio_all > 3.66)
y_pred_all_original[mask_3_1 & neg_mask, 2] = 0.7200

# 3-2: Ratio <= 3.66 -> 0.3414
mask_3_2 = mask_g3 & (ratio_all <= 3.66)
y_pred_all_original[mask_3_2 & neg_mask, 2] = 0.5950

# (C) 엑셀 저장
result_df = df.copy()
target_names = ['sigma_f', 'b', 'e_f', 'c']

for i, name in enumerate(target_names):
    result_df[f'{name}_Pred'] = y_pred_all_original[:, i]

output_filename = 'PINN_baseline42.xlsx'
result_df.to_excel(output_filename, index=False)

print(f"작업 완료! 결과 파일: {output_filename}")
