import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os, random
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. 데이터 불러오기
file_path = '원소열처리326ai.xlsx' # 업로드한 파일명
df = pd.read_excel(file_path)

# 데이터가 없는 경우를 대비한 예외처리
if df.empty:
    raise ValueError("데이터를 불러오지 못했습니다. 파일 경로를 확인해주세요.")

# 2. 데이터 전처리
# A열: Material (순서 유지를 위해 따로 저장해둠)
materials = df.iloc[:, 0].values

# X (Input): B~V열 (Index 1~21)
X = df.iloc[:, 1:22].values 

# y (Target): W~Z열 (Index 22~25)
y = df.iloc[:, 22:26].values
target_names = ['sigma_f', 'b', 'e_f', 'c'] # 예측할 4개 변수 이름

# 스케일링 (학습 효과 증대)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 3. 데이터 분할 (학습용) - 70:15:15
# random_state=42로 고정하여 항상 같은 데이터가 학습/테스트로 나뷤
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. ANN 모델 구성 및 학습
model = Sequential([
    Dense(128, input_dim=X.shape[1], kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.01),
    Dropout(0.2),

    Dense(64, kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.01),
    Dropout(0.1),

    Dense(32, kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.01),
    Dense(16),
    LeakyReLU(alpha=0.01),
    Dense(4, activation='linear')
])

# model = Sequential([
#     Dense(128, activation='relu', input_dim=X.shape[1]),
#     Dropout(0.2),
#     Dense(64, activation='relu'),
#     Dropout(0.1),
#     Dense(32, activation='relu'),
#     Dense(16, activation='relu'),   # ← ⭐ 추가된 층
#     Dense(4, activation='linear')
# ])


# model = Sequential([
#     Dense(128, activation='relu',
#           input_dim=X.shape[1],
#           kernel_regularizer=l2(1e-4)),
#     Dropout(0.2),
#     Dense(64, activation='relu',
#           kernel_regularizer=l2(1e-4)),
#     Dropout(0.1),
#     Dense(32, activation='relu',
#           kernel_regularizer=l2(1e-4)),
#     Dense(16, activation='relu'),
#     Dense(4, activation='linear')
# ])


# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=10,
#     min_lr=1e-5,
#     verbose=1
# )

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

print("모델 학습을 시작합니다...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0 # 학습 로그 생략 (깔끔한 출력을 위해)
)
print("모델 학습 완료.")

# ---------------------------------------------------------
# 5. [핵심] 전체 데이터에 대한 예측 및 엑셀 저장
# ---------------------------------------------------------

# 전체 데이터(X_scaled)를 순서대로 넣어 예측 수행
y_pred_all_scaled = model.predict(X_scaled)

# 스케일링 된 예측값을 원래 단위로 복원
y_pred_all_original = scaler_y.inverse_transform(y_pred_all_scaled)

# 결과 저장을 위한 데이터프레임 생성
result_df = df.copy()

# 예측 결과 열 추가 (이름 뒤에 '_Pred' 붙임)
for i, name in enumerate(target_names):
    # 각 파라미터별 예측값 열 추가
    result_df[f'{name}_Pred'] = y_pred_all_original[:, i]
    
    # (선택사항) 오차율을 보고 싶다면 아래 주석을 해제하세요
    # result_df[f'{name}_Error'] = result_df[name] - result_df[f'{name}_Pred']

# 엑셀로 저장
output_filename = 'Prediction_Result_김도진_31.xlsx'
result_df.to_excel(output_filename, index=False)

print("-" * 50)
print(f"작업 완료! 결과가 '{output_filename}' 파일로 저장되었습니다.")
print(f"파일 내용: A열({df.columns[0]}) 순서 유지, 끝에 예측값 4개 열 추가됨.")
print("-" * 50)

# 간단한 결과 확인 (상위 5개)
print("\n[저장된 데이터 미리보기 (Material 및 예측값)]")
print(result_df[['Material'] + [f'{n}_Pred' for n in target_names]].head())