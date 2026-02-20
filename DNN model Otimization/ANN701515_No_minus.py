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
# 5. [개선] 6개 물리 그룹별 Median 보정 로직
# ---------------------------------------------------------

# 1. 예측 및 복원
y_pred_all_scaled = model.predict(X_scaled)
y_pred_all_original = scaler_y.inverse_transform(y_pred_all_scaled)

result_df = df.copy()
for i, name in enumerate(target_names):
    result_df[f'{name}_Pred'] = y_pred_all_original[:, i]

# 2. 6개 그룹 정의를 위한 파생 변수 생성 (UTS/HB ratio)
# 엑셀 기준: UTS는 'TS'열, HB는 'HB'열이라고 가정합니다.
result_df['UTS_HB_Ratio'] = result_df['TS'] / result_df['HB']

# 3. 그룹 판별 함수 정의
def get_material_group(row):
    uts = row['TS']
    ratio = row['UTS_HB_Ratio']
    
    # UTS 기준 그룹 분리 (802, 1238)
    if uts < 802:
        uts_g = 'Low'
    elif uts < 1238:
        uts_g = 'Mid'
    else:
        uts_g = 'High'
        
    # Ratio 기준 분리 (3.66)
    ratio_g = 'Under' if ratio < 3.66 else 'Over'
    
    return f"{uts_g}_{ratio_g}"

# 전체 데이터에 그룹 레이블 할당
result_df['Group_Label'] = result_df.apply(get_material_group, axis=1)

# 4. 각 그룹별 실제 정답(e_f)의 중앙값 계산 (사전 맵핑)
group_median_map = result_df.groupby('Group_Label')['e_f'].median().astype(np.float32).to_dict()

# 5. 보정 실행
EF_COL = 'e_f_Pred'
negative_mask = result_df[EF_COL] <= 0
num_negatives = negative_mask.sum()

if num_negatives > 0:
    print(f"\n[물리 보정] 음수 예측 {num_negatives}건 발견. 6개 그룹별 Median으로 대체합니다.")
    
    for idx in result_df[negative_mask].index:
        group = result_df.loc[idx, 'Group_Label']
        m_value = group_median_map[group]
        result_df.loc[idx, EF_COL] = m_value
        print(f" - Index {idx} ({group}): {m_value:.4f}로 보정됨")
else:
    print("\n[확인] 모든 e_f 예측값이 양수입니다.")

# 임시 컬럼 삭제 후 저장
# result_df.drop(columns=['UTS_HB_Ratio', 'Group_Label'], inplace=True)
output_filename = 'Prediction_Result_김도진_Nominus_31.xlsx'
result_df.to_excel(output_filename, index=False)

print("-" * 50)
print(f"작업 완료! 결과가 '{output_filename}' 파일로 저장되었습니다.")
print(f"파일 내용: A열({df.columns[0]}) 순서 유지, 끝에 예측값 4개 열 추가됨.")
print("-" * 50)