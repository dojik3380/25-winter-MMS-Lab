import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

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
file_path = '원소열처리326ai.xlsx'

try:
    # 1열(Material)을 문자열로 읽기 위해 converters 사용
    df = pd.read_excel(file_path)
    # 결측치 제거
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

print(f"데이터 로드 완료: {df.shape}")

# 2. 데이터 준비
# X (Inputs): B~V열 (Index 1~21)
X = df.iloc[:, 1:22].values 
# y (Targets): W~Z열 (Index 22~25) -> Fatigue Parameters
y = df.iloc[:, 22:26].values
target_names = ['sigma_f', 'b', 'e_f', 'c']

# 최종 결과를 담을 배열 (전체 데이터 크기와 동일)
final_oof_predictions = np.zeros_like(y)

# 각 Fold별 결과를 저장할 딕셔너리
fold_sheets = {}

# 3. K-Fold 설정 (K=5)
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

print(f"\n[{K}-Fold Cross Validation 및 시트 분할 저장 시작]")

# 4. 루프 실행
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1} / {K} 진행 중 ---")
    
    # 데이터 분할
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 스케일링 (Data Leakage 방지: Train 기준으로 fit)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # 모델 구성
    # model = Sequential([
    #     Dense(128, activation='relu', input_dim=X.shape[1]),
    #     Dropout(0.2),
    #     Dense(64, activation='relu'),
    #     Dropout(0.1),
    #     Dense(32, activation='relu'),
    #     Dense(16, activation='relu'),   # ← ⭐ 추가된 층
    #     Dense(4, activation='linear')
    # ])

    model = Sequential([
    Dense(128, input_dim=X.shape[1],kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.01),
    Dropout(0.2),

    Dense(64,kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.01),
    Dropout(0.1),

    Dense(32,kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.01),
    Dense(16),
    LeakyReLU(alpha=0.01),
    Dense(4, activation='linear')
    ])

    # model = Sequential([
    # Dense(128, activation='relu',
    #       input_dim=X.shape[1],
    #       kernel_regularizer=l2(1e-4)),
    # Dropout(0.2),
    # Dense(64, activation='relu',
    #       kernel_regularizer=l2(1e-4)),
    # Dropout(0.1),
    # Dense(32, activation='relu',
    #       kernel_regularizer=l2(1e-4)),
    # Dense(16, activation='relu'),
    # Dense(4, activation='linear')
    # ])
    
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='mse',
        metrics=['mae']
    )
    # reduce_lr = ReduceLROnPlateau(
    # monitor='val_loss',
    # factor=0.5,
    # patience=10,
    # min_lr=1e-5,
    # verbose=1
    # )

    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    # 학습
    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=500,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    
    # 예측
    pred_scaled = model.predict(X_val_scaled)
    pred_original = scaler_y.inverse_transform(pred_scaled)
    
    # 1) 전체 결과 배열(Sheet 1용)에 현재 Fold 예측값 채워넣기
    final_oof_predictions[val_idx] = pred_original
    
    # 2) 현재 Fold 데이터 따로 저장 (Sheet 2~6용)
    # 현재 검증셋에 해당하는 원본 데이터 행을 가져옴
    fold_df = df.iloc[val_idx].copy()
    
    # 예측값 컬럼 추가
    for i, name in enumerate(target_names):
        fold_df[f'{name}_Pred'] = pred_original[:, i]
        
    # 딕셔너리에 저장 (키: 시트 이름)
    sheet_name = f'Fold_{fold+1}'
    fold_sheets[sheet_name] = fold_df
    
    # 성능 출력
    r2 = r2_score(y_val, pred_original)
    print(f"Fold {fold+1} 완료 | Validation R2: {r2:.4f}")

# 5. 엑셀 파일로 저장 (Multi-Sheet)
output_filename = 'KFold_Detailed_Analysis_김도진_31.xlsx'

print(f"\n엑셀 파일 생성 중: {output_filename}")
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    
    # Sheet 1: Final Result (전체 데이터 + 최종 예측값)
    final_df = df.copy()
    for i, name in enumerate(target_names):
        final_df[f'{name}_Pred'] = final_oof_predictions[:, i]
        
    final_df.to_excel(writer, sheet_name='Final_Result', index=False)
    print("- Sheet 1: 'Final_Result' 저장 완료 (전체 데이터)")
    
    # Sheet 2 ~ 6: 각 Fold별 데이터
    for i in range(1, K + 1):
        sheet_name = f'Fold_{i}'
        fold_data = fold_sheets[sheet_name]
        fold_data.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"- Sheet {i+1}: '{sheet_name}' 저장 완료 ({len(fold_data)}행)")

print("\n모든 작업이 완료되었습니다.")