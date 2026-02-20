import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os, random
from tensorflow.keras.regularizers import l2

# =========================================================
# 0. 시드 고정
# =========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 1. 데이터 불러오기
# =========================================================
file_path = '원소열처리326ai.xlsx'
df = pd.read_excel(file_path).dropna()
print(f"데이터 로드 완료: {df.shape}")

# =========================================================
# 2. 데이터 준비
# =========================================================
X = df.iloc[:, 1:22].values
y = df.iloc[:, 22:26].values
target_names = ['sigma_f', 'b', 'e_f', 'c']

final_oof_predictions = np.zeros_like(y)
fold_sheets = {}

# =========================================================
# 3. 그룹 분류 함수 (⚠ ratio 컬럼 생성 없음)
# =========================================================
def get_material_group(row):
    uts = row['TS']
    ratio = row['TS'] / row['HB']  # 내부 계산만, 저장 안 함

    if uts < 802:
        uts_g = 'Low'
    elif uts < 1238:
        uts_g = 'Mid'
    else:
        uts_g = 'High'

    ratio_g = 'Under' if ratio < 3.66 else 'Over'
    return f"{uts_g}_{ratio_g}"

# 전체 데이터 기준 e_f 중앙값 (보정용)
df_ref = df.copy()
df_ref['Group_Label'] = df_ref.apply(get_material_group, axis=1)
group_median_map = df_ref.groupby('Group_Label')['e_f'].median().to_dict()

# =========================================================
# 4. K-Fold 설정
# =========================================================
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
print(f"\n[{K}-Fold Cross Validation 시작]")

# =========================================================
# 5. K-Fold 루프
# =========================================================
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1} / {K} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)

    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)

    # model = Sequential([
    #     Dense(128, activation='relu', input_dim=X.shape[1]),
    #     Dropout(0.2),
    #     Dense(64, activation='relu'),
    #     Dropout(0.1),
    #     Dense(32, activation='relu'),
    #     Dense(16, activation='relu'),   # ← ⭐ 추가된 층
    #     Dense(4)
    # ])

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
    

    # reduce_lr = ReduceLROnPlateau(
    # monitor='val_loss',
    # factor=0.5,
    # patience=10,
    # min_lr=1e-5,
    # verbose=1
    # )

    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='mse'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )

    model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=500,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    pred = scaler_y.inverse_transform(model.predict(X_val_s))

    fold_df = df.iloc[val_idx].copy()

    for i, name in enumerate(target_names):
        fold_df[f'{name}_Pred'] = pred[:, i]

    # Group_Label만 추가 (ratio 컬럼 없음)
    fold_df['Group_Label'] = fold_df.apply(get_material_group, axis=1)

    # e_f 음수 보정
    neg_mask = fold_df['e_f_Pred'] <= 0
    for idx in fold_df[neg_mask].index:
        g = fold_df.loc[idx, 'Group_Label']
        fold_df.loc[idx, 'e_f_Pred'] = group_median_map[g]

    final_oof_predictions[val_idx] = fold_df[[f'{n}_Pred' for n in target_names]].values
    fold_sheets[f'Fold_{fold+1}'] = fold_df

    r2 = r2_score(y_val, final_oof_predictions[val_idx])
    print(f"Fold {fold+1} R2 (보정후): {r2:.4f}")

# =========================================================
# 6. 엑셀 저장 (🔥 Ratio 없음, Group_Label 맨 오른쪽)
# =========================================================
output_filename = 'KFold_Analysis_김도진_Nominus_31.xlsx'

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    final_df = df.copy()

    for i, name in enumerate(target_names):
        final_df[f'{name}_Pred'] = final_oof_predictions[:, i]

    # ⭐ 맨 마지막 컬럼
    final_df['Group_Label'] = final_df.apply(get_material_group, axis=1)

    final_df.to_excel(writer, sheet_name='Final_Result', index=False)

    for k, v in fold_sheets.items():
        v.to_excel(writer, sheet_name=k, index=False)

print(f"\n완료: '{output_filename}' 생성")
