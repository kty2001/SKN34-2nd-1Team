"""
데이터 분리 및 스케일링 (노트북 3절).
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def split_and_scale(df, target="Churn", test_size=0.2, random_state=RANDOM_STATE):
    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Train:", X_train.shape, " Test:", X_test.shape)
    print("Train 이탈률:", y_train.mean().round(3), " Test 이탈률:", y_test.mean().round(3))

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }
