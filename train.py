import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

print("🚀 1. train.py 내부에서 데이터셋 원격 로드 시작...")
url = "https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv"
df = pd.read_csv(url)
df = df.dropna()

# 특성 선택 (Adult mortality)
features = ['Adult mortality', 'BMI', 'GDP', 'Alcohol']
target = 'Life expectancy'

X = df[features]
y = df[target]

# 데이터 분할
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 과대적합용 50개 샘플 무작위 추출
np.random.seed(42)
train_indices = np.random.choice(X_train_full.index, size=50, replace=False)
X_train = X_train_full.loc[train_indices]
y_train = y_train_full.loc[train_indices]

# 5개 변수 튜플 구조 저장
print("💾 2. data_split.pkl 파일 물리적 저장 중...")
joblib.dump((X_train, X_test, y_train, y_test, X_train_full), 'data_split.pkl')

# 파이프라인 3종 학습 및 저장
pipelines = {
    "Linear": Pipeline([('scaler', StandardScaler()), ('linear', LinearRegression())]),
    "Poly": Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())]),
    "Ridge": Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=3)), ('ridge', Ridge(alpha=1.0))])
}

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f'model_{name.lower()}.pkl')
    print(f"✅ {name} 모델 학습 및 파일 저장 완료!")

print("🎉 train.py 내의 모든 프로세스가 정상 종료되었습니다!")
