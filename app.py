import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("🧬 WHO 기대수명 예측 및 다중 특성 회귀 모델 비교")
st.write("선택된 4개 특성: `Adult mortality`, `BMI`, `GDP`, `Alcohol` (훈련 샘플 50개 제한)")

# 데이터 및 모델 로드 (5개 변수 구조 일치)
@st.cache_resource
def load_resources():
    X_train, X_test, y_train, y_test, X_train_full = joblib.load('data_split.pkl')
    models = {
        "Linear": joblib.load('model_linear.pkl'),
        "Poly": joblib.load('model_poly.pkl'),
        "Ridge": joblib.load('model_ridge.pkl')
    }
    return X_train, X_test, y_train, y_test, X_train_full, models

X_train, X_test, y_train, y_test, X_train_full, models = load_resources()

# 모델 성능 비교 화면 구현
st.header("📊 1. 모델 성능 비교 (Model Performance Comparison)")
metrics_list = []
for name, model in models.items():
    complexity = model.named_steps['poly'].n_output_features_ if 'poly' in model.named_steps else len(X_train.columns) + 1
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics_list.append({
        "Model": name, "Complexity": complexity,
        "Train R²": r2_score(y_train, train_pred), "Test R²": r2_score(y_test, test_pred),
        "Train MSE": mean_squared_error(y_train, train_pred), "Test MSE": mean_squared_error(y_test, test_pred)
    })
metrics_df = pd.DataFrame(metrics_list)

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("💡 평가지표 테이블")
    st.dataframe(metrics_df.style.format({"Train R²": "{:.4f}", "Test R²": "{:.4f}", "Train MSE": "{:.2f}", "Test MSE": "{:.2f}"}), use_container_width=True)

with col2:
    st.subheader("📉 Test R² 점수 비교 그래프")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x="Model", y="Test R²", data=metrics_df, ax=ax, palette="muted")
    ax.set_ylim(min(metrics_df["Test R²"].min() - 0.1, 0), 1.0)
    st.pyplot(fig)

st.markdown("---")

# 실시간 동적 예측 결과 출력
st.header("🔮 2. 실시간 기대수명 예측 인터페이스")
st.sidebar.header("🎛️ 입력 특성 조절 (Features)")

# 컬럼명 'Adult mortality' 정확히 매칭
adult_mortality = st.sidebar.slider("Adult mortality (성인 사망률)", int(X_train_full['Adult mortality'].min()), int(X_train_full['Adult mortality'].max()), int(X_train['Adult mortality'].median()))
bmi = st.sidebar.slider("BMI (체질량지수)", float(X_train_full['BMI'].min()), float(X_train_full['BMI'].max()), float(X_train['BMI'].median()))
gdp = st.sidebar.slider("GDP (1인당 국내총생산)", float(X_train_full['GDP'].min()), float(X_train_full['GDP'].max()), float(X_train['GDP'].median()))
alcohol = st.sidebar.slider("Alcohol (알코올 섭취량)", float(X_train_full['Alcohol'].min()), float(X_train_full['Alcohol'].max()), float(X_train['Alcohol'].median()))

selected_model_name = st.selectbox("🎯 분석 및 예측에 사용할 회귀 모델을 선택하세요.", ["Linear", "Poly", "Ridge"])
input_data = pd.DataFrame([{'Adult mortality': adult_mortality, 'BMI': bmi, 'GDP': gdp, 'Alcohol': alcohol}])

prediction = models[selected_model_name].predict(input_data)[0]
st.write("### 📋 입력된 데이터 파라미터")
st.dataframe(input_data)
st.subheader(f"🚀 {selected_model_name} 모델의 예측 결과")
st.metric(label="예측된 기대수명 (Life Expectancy)", value=f"{prediction:.2f} 세")
