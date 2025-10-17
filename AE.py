import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------
# 🧭 기본 설정
# -----------------------------------
st.set_page_config(
    page_title="제조 설비 이상 탐지 시스템",
    layout="wide",
    page_icon="🏭"
)

st.title("🏭 제조 설비 이상 탐지 시스템")
st.markdown("""
업로드된 **센서 로그 데이터(CSV)**를 기반으로  
자동으로 이상치 구간과 발생 시점을 분석합니다.
""")

# -----------------------------------
# 📂 파일 업로드
# -----------------------------------
uploaded_file = st.file_uploader("📁 센서 데이터 파일 업로드 (CSV 형식)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 데이터 업로드 완료!")

    # -----------------------------------
    # 🧹 데이터 정제
    # -----------------------------------
    if 'sensor_15' in df.columns:
        df = df.drop(columns=['sensor_15'])
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df.columns = df.columns.str.strip().str.lower()

    if 'timestamp' not in df.columns:
        st.error("❌ 'timestamp' 컬럼이 없습니다. 시간 정보를 포함한 CSV를 업로드해주세요.")
        st.stop()

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp')

    # machine_status 없을 수 있으므로 optional
    if 'machine_status' in df.columns:
        df['machine_status'] = df['machine_status'].astype(str).str.strip().str.upper()
    else:
        df['machine_status'] = "UNKNOWN"

    # -----------------------------------
    # 🔍 센서 데이터만 추출
    # -----------------------------------
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    if not sensor_cols:
        st.error("❌ 'sensor_'로 시작하는 센서 컬럼이 없습니다.")
        st.stop()

    normal_df = df[df['machine_status'] == 'NORMAL'] if 'NORMAL' in df['machine_status'].unique() else df.copy()
    X_train = normal_df[sensor_cols].apply(pd.to_numeric, errors='coerce')
    X_test = df[sensor_cols].apply(pd.to_numeric, errors='coerce')

    # 결측 처리
    X_train = X_train.interpolate(limit_direction='both').fillna(X_train.mean())
    X_test = X_test.interpolate(limit_direction='both').fillna(X_train.mean())

    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------
    # 🧠 Autoencoder 모델
    # -----------------------------------
    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    bottleneck = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(bottleneck)
    decoded = Dense(32, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    with st.spinner('모델 학습 중... ⏳'):
        autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=30,
            batch_size=256,
            validation_split=0.2,
            callbacks=[es],
            verbose=0
        )

    st.success("✅ 이상 탐지 모델 학습 완료!")

    # -----------------------------------
    # 📉 이상 탐지
    # -----------------------------------
    reconstructions = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
    df['reconstruction_error'] = mse

    threshold = np.percentile(mse, 99)
    df['anomaly'] = df['reconstruction_error'] > threshold

    # -----------------------------------
    # 📊 통계 분석
    # -----------------------------------
    total_records = len(df)
    num_anomalies = df['anomaly'].sum()
    anomaly_ratio = (num_anomalies / total_records) * 100

    st.subheader("📈 분석 결과 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("총 데이터 수", f"{total_records:,}")
    c2.metric("이상치 탐지 수", f"{num_anomalies:,}")
    c3.metric("이상치 비율", f"{anomaly_ratio:.2f}%")

    if 'machine_status' in df.columns:
        status_summary = df.groupby('machine_status')['anomaly'].agg(['sum', 'mean']).rename(columns={'sum': '이상치 수', 'mean': '비율'})
        status_summary['비율'] = (status_summary['비율'] * 100).round(2)
        st.write("### 상태별 이상 탐지 비율")
        st.dataframe(status_summary)

    # -----------------------------------
    # 📈 시각화
    # -----------------------------------
    st.subheader("📊 이상치 시각화")
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df['timestamp'], df['reconstruction_error'], label='Reconstruction Error', color='steelblue')
    ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold (99%)')
    anomalies = df[df['anomaly']]
    ax.scatter(anomalies['timestamp'], anomalies['reconstruction_error'], color='orange', s=12, label='Anomaly')
    ax.legend()
    ax.set_title("Reconstruction Error Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Reconstruction Error")
    st.pyplot(fig)

    # -----------------------------------
    # 📍 이상 발생 구간 요약
    # -----------------------------------
    st.subheader("🧭 이상 발생 구간 요약")
    anomalies_sorted = anomalies[['timestamp', 'reconstruction_error']].sort_values('timestamp')
    if not anomalies_sorted.empty:
        start_time = anomalies_sorted['timestamp'].min()
        end_time = anomalies_sorted['timestamp'].max()
        st.write(f"📅 **이상치 발생 기간:** {start_time.strftime('%Y-%m-%d %H:%M:%S')} → {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"⚠️ **이상치 구간 개수:** {len(anomalies_sorted)}")
        st.dataframe(anomalies_sorted.head(20))
    else:
        st.info("✅ 이상치가 탐지되지 않았습니다. (모든 구간 정상)")

    # -----------------------------------
    # ⬇️ 결과 다운로드
    # -----------------------------------
    st.download_button(
        label="📥 이상 탐지 결과 CSV 다운로드",
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name="anomaly_detection_result.csv",
        mime="text/csv"
    )

else:
    st.info("👆 CSV 파일을 업로드하면 자동으로 이상 탐지 분석이 시작됩니다.")
