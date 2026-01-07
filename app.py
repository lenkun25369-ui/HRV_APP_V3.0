import os, json, tempfile, subprocess
import streamlit as st
import requests
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shock_rate import predict_shock

# =========================================
# UI Header
# =========================================
st.title("SHIELD")
st.caption("HRV Sepsis Early Warning System Powerd by AI")

risk_placeholder = st.empty()
ecg_hrv_placeholder = st.empty()

qp = st.experimental_get_query_params()
token_q = qp.get("token", [""])[0]
obs_q   = qp.get("obs", [""])[0]

# =========================================
# Check Models
# =========================================
@st.cache_resource
def _check_models_exist():
    assert os.path.exists("models/model_focalloss.h5")
    assert os.path.exists("models/xgb_model.json")

_check_models_exist()

# =========================================
# FHIR Fetch
# =========================================
def fetch_observation(token, obs_url):
    r = requests.get(
        obs_url,
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
        timeout=20
    )
    r.raise_for_status()
    return r.json()

# =========================================
# Patient Data Placeholder
# =========================================
st.markdown("---")
patient_data_placeholder = st.empty()
with patient_data_placeholder.container():
    st.expander("Patient Data (Click to Expand)", expanded=False)

# =========================================
# Token & Observation URL
# =========================================
token = st.text_input("Token", value=token_q, type="password")
obs_url = st.text_input("Observation URL", value=obs_q)

# =========================================
# Auto Run Logic
# =========================================
if token and obs_url:

    # ⭐⭐⭐ 重流程只跑一次 ⭐⭐⭐
    if "analysis_done" not in st.session_state:

        with st.spinner("Fetching Patient Data..."):
            obs = fetch_observation(token, obs_url)

        with patient_data_placeholder.container():
            with st.expander("Patient Data (Click to Expand)", expanded=False):
                st.json(obs)

        with tempfile.TemporaryDirectory() as td:
            obs_path = os.path.join(td, "obs.json")
            ecg_csv  = os.path.join(td, "ECG_5min.csv")
            h0_csv   = os.path.join(td, "h0.csv")

            with open(obs_path, "w") as f:
                json.dump(obs, f)

            # ----- Parse ECG -----
            proc = subprocess.run(
                ["python", "parse_fhir_ecg_to_csv.py", obs_path, ecg_csv],
                capture_output=True,
                text=True
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr)

            ecg_df = pd.read_csv(ecg_csv, header=None)
            ecg_signal = (
                pd.to_numeric(ecg_df.iloc[:, 0], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
                .ravel()
            )
            if ecg_signal.size == 0:
                raise RuntimeError("ECG signal empty")

            # ----- Generate HRV -----
            proc = subprocess.run(
                ["python", "generate_HRV_10_features.py", ecg_csv, h0_csv],
                capture_output=True,
                text=True
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr)

            hrv_df = pd.read_json(proc.stdout.splitlines()[-1], orient="records")

            # ----- Predict Risk -----
            preds = predict_shock(h0_csv)

        # ⭐ 存進 session_state
        st.session_state.obs = obs
        st.session_state.ecg_signal = ecg_signal
        st.session_state.hrv_df = hrv_df
        st.session_state.preds = preds
        st.session_state.analysis_done = True

        st.success("Done")

    # ===== 以下開始：只 rerender，不重算 =====
    ecg_signal = st.session_state.ecg_signal
    hrv_df = st.session_state.hrv_df
    preds = st.session_state.preds

    # =========================================
    # Risk Visualization
    # =========================================
    risk_pct = round(preds[0] * 100, 2)
    risk_label = "LOW RISK" if risk_pct < 20 else "MODERATE RISK" if risk_pct < 40 else "HIGH RISK"
    risk_color = "#2ecc71" if risk_pct < 20 else "#f39c12" if risk_pct < 40 else "#e74c3c"

    with risk_placeholder.container():
        pie_col, value_col = st.columns([1, 2])
        with pie_col:
            components.html(
                f"<div style='width:120px;height:120px;border-radius:50%;background:conic-gradient({risk_color} {risk_pct}%,#2c2c2c {risk_pct}%);'></div>",
                height=140,
            )
        with value_col:
            st.markdown(f"### {risk_pct:.2f}%  \n**{risk_label}**")

    # =========================================
    # ECG Input & HRV Features
    # =========================================
    with ecg_hrv_placeholder.container():
        st.markdown("---")
        st.subheader("ECG Input & HRV Features")

        hr = st.session_state.ecg_signal
        n = len(hr)

        start_idx = st.slider(
            "View window start index",
            min_value=0,
            max_value=max(0, n - 50),
            value=750,
            step=1
        )

        win = hr[start_idx:start_idx + 50]
        x = np.arange(start_idx, start_idx + len(win))

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(x, win)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("bpm")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**HRV Features Output**")
        row = hrv_df.iloc[0]
        cols1 = st.columns(5)
        cols2 = st.columns(5)
        for i in range(5):
            cols1[i].metric(row.index[i], f"{row.values[i]:.3f}")
            cols2[i].metric(row.index[i+5], f"{row.values[i+5]:.3f}")

else:
    st.info("Please enter Token and Observation URL to start calculation")
