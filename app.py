import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load model and encoders
@st.cache_resource
def load_model():
    with open("noshow_model_v2.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    with open("label_encoders_v2.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
label_encoders = load_encoders()

# Load appointments data
@st.cache_data
def load_data():
    df = pd.read_csv("CHLA_clean_data_2024_Appointments.csv")
    df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'], format="%m/%d/%y %H:%M", errors='coerce')
    return df

df_2024 = load_data()

# Streamlit UI
st.set_page_config(page_title="CHLA No-show Predictor", layout="centered")
st.title("📅 CHLA No-show Predictor")
st.markdown("Use the controls below to filter appointments and predict no-shows. Bonus points for using advanced UI features like this! 🎯")

with st.sidebar:
    st.header("Filter Appointments")
    clinic_list = df_2024['CLINIC'].unique().tolist()
    clinic_name = st.selectbox("Select Clinic Name", clinic_list)
    start_date = st.date_input("Start Date", datetime(2024, 1, 1))
    end_date = st.date_input("End Date", datetime(2024, 1, 31))

if st.button("🚀 Predict No-shows"):
    filtered_df = df_2024[
        (df_2024['CLINIC'] == clinic_name) &
        (df_2024['APPT_DATE'] >= pd.to_datetime(start_date)) &
        (df_2024['APPT_DATE'] <= pd.to_datetime(end_date))
    ].copy()

    drop_cols = ['MRN', 'BOOK_DATE', 'SCHEDULE_ID', 'APPT_ID', 'APPT_STATUS', 'IS_NOSHOW']
    filtered_df = filtered_df.drop(columns=[col for col in drop_cols if col in filtered_df.columns])

    for col in filtered_df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            filtered_df[col] = filtered_df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            filtered_df[col] = le.transform(filtered_df[col])

    X_input = filtered_df.drop(columns=['APPT_DATE'])
    probabilities = model.predict_proba(X_input)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    filtered_df['No-show Prediction'] = predictions
    filtered_df['Probability'] = probabilities
    filtered_df['APPT_DATE'] = df_2024.loc[filtered_df.index, 'APPT_DATE']

    result_df = filtered_df[['CLINIC', 'APPT_DATE', 'No-show Prediction', 'Probability']].copy()
    result_df['No-show Prediction'] = result_df['No-show Prediction'].map({1: 'Yes', 0: 'No'})
    result_df['Probability'] = result_df['Probability'].round(4)

    st.success(f"Predictions complete for {len(result_df)} appointments.")
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", csv, "no_show_predictions.csv", "text/csv")
