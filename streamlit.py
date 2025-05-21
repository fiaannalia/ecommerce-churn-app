import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import pickle

# Judul
st.title("Prediksi Customer Churn")

# Sidebar
st.sidebar.header("Please Input Customer Feature")


def input_user():
    Tenure = st.sidebar.slider("Tenure", min_value=0, max_value=50, value=0, step=1)
    WarehouseToHome = st.sidebar.slider("WarehouseToHome", min_value=0, max_value=50, value=0, step=1)
    NumberOfDeviceRegistered = st.sidebar.selectbox("NumberOfDeviceRegistered", [1, 2, 3, 4, 5, 6])
    SatisfactionScore = st.sidebar.selectbox("SatisfactionScore", [1, 2, 3, 4, 5])
    NumberOfAddress = st.sidebar.slider("NumberOfAddress", min_value=1, max_value=15, value=1, step=1)
    Complain = st.sidebar.radio("Complain", [0, 1])
    DaySinceLastOrder = st.sidebar.slider("DaySinceLastOrder", min_value=0, max_value=30, value=0, step=1)
    CashbackAmount = st.sidebar.number_input("CashbackAmount", min_value=0, max_value=500, value=0, step=1)
    PreferedOrderCat = st.sidebar.selectbox("PreferedOrderCat", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"])
    MaritalStatus = st.sidebar.radio("MaritalStatus", ["Married", "Single", 'Divorced'])

    df = pd.DataFrame()
    df["Tenure"] = [Tenure]
    df["WarehouseToHome"] = [WarehouseToHome]
    df["NumberOfDeviceRegistered"] = [NumberOfDeviceRegistered]
    df["SatisfactionScore"] = [SatisfactionScore]
    df["NumberOfAddress"] = [NumberOfAddress]
    df["Complain"] = [Complain]
    df["DaySinceLastOrder"] = [DaySinceLastOrder]
    df["CashbackAmount"] = [CashbackAmount]
    df["PreferedOrderCat"] = [PreferedOrderCat]
    df["MaritalStatus"] = [MaritalStatus]

    return df

# membuat dataframe berdasarkan input
df_customer = input_user()
df_customer.index = [0]
st.write(df_customer)

# predict customer yang diinput
model_loaded = pickle.load(open("model_lgbm.sav", "rb"))
kelas = model_loaded.predict(df_customer)

# output
if kelas == 0:
    st.write("customer will not churn")
else:
    st.write("customer will churn")