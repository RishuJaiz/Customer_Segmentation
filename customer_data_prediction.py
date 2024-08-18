import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_page_config(page_title="Customer Segmentation Predictor", layout="wide")

# Custom CSS to improve the aesthetics
st.markdown('''
<style>
body {
    background-color: #f0f2f6;
    color: #333;
}
.css-18e3th9 {
    padding: 0em 1em;
}
div.stButton > button {
    width: 100%;
    border-radius: 20px;
    border: 1px solid #0e76a8;
    background-color: #0e76a8;
    color: #fff;
    font-size: 16px;
    height: 2.5em;
    line-height: 2.5em;
    padding: 0 1.5em;
}
div.stTextInput > div > div > input {
    border-radius: 10px;
}
div.stNumberInput > div > div {
    border-radius: 10px;
}
</style>
''', unsafe_allow_html=True)

st.title(":bar_chart: Customer Segmentation Prediction")

# Form layout
with st.form("my_form"):
    st.write("#### Please enter the following customer details:")
    
    cols = st.columns((2, 2, 2, 2, 2, 2))
    
    with cols[0]:
        balance = st.number_input('Balance', step=0.001, format="%.6f", help="Total balance on the credit card.")
    with cols[1]:
        balance_frequency = st.number_input('Balance Frequency', step=0.001, format="%.6f", help="Frequency of balance updates.")
    with cols[2]:
        purchases = st.number_input('Purchases', step=0.01, format="%.2f", help="Total purchase amount.")
    with cols[3]:
        oneoff_purchases = st.number_input('OneOff Purchases', step=0.01, format="%.2f", help="Total amount of one-off purchases.")
    with cols[4]:
        installments_purchases = st.number_input('Installments Purchases', step=0.01, format="%.2f", help="Total amount of installment purchases.")
    with cols[5]:
        cash_advance = st.number_input('Cash Advance', step=0.01, format="%.6f", help="Total cash advance amount.")

    cols2 = st.columns((2, 2, 2, 2, 2, 2))
    with cols2[0]:
        purchases_frequency = st.number_input('Purchases Frequency', step=0.01, format="%.6f", help="How often purchases are made.")
    with cols2[1]:
        oneoff_purchases_frequency = st.number_input('OneOff Purchases Frequency', step=0.1, format="%.6f", help="Frequency of one-off purchases.")
    with cols2[2]:
        purchases_installment_frequency = st.number_input('Purchases Installments Frequency', step=0.1, format="%.6f", help="Frequency of installment purchases.")
    with cols2[3]:
        cash_advance_frequency = st.number_input('Cash Advance Frequency', step=0.1, format="%.6f", help="Frequency of cash advances.")
    with cols2[4]:
        cash_advance_trx = st.number_input('Cash Advance Trx', step=1, help="Number of cash advance transactions.")
    with cols2[5]:
        purchases_trx = st.number_input('Purchases TRX', step=1, help="Number of purchase transactions.")

    cols3 = st.columns((1, 1, 1, 1, 1))
    with cols3[0]:
        credit_limit = st.number_input('Credit Limit', step=0.1, format="%.1f", help="Credit limit on the card.")
    with cols3[1]:
        payments = st.number_input('Payments', step=0.01, format="%.6f", help="Total payments made.")
    with cols3[2]:
        minimum_payments = st.number_input('Minimum Payments', step=0.01, format="%.6f", help="Minimum required payments.")
    with cols3[3]:
        prc_full_payment = st.number_input('PRC Full Payment', step=0.01, format="%.6f", help="Percentage of full payments made.")
    with cols3[4]:
        tenure = st.number_input('Tenure', step=1, help="Tenure of the credit card service.")

    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance, purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency, cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]

    submitted = st.form_submit_button("🔍 Submit Prediction")

if submitted:
    clust = loaded_model.predict(data)[0]
    st.success(f'Data Belongs to Cluster: {clust}')

    cluster_df1 = df[df['Cluster'] == clust]
    for c in cluster_df1.drop(['Cluster'], axis=1).columns:
        fig, ax = plt.subplots(figsize=(20, 3))
        sns.histplot(cluster_df1[c], kde=False, ax=ax)
        st.pyplot(fig)
