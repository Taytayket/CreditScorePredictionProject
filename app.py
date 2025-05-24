import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Score Predictor", layout="centered")

st.title("💳 Credit Risk Scoring App")
st.markdown("This tool estimates your credit score and default risk based on LendingClub-style inputs.")

with st.form("loan_form"):
    st.subheader("Loan & Personal Info")

    loan_amnt = st.number_input(
        "Loan Amount ($)", 
        min_value=500, 
        max_value=1_000_000, 
        step=500, 
        value=10_000
    )

    term = st.selectbox("Loan Term", [
        "12 months", "24 months", "36 months", "48 months", "60 months", "72 months"
    ])

    int_rate = st.slider(
        "Interest Rate (%)", 
        min_value=0.0, 
        max_value=40.0, 
        step=0.1, 
        value=13.56
    )

    installment = st.number_input("Monthly Installment ($)", min_value=0.0, step=10.0, value=300.0)

    grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"])
    sub_grade = st.selectbox("Sub Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)])

    emp_length = st.selectbox("Employment Length", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", 
        "7 years", "8 years", "9 years", "10+ years"
    ])

    home_ownership = st.selectbox("Home Ownership", [
        "RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"
    ])

    annual_inc = st.number_input(
        "Annual Income ($)", 
        min_value=0.0, 
        max_value=10_000_000.0, 
        step=1000.0, 
        value=60_000.0
    )

    purpose = st.selectbox("Loan Purpose", [
        "credit_card", "debt_consolidation", "home_improvement", "major_purchase", 
        "medical", "car", "vacation", "moving", "small_business", "wedding", "house", 
        "renewable_energy", "educational", "other"
    ])

    dti = st.slider("DTI (Debt-to-Income Ratio)", 0.0, 60.0, step=0.5, value=15.0)
    open_acc = st.number_input("Open Accounts", min_value=0, max_value=100, step=1, value=10)
    total_acc = st.number_input("Total Accounts", min_value=0, max_value=200, step=1, value=25)

    earliest_cr_line = st.text_input("Earliest Credit Line (YYYY-MM-DD)", value="2005-05-01")

    submitted = st.form_submit_button("Predict Credit Score")

if submitted:
    input_data = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "purpose": purpose,
        "dti": dti,
        "open_acc": open_acc,
        "total_acc": total_acc,
        "earliest_cr_line": earliest_cr_line
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

        if response.status_code == 200:
            result = response.json()

            st.success(f"🎯 Credit Score: **{result['credit_score']}**")
            st.info(f"💡 Risk Level: **{result['risk_level']}**")
            st.metric("📉 Default Probability", f"{result['predicted_probability']*100:.2f} %")

            st.subheader("Score Explanation")
            st.markdown("These features had the strongest influence on the prediction:")

            contrib_data = pd.DataFrame(result["explanation"])
            contrib_data["Impact"] = contrib_data["contribution"].apply(lambda x: "↑" if x > 0 else "↓")

            fig, ax = plt.subplots()
            contrib_data.plot(
                x="feature",
                y="contribution",
                kind="barh",
                color=contrib_data["contribution"].apply(lambda x: "#00C49F" if x > 0 else "#FF4B4B"),
                ax=ax
            )
            ax.set_xlabel("Contribution to Score (weighted)")
            ax.invert_yaxis()
            st.pyplot(fig)

        else:
            st.error(f"Prediction failed. Status code: {response.status_code}")

    except Exception as e:
        st.error(f"Error: {e}")

