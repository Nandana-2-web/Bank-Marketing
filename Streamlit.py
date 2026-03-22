import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -----------------------------
# Load Model + Dataset
# -----------------------------

model = pickle.load(open("bank_model.pkl","rb"))
df = pd.read_csv("bank.csv", sep=";")
df.columns = df.columns.str.strip()

st.set_page_config(page_title="Bank Marketing Prediction", page_icon="🏦", layout="centered")

# -----------------------------
# Sidebar Navigation
# -----------------------------

page = st.sidebar.selectbox(
    "Navigation",
    ["Customer Prediction","Prediction Probability","Dataset Insights"]
)


# -----------------------------
# CSS Styling
# -----------------------------

st.markdown("""
<style>

.stApp{
background-image:url("https://images.unsplash.com/photo-1556745757-8d76bdb6984b");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

.main-title{
font-size:95px;
font-weight:900;
text-align:center;
color:white;
margin-top:120px;
}

.subtitle{
text-align:center;
font-size:40px;
font-weight:bold;
color:white;
margin-bottom:40px;
}

label{
font-weight:bold !important;
color:white !important;
}

.stButton>button{
width:100%;
background-color:#0a4f9c;
color:white;
font-size:22px;
font-weight:bold;
border-radius:10px;
padding:12px;
}

/* SUCCESS BOX */

.success-box{
background: rgba(0,150,0,0.8);
color:white;
font-size:32px;
font-weight:bold;
padding:20px;
border-radius:10px;
text-align:center;
margin-top:20px;
}

/* ERROR BOX */

.error-box{
background: rgba(200,0,0,0.8);
color:white;
font-size:32px;
font-weight:bold;
padding:20px;
border-radius:10px;
text-align:center;
margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Dropdown Mappings
# -----------------------------

job_options={"Choose":0,"1 Admin":1,"2 Technician":2,"3 Services":3,"4 Management":4,"5 Retired":5,"6 Student":6}
marital_options={"Choose":0,"1 Single":1,"2 Married":2,"3 Divorced":3}
education_options={"Choose":0,"1 Primary":1,"2 Secondary":2,"3 Tertiary":3}
yes_no={"Choose":0,"0 Yes":0,"2 No":2}
contact_options={"Choose":0,"1 Cellular":1,"2 Telephone":2}

month_options={
"Choose":0,"1 Jan":1,"2 Feb":2,"3 Mar":3,"4 Apr":4,"5 May":5,"6 Jun":6,
"7 Jul":7,"8 Aug":8,"9 Sep":9,"10 Oct":10,"11 Nov":11,"12 Dec":12
}

day_options={"Choose":0,"1 Monday":1,"2 Tuesday":2,"3 Wednesday":3,"4 Thursday":4,"5 Friday":5}

# -----------------------------
# Input Form (Used in multiple pages)
# -----------------------------

def get_inputs():

    age = st.slider("Age",18,90,30)

    col1,col2=st.columns(2)

    with col1:
        job=st.selectbox("Job",list(job_options.keys()))
        marital=st.selectbox("Marital Status",list(marital_options.keys()))
        education=st.selectbox("Education",list(education_options.keys()))
        default=st.selectbox("Credit Default",list(yes_no.keys()))

    with col2:
        housing=st.selectbox("Housing Loan",list(yes_no.keys()))
        loan=st.selectbox("Personal Loan",list(yes_no.keys()))
        contact=st.selectbox("Contact Type",list(contact_options.keys()))
        month=st.selectbox("Month",list(month_options.keys()))

    day=st.selectbox("Day of Week",list(day_options.keys()))

    input_data=np.array([[age,
    job_options[job],
    marital_options[marital],
    education_options[education],
    yes_no[default],
    yes_no[housing],
    yes_no[loan],
    contact_options[contact],
    month_options[month],
    day_options[day]]])

    return input_data

# ==========================================================
# 1️⃣ CUSTOMER PREDICTION
# ==========================================================

if page=="Customer Prediction":

    st.markdown('<h1 class="main-title">🏦 Smart Bank Predictor</h1>',unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict if a customer will subscribe</p>',unsafe_allow_html=True)

    input_data=get_inputs()

    if st.button("Predict"):

        prediction=model.predict(input_data)

        if prediction[0] == 1:
            st.markdown(
                '<div class="success-box">✅ Customer WILL Subscribe to the Deposit</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                '<div class="error-box">❌ Customer WILL NOT Subscribe to the Deposit</div>',
                unsafe_allow_html=True
            )
# ==========================================================
# 2️⃣ PREDICTION PROBABILITY
# ==========================================================

elif page=="Prediction Probability":

    st.markdown('<h1 class="main-title">📊 Prediction Probability</h1>',unsafe_allow_html=True)

    input_data=get_inputs()

    if st.button("Show Probability"):

        prob=model.predict_proba(input_data)

        success=prob[0][1]*100
        failure=prob[0][0]*100

        # Gauge Chart
        fig=go.Figure(go.Indicator(
            mode="gauge+number",
            value=success,
            title={'text':"Subscription Probability"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"green"}
            }
        ))

        st.plotly_chart(fig)

# ==========================================================
# 3️⃣ DATASET INSIGHTS
# ==========================================================

elif page=="Dataset Insights":

    st.markdown('<h1 class="main-title">📊 Dataset Insights</h1>',unsafe_allow_html=True)

    success_count=(df['y']=="yes").sum()
    failure_count=(df['y']=="no").sum()

    total=len(df)

    success_prob=round((success_count/total)*100,2)
    failure_prob=round((failure_count/total)*100,2)

    col1,col2=st.columns(2)

    with col1:
        st.metric("Success Probability",f"{success_prob}%")

    with col2:
        st.metric("Failure Probability",f"{failure_prob}%")

    # Pie Chart
    fig,ax=plt.subplots(figsize=(3,2))

    ax.pie(
        [success_count,failure_count],
        labels=["Subscribed","Not Subscribed"],
        autopct="%1.1f%%"
    )

    ax.set_title("Customer Subscription Distribution")

    st.pyplot(fig)

    # -----------------------------
    # Feature Importance
    # -----------------------------

    st.subheader("📈 Feature Importance")

    if hasattr(model,"feature_importances_"):

        features=[
        "Age","Job","Marital","Education","Default",
        "Housing","Loan","Contact","Month","Day"
        ]

        importance=model.feature_importances_

        fig,ax=plt.subplots()

        ax.barh(features,importance)

        ax.set_title("Feature Importance")

        st.pyplot(fig)

    else:
        st.info("Feature importance not available for this model")

    # -----------------------------
    # Risk Score Meter
    # -----------------------------

    st.subheader("🧠 Customer Risk Score Example")

    risk_score=(failure_prob)

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text':"Average Risk Score"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"red"}
        }
    ))

    st.plotly_chart(fig)

    # -----------------------------
    # Find Successful Customer Profiles
    # -----------------------------

    st.subheader("🎯 Find Successful Customer Profiles")

    if st.button("Generate Successful Profiles"):

        import random

        successful_profiles = []

        for i in range(1000):

            test = np.array([[

                random.randint(25, 70),
                random.randint(1, 6),
                random.randint(1, 3),
                random.randint(1, 3),
                random.randint(1, 2),
                random.randint(1, 2),
                random.randint(1, 2),
                random.randint(1, 2),
                random.randint(1, 12),
                random.randint(1, 5)

            ]])

            pred = model.predict(test)

            if pred[0] == 1:
                successful_profiles.append(test[0])

            if len(successful_profiles) == 10:
                break

        if successful_profiles:



            job_map = {1: "Admin", 2: "Technician", 3: "Services", 4: "Management", 5: "Retired", 6: "Student"}
            marital_map = {1: "Single", 2: "Married", 3: "Divorced"}
            education_map = {1: "Primary", 2: "Secondary", 3: "Tertiary"}
            yes_no_map = {1: "Yes", 2: "No"}
            contact_map = {1: "Cellular", 2: "Telephone"}

            month_map = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }

            day_map = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday"}

            for p in successful_profiles:
                st.write({
                    "Age": p[0],
                    "Job": job_map.get(p[1], ""),
                    "Marital": marital_map.get(p[2], ""),
                    "Education": education_map.get(p[3], ""),
                    "Default": yes_no_map.get(p[4], ""),
                    "Housing Loan": yes_no_map.get(p[5], ""),
                    "Personal Loan": yes_no_map.get(p[6], ""),
                    "Contact": contact_map.get(p[7], ""),
                    "Month": month_map.get(p[8], ""),
                    "Day": day_map.get(p[9], "")
                })

        else:
            st.warning("No successful profiles found")