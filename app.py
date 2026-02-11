import streamlit as st
import joblib
import numpy as np

# -------- LOAD FILES --------
model = joblib.load("rf_model.pkl")
le_cat = joblib.load("cat_encoder.pkl")
le_rec = joblib.load("rec_encoder.pkl")

st.set_page_config(page_title="Smart Recommender", layout="wide")

# -------- STYLE --------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#fdfbfb,#ebedee);
}

.title {
    font-size:42px;
    font-weight:800;
    text-align:center;
    color:#1a237e;
}

.card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown('<div class="title">🛍️ Product Recommendation AI</div>', unsafe_allow_html=True)
st.write("Personalized suggestions based on user behaviour")

# -------- INPUT PANEL --------
col1,col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    browsing = st.number_input("Browsing Count",0,100,15)
    purchases = st.number_input("Purchase History",0,20,2)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    category = st.selectbox(
        "Preferred Category",
        le_cat.classes_
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -------- PREDICT --------
if st.button("Recommend Product"):

    encoded_cat = le_cat.transform([category])[0]

    data = np.array([[browsing,purchases,encoded_cat]])

    pred = model.predict(data)

    recommendation = le_rec.inverse_transform(pred)[0]

    st.success(f"🎯 Suggested Category: **{recommendation}**")
    st.balloons()

# -------- FOOTER --------
st.markdown("---")
st.caption("Random Forest Recommendation Engine")
