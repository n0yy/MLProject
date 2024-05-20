import streamlit as st
from utils.model import predict, load

st.title("Social Media Usage and Emotional Well-Being | ML | Random Forest Classifier")
st.caption("Mengklasifikasi Emosi Dominan pengguna berdasarkan aktifitas mereka dalam bersosial media.")

col1, col2 = st.columns(2)

age = col1.number_input("Enter your Age", min_value=5, max_value=70, step=1)
gender = col1.selectbox("Enter your Gender", ("Female", "Male", "Non-binary"))
platform = col1.selectbox("Enter the platform you often use", 
                          ("Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "Whatsapp", "Telegram"))
daily_usage = col1.number_input("Enter Daily usage (hour)", 0, 1_000, 0) * 60

daily_post = col2.number_input("Enter Posts per day", 0, 256, 0)
daily_likes = col2.number_input("Enter Likes Received per day", 0, 10_000, 0)
daily_comment = col2.number_input("Enter Comment Received per day", 0, 1_000, 0)
daily_messages = col2.number_input("Enter messages Received per day", 0, 1_000, 0)

data = {
    "Age": [age],
    "Gender": [gender],
    "Platform": [platform],
    "Daily_Usage_Time (minutes)": [daily_usage],
    "Posts_Per_Day": [daily_post],
    "Likes_Received_Per_Day": [daily_likes],
    "Comments_Received_Per_Day": [daily_comment],
    "Messages_Sent_Per_Day": [daily_messages]
}

model = load("./model/rf_model.pkl")

# Click Event
if st.button("Diagnose"):
    st.subheader("Result", divider="violet")
    
    pred, chart_data = predict(data, model)
    
    # Display it
    st.write(f"You Seem : **{pred[0]}**")
    st.write("##### With Probability")
    st.bar_chart(chart_data)
    
    st.write("by @nangdosan")
 