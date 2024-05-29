import streamlit as st
from src.model import predict, load
from src.llm import make_prompt, get_summary

st.set_page_config(
    page_title="Psychobot AI",
    page_icon="random"
)

st.image("src/assets/LOGO.png")
st.title("KENALI EMOSIMU SAAT BER-MEDIA SOSIAL")
st.caption("Psychobot adalah sebuah sistem AI yang membantu kamu mengenali kondisi emosi dominan berdasarkan aktivitasmu di media sosial.")
st.text("")
st.text("")
st.write("#### # Coba Sekarang dengan mengisi form dibawah ini :writing_hand:")
st.text("")

col1, col2 = st.columns(2)

age = col1.number_input("Umur", min_value=5, max_value=70, step=1)
gender = col1.selectbox("Jenis Kelamin", ("Female", "Male", "Non-binary"))
platform = col1.selectbox("Sosial Media yang digunakan", 
                          ("Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "Whatsapp", "Telegram"))
daily_usage = col1.number_input("Masukkan penggunaan harian (jam)", 0, 1_000, 0) * 60

daily_post = col2.number_input("Masukkan Jumlah Postingan per hari", 0, 256, 0)
daily_likes = col2.number_input("Masukkan Jumlah Suka yang Diterima per hari", 0, 10_000, 0)
daily_comment = col2.number_input("Masukkan Jumlah Komentar yang Diterima per hari", 0, 1_000, 0)
daily_messages = col2.number_input("Masukkan Jumlah Pesan yang Diterima per hari", 0, 1_000, 0)



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

model = load("./model/lgbm-v.0.1.0.pkl")

# Click Event
st.text("")
if st.button("Analisis Sekarang!", type="primary"):
    st.subheader("Hasil", divider="violet")
    
    pred, chart_data = predict(data, model)
    
    # Display it
    st.write(f"Kamu terlihat : **{pred[0]}**")
    st.write("##### Statistik")
    st.bar_chart(chart_data, x="Emotional", y="Probability")
    
    with st.spinner("Tunggu sebentar ..."):
        prompt = make_prompt(chart_data)
        md = get_summary(prompt)

    st.markdown(md)  