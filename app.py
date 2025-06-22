import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load models & tools
sentiment_model_f1 = joblib.load('model-timnas-sentiment-svm-smote-best-f1.joblib')
sentiment_model_acc = joblib.load('model-timnas-sentiment-svm-smote-best-acc.joblib')
emotion_model_f1 = joblib.load('model-timnas-emotion-svm-smote-best-f1.joblib')
emotion_model_acc = joblib.load('model-timnas-emotion-svm-original-best-acc.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
le_sentiment = joblib.load('label_encoder_sentiment.joblib')
le_emotion = joblib.load('label_encoder_emotion.joblib')

# Page config
st.set_page_config(page_title="Klasifikasi Komentar Timnas", layout="wide")
# st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8KoPF-jkVKduPOCLtePC2aGHmyhIEZkHg5Q&s", use_container_width=True)
st.title("🇮🇩 Klasifikasi Komentar Timnas Indonesia - Sentiment & Emotion")

st.markdown("""
Aplikasi ini dapat memprediksi **sentimen** (positif, negatif, netral) dan **emosi** (marah, bahagia, dll.) dari komentar-komentar warganet terhadap Timnas Indonesia 🇮🇩.
""")

# Model pilihan
model_choice = st.selectbox("🧠 Pilih model berdasarkan:", ["Akurasi Tertinggi", "F1-Score Tertinggi"])

if model_choice == "Akurasi Tertinggi":
    sentiment_model = sentiment_model_acc
    emotion_model = emotion_model_acc
else:
    sentiment_model = sentiment_model_f1
    emotion_model = emotion_model_f1

# ---------------------------------------------------------
# PREDIKSI KOMENTAR TUNGGAL
st.header("📌 Prediksi Komentar Tunggal")
text_input = st.text_area("Masukkan komentar:", height=100)

if st.button("🔍 Prediksi"):
    if text_input.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        X_input = vectorizer.transform([text_input])
        pred_sentiment = le_sentiment.inverse_transform(sentiment_model.predict(X_input))[0]    
        pred_emotion = le_emotion.inverse_transform(emotion_model.predict(X_input))[0]

        st.success("✅ Hasil Prediksi:")
        st.write(f"**Sentimen**: {pred_sentiment.upper()}")
        st.write(f"**Emosi**: {pred_emotion.upper()}")

# ---------------------------------------------------------
# PREDIKSI BATCH FILE EXCEL
st.header("📂 Prediksi Batch dari File Excel")

uploaded_file = st.file_uploader("Unggah file Excel (.xlsx) berisi komentar", type=["xlsx"])
if uploaded_file:
    try:
        df_input = pd.read_excel(uploaded_file)
        if 'comment_text' not in df_input.columns:
            st.error("❌ Kolom 'comment_text' tidak ditemukan.")
        else:
            X_vec = vectorizer.transform(df_input['comment_text'])
            df_input['sentiment_pred'] = le_sentiment.inverse_transform(sentiment_model.predict(X_vec))
            df_input['emotion_pred'] = le_emotion.inverse_transform(emotion_model.predict(X_vec))

            st.success("✅ Prediksi selesai!")
            st.dataframe(df_input[['comment_text', 'sentiment_pred', 'emotion_pred']].head(10))

            # Visualisasi
            st.subheader("📊 Distribusi Hasil Prediksi")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Sentiment:")
                fig3, ax3 = plt.subplots()
                sns.countplot(y='sentiment_pred', data=df_input, order=df_input['sentiment_pred'].value_counts().index, ax=ax3)
                st.pyplot(fig3)

            with col2:
                st.write("Emotion:")
                fig4, ax4 = plt.subplots()
                sns.countplot(y='emotion_pred', data=df_input, order=df_input['emotion_pred'].value_counts().index, ax=ax4)
                st.pyplot(fig4)

            # Download hasil
            output_file = "hasil_prediksi_batch.xlsx"
            df_input.to_excel(output_file, index=False)
            with open(output_file, "rb") as f:
                st.download_button("💾 Download hasil prediksi", f, file_name=output_file)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & scikit-learn")
