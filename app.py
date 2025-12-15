import streamlit as st
from transformers import pipeline
import torch
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Speech-to-Text Whisper", page_icon="🗣️")
st.title("🗣️ Speech-to-Text dengan OpenAI Whisper")
st.write("Upload file audio (mp3, wav, m4a, ogg) → langsung transkripsi. Support Bahasa Indonesia!")

# Load model (cache)
@st.cache_resource
def load_model():
    with st.spinner("Loading model Whisper... "):
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",  
            device=device
        )
    return pipe

pipe = load_model()

# Upload file
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    help="Maksimal ukuran file ~200MB"
)

if uploaded_file is not None:
    # Tampilkan audio player
    st.audio(uploaded_file.read(), format='audio/wav')  # Reset pointer setelah read
    uploaded_file.seek(0)  # Reset pointer ke awal

    if st.button("🎤 Mulai Transkripsi", type="primary"):
        with st.spinner("Sedang mentranskripsi... (bisa 10-90 detik tergantung panjang audio)"):
            try:
                # Baca file sebagai bytes langsung
                audio_bytes = uploaded_file.read()

                # Pipeline Whisper bisa terima bytes langsung!
                result = pipe(
                    audio_bytes,
                    generate_kwargs={"language": "indonesian", "task": "transcribe"},
                    chunk_length_s=30,
                    batch_size=8
                )
                transcription = result["text"]

                st.success("Transkripsi selesai!")
                st.subheader("Hasil Transkripsi:")
                st.write(transcription)

                # Download teks
                st.download_button(
                    label="📥 Download Teks (.txt)",
                    data=transcription,
                    file_name=f"transkripsi_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Terjadi error: {str(e)}")
                st.write("Coba file audio lain atau ukuran lebih kecil.")

# Footer
st.caption("Dibuat dengan 🤗 Hugging Face + OpenAI Whisper")
