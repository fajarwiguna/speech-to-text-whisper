import streamlit as st
from transformers import pipeline
import torch
import io
import numpy as np
import soundfile as sf

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Speech-to-Text Whisper",
    page_icon="🗣️",
    layout="centered"
)

st.title("🗣️ Speech-to-Text dengan OpenAI Whisper")
st.write(
    "Upload audio (wav, mp3, m4a, ogg, flac) → transkripsi otomatis.\n\n"
    "**Catatan:** Model akan di-load sekali (±1–3 menit)."
)

# -----------------------------
# Load Whisper model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    with st.spinner("Loading Whisper model..."):
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=device
        )
    return pipe

pipe = load_model()

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    help="Disarankan durasi < 5 menit untuk hasil cepat"
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🎤 Mulai Transkripsi", type="primary"):
        with st.spinner("Sedang mentranskripsi..."):
            try:
                # Baca audio langsung dari bytes (tanpa ffmpeg)
                audio_bytes = uploaded_file.read()
                audio, sr = sf.read(io.BytesIO(audio_bytes))

                # Jika stereo → mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Transkripsi
                result = pipe(
                    audio,
                    generate_kwargs={
                        "language": "id",
                        "task": "transcribe"
                    }
                )

                transcription = result["text"]

                st.success("Transkripsi selesai!")
                st.subheader("📝 Hasil Transkripsi")
                st.write(transcription)

                st.download_button(
                    "📥 Download hasil (.txt)",
                    transcription,
                    file_name="transkripsi.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Terjadi error: {e}")
