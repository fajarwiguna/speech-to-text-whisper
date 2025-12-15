import streamlit as st
from transformers import pipeline
import torch
import os
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Speech-to-Text Whisper", page_icon="🗣️")
st.title("🗣️ Speech-to-Text dengan OpenAI Whisper")
st.write("Upload audio (mp3, wav, m4a, ogg) → transkripsi otomatis. Support Bahasa Indonesia!")

@st.cache_resource
def load_model():
    with st.spinner("Loading model Whisper..."):
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=device
        )
    return pipe

pipe = load_model()

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    help="File maksimal ~200MB"
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Simpan ke temporary file (path string) → butuh ffmpeg untuk load non-wav
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("🎤 Mulai Transkripsi", type="primary"):
        with st.spinner("Sedang mentranskripsi... (10-90 detik)"):
            try:
                result = pipe(
                    tmp_path,  # Pass path string
                    generate_kwargs={"language": "indonesian", "task": "transcribe"},
                    chunk_length_s=30,
                    batch_size=8
                )
                transcription = result["text"]

                st.success("Selesai!")
                st.subheader("Hasil Transkripsi:")
                st.write(transcription)

                st.download_button(
                    "📥 Download Teks (.txt)",
                    transcription,
                    file_name="transkripsi.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Hapus tmp file
        os.unlink(tmp_path)
