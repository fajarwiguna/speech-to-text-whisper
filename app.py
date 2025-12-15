import streamlit as st
from transformers import pipeline
import torch
import os
from tempfile import NamedTemporaryFile
import ffmpeg

st.set_page_config(page_title="Speech-to-Text Whisper", page_icon="🗣️")
st.title("🗣️ Speech-to-Text dengan OpenAI Whisper")
st.write("Upload file audio (mp3, wav, m4a) dan dapatkan transkripsi teks. Support Bahasa Indonesia!")

# Load model 
@st.cache_resource
def load_model():
    with st.spinner("Loading model Whisper-small... (pertama kali agak lama)"):
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",  
            device=device
        )
    return pipe

pipe = load_model()

# Upload audio
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "ogg"],
    help="Format yang didukung: wav, mp3, m4a, ogg"
)

if uploaded_file is not None:
    # Tampilkan audio
    st.audio(uploaded_file, format="audio/wav")

    # Simpan ke temporary file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("🎤 Mulai Transkripsi", type="primary"):
        with st.spinner("Sedang mentranskripsi... (bisa 10-60 detik tergantung panjang audio)"):
            try:
                # Transkripsi dengan bahasa Indonesia 
                result = pipe(
                    tmp_path,
                    generate_kwargs={"language": "indonesian", "task": "transcribe"},
                    chunk_length_s=30,
                    batch_size=8
                )
                transcription = result["text"]

                st.success("Transkripsi selesai!")
                st.subheader("Hasil Transkripsi:")
                st.write(transcription)

                # Tombol download teks
                st.download_button(
                    label="📥 Download Teks (.txt)",
                    data=transcription,
                    file_name="transkripsi.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Hapus file temporary
        os.unlink(tmp_path)

# Footer
st.caption("Dibuat dengan 🤗 Hugging Face Transformers + OpenAI Whisper | Untuk tugas akhir NLP")
