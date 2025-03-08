import streamlit as st
import openai
import numpy as np
import plotly.express as px
from moviepy.editor import VideoFileClip
from scipy.spatial.distance import cosine

# Set your OpenAI API key
openai.api_key = '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'

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

# Function to extract audio from video
def extract_audio_from_video(video_file):
    video = VideoFileClip(video_file)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# Function to get text embeddings
def get_text_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response['data'][0]['embedding']

# Streamlit app
st.title("Multimedia Similarity Checker")

# Sidebar input selection
input_type = st.sidebar.selectbox("Select Input Type", ["Text", "Audio", "Video"])

# Ask for two files
st.subheader("Upload Two Files for Comparison")
file1 = st.file_uploader("Upload First File", type=["txt", "mp3", "wav", "mp4"], key="file1")
file2 = st.file_uploader("Upload Second File", type=["txt", "mp3", "wav", "mp4"], key="file2")

if file1 and file2:
    embeddings = []
    labels = []

    for idx, uploaded_file in enumerate([file1, file2]):
        if input_type == "Text":
            text = uploaded_file.getvalue().decode("utf-8")
        elif input_type == "Audio":
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = transcribe_audio(uploaded_file.name)
        elif input_type == "Video":
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_path = extract_audio_from_video(uploaded_file.name)
            text = transcribe_audio(audio_path)

        embedding = get_text_embedding(text)
        embeddings.append(embedding)
        labels.append(uploaded_file.name)

        st.write(f"**{uploaded_file.name}** - Transcribed Text:")
        st.write(text)

    # Convert embeddings to 2D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot embeddings
    fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], text=labels)
    fig.update_layout(title="Embeddings Visualization", xaxis_title="PCA Component 1", yaxis_title="PCA Component 2")
    st.plotly_chart(fig)

    # Calculate similarity
    similarity = 1 - cosine(embeddings[0], embeddings[1])
    st.subheader(f"Similarity Percentage: **{similarity * 100:.2f}%**")
