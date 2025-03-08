import streamlit as st
import openai
import numpy as np
import plotly.express as px
from moviepy.editor import VideoFileClip
from scipy.spatial.distance import cosine

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

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
st.title("Multimedia Embedding Visualizer")

# Sidebar for input selection
input_type = st.sidebar.selectbox("Select Input Type", ["Text", "Audio", "Video"])

uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

if uploaded_files:
    embeddings = []
    labels = []

    for uploaded_file in uploaded_files:
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

        st.write(f"**{uploaded_file.name}**")
        st.write(text)

    # Convert embeddings to 2D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot embeddings
    fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], text=labels)
    fig.update_layout(title="Embeddings Visualization", xaxis_title="PCA Component 1", yaxis_title="PCA Component 2")
    st.plotly_chart(fig)

    # Calculate and display similarity percentages
    if len(embeddings) > 1:
        st.subheader("Similarity Percentages")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                st.write(f"Similarity between **{labels[i]}** and **{labels[j]}**: {similarity * 100:.2f}%")
