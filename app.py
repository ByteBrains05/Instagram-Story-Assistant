
import streamlit as st
from PIL import Image
import torch
import clip
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import random

st.set_page_config(page_title="Instagram Story Assistant", layout="centered")
st.title("📸 Instagram Story Assistant 🎶")

# Load caption dataset
@st.cache_data
def load_captions():
    df = pd.read_csv("final_instagram_caption_dataset.csv")  # Relative path
    return df.groupby("Label")["Caption"].apply(list).to_dict()

caption_dict = load_captions()

labels = [
    "love", "funny", "friends", "winter", "music", "fashion", "food", "family", "travel",
    "beach", "christmas", "motivational", "holiday", "new_year", "inspirational", "pets",
    "baddie", "wedding", "art", "nature", "seasonal", "spring", "sassy", "general",
    "positive", "fitness", "short", "top", "books", "birthday", "saree", "cat", "dog",
    "selfie", "group", "girls", "sunset", "sky", "study", "pool", "bikini", "flowers", "quotes"
]

# Load CLIP model (cached)
@st.cache_resource
def load_clip_model():
    return clip.load("RN50", device="cpu")  # use "cpu" to reduce memory

model, preprocess = load_clip_model()

# Spotify setup
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_spotify_songs(query):
    try:
        results = sp.search(q=query + " vibes", type="track", limit=3)
        return results["tracks"]["items"]
    except Exception as e:
        st.error(f"Spotify error: {e}")
        return []

# Session setup
if "shown_captions" not in st.session_state:
    st.session_state.shown_captions = []
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None
if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
manual_input = st.text_input("Or type a theme (e.g., love, music, winter)")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_hash = hash(image.tobytes())
    if st.session_state.last_image_hash != image_hash:
        st.session_state.last_image_hash = image_hash

        image_input = preprocess(image).unsqueeze(0).to("cpu")
        text_inputs = clip.tokenize(labels).to("cpu")
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).squeeze(0)
        best_label_idx = similarity.argmax().item()
        st.session_state.predicted_label = labels[best_label_idx]

        label = st.session_state.predicted_label
        if label in caption_dict:
            st.session_state.shown_captions = random.sample(
                caption_dict[label], min(3, len(caption_dict[label]))
            )
        else:
            st.session_state.shown_captions = ["No captions found for this theme."]

    st.success(f"🎯 Predicted Theme: **{st.session_state.predicted_label}**")

    if st.button("🔁 Regenerate Captions"):
        label = st.session_state.predicted_label
        if label in caption_dict:
            st.session_state.shown_captions = random.sample(
                caption_dict[label], min(3, len(caption_dict[label]))
            )

    st.subheader("📝 Aesthetic Captions")
    for cap in st.session_state.shown_captions:
        st.code(cap)  # shows clean caption with copy button

    st.subheader("🎵 Spotify Songs")
    songs = get_spotify_songs(st.session_state.predicted_label)
    for song in songs:
        st.markdown(f"**{song['name']}** by *{song['artists'][0]['name']}*")
        st.image(song['album']['images'][0]['url'], width=120)
        st.markdown(f"[▶️ Listen on Spotify]({song['external_urls']['spotify']})")
        st.markdown("---")

elif manual_input:
    st.success(f"🔹 Using your theme: **{manual_input}**")

    if st.button("🔁 Regenerate Captions"):
        if manual_input in caption_dict:
            st.session_state.shown_captions = random.sample(
                caption_dict[manual_input], min(3, len(caption_dict[manual_input]))
            )
        else:
            st.session_state.shown_captions = ["No captions found."]

    st.subheader("📝 Aesthetic Captions")
    for cap in st.session_state.shown_captions:
        st.code(cap)

    st.subheader("🎵 Spotify Songs")
    songs = get_spotify_songs(manual_input)
    for song in songs:
        st.markdown(f"**{song['name']}** by *{song['artists'][0]['name']}*")
        st.image(song['album']['images'][0]['url'], width=120)
        st.markdown(f"[▶️ Listen on Spotify]({song['external_urls']['spotify']})")
        st.markdown("---")

else:
    st.info("📤 Upload a photo or enter a theme to get started.")
