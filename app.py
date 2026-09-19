"""ISOM5240 Individual Lab — Storytelling App for children (ages 3–10)."""

import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_pipelines():
    """Load Hugging Face pipelines once; reuse across Streamlit reruns."""
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    storyteller = pipeline(
        "text-generation",
        model="distilgpt2",
    )
    # HF TTS (not gTTS / pyttsx3) — small English model for Streamlit Cloud
    tts = pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
    )
    return captioner, storyteller, tts
