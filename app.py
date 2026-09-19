"""ISOM5240 Individual Lab — Storytelling App for children (ages 3–10)."""

import streamlit as st
from PIL import Image
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


def caption_image(image):
    """Convert image to RGB and return a short English caption via BLIP."""
    captioner, _, _ = load_pipelines()

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    result = captioner(image)
    if isinstance(result, list) and result:
        return (result[0].get("generated_text") or "").strip()
    if isinstance(result, dict):
        return (result.get("generated_text") or "").strip()
    return str(result).strip()
