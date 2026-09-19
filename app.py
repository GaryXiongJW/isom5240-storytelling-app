"""ISOM5240 Individual Lab — Storytelling App for children (ages 3–10)."""

import io

import numpy as np
import soundfile as sf
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


def generate_story(caption):
    """Generate a child-friendly English story (~50–100 words) from a caption."""
    _, storyteller, _ = load_pipelines()

    prompt = (
        "Write a happy short story for children aged 3 to 10. "
        "Use simple English words. No scary parts. "
        f"The story is about: {caption}. Story:\n"
    )

    min_words, max_words = 50, 100
    story = ""
    pad_token_id = getattr(storyteller.tokenizer, "eos_token_id", None)

    for attempt in range(3):
        outputs = storyteller(
            prompt,
            max_new_tokens=120 + attempt * 40,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            truncation=True,
            pad_token_id=pad_token_id,
        )
        text = outputs[0]["generated_text"]
        if text.startswith(prompt):
            story = text[len(prompt) :].strip()
        else:
            story = text.replace(prompt, "", 1).strip()

        story = " ".join(story.split())
        words = story.split()

        if len(words) > max_words:
            story = " ".join(words[:max_words])
            if not story.endswith((".", "!", "?")):
                story += "."
            words = story.split()

        if len(words) >= min_words:
            return story

    # Soft pad if the small model still undershoots (keeps ~50–100 words)
    filler = (
        "They played together and laughed under the bright sun. "
        "Everyone felt happy, kind, and safe. "
        "It was a wonderful day to remember."
    )
    words = story.split()
    while len(words) < min_words:
        story = (story + " " + filler).strip()
        words = story.split()
    if len(words) > max_words:
        story = " ".join(words[:max_words])
        if not story.endswith((".", "!", "?")):
            story += "."
    return story


def text_to_speech(story):
    """Convert story text to WAV audio bytes using Hugging Face TTS."""
    _, _, tts = load_pipelines()

    result = tts(story)
    audio = np.asarray(result["audio"]).squeeze()
    sampling_rate = int(result["sampling_rate"])

    buffer = io.BytesIO()
    sf.write(buffer, audio, sampling_rate, format="WAV")
    return buffer.getvalue()
