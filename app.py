"""ISOM5240 Individual Lab — Storytelling App for children (ages 3–10)."""

import io
import re

import streamlit as st
from gtts import gTTS
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)

# Simple keyword gate (not an AI safety model) — blocks obvious unsafe words.
UNSAFE_KEYWORDS = {
    "kill",
    "killed",
    "killing",
    "murder",
    "blood",
    "bloody",
    "gun",
    "guns",
    "weapon",
    "weapons",
    "dead",
    "death",
    "die",
    "died",
    "dying",
    "scary",
    "scare",
    "scared",
    "horror",
    "terror",
    "violent",
    "violence",
    "war",
    "bomb",
    "knife",
    "hurt",
    "hate",
    "abuse",
    "nude",
    "naked",
    "sex",
    "drug",
    "drugs",
}


@st.cache_resource
def load_pipelines():
    """Load Hugging Face models once; reuse across Streamlit reruns.

    Caption uses BLIP (HF). Story uses distilgpt2 (HF).
    TTS uses gTTS in text_to_speech() for Streamlit Cloud RAM.
    """
    # Newer transformers removed the "image-to-text" pipeline task name,
    # so we load BLIP with Processor + Model (still Hugging Face).
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    storyteller = pipeline(
        "text-generation",
        model="distilgpt2",
    )
    return caption_processor, caption_model, storyteller


def is_kid_safe_text(text):
    """Return (True, None) if text looks kid-safe, else (False, matched_word)."""
    if not text:
        return True, None
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    for word in UNSAFE_KEYWORDS:
        if word in tokens:
            return False, word
    return True, None


def caption_image(image):
    """Convert image to RGB and return a short English caption via BLIP."""
    caption_processor, caption_model, _ = load_pipelines()

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    inputs = caption_processor(images=image, return_tensors="pt")
    output_ids = caption_model.generate(**inputs, max_new_tokens=30)
    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def generate_story(caption):
    """Generate a child-friendly English story (~50–100 words) from a caption."""
    _, _, storyteller = load_pipelines()

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
    """Convert story text to MP3 bytes using gTTS (Cloud-friendly)."""
    buffer = io.BytesIO()
    gTTS(text=story, lang="en").write_to_fp(buffer)
    return buffer.getvalue()


def main():
    """Streamlit UI only — all model work stays in the functions above."""
    st.set_page_config(
        page_title="Story Time",
        page_icon="📖",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-size: 1.15rem;
        }
        h1 { font-size: 2.4rem !important; }
        h2, h3 { font-size: 1.6rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Story Time")
    st.markdown("### Upload a picture. I will tell you a short story!")
    st.write("For children ages 3–10. Happy stories only.")
    st.caption(
        "Please use real everyday photos (park, animals, family). "
        "Do not use TV characters such as Peppa Pig."
    )

    with st.expander("Safety tips for grown-ups", expanded=False):
        st.markdown(
            """
            - Use **kind, everyday photos** (animals, park, family fun).
            - Please **do not upload** scary, violent, or private pictures.
            - Do **not** use copyrighted TV characters (for example Peppa Pig).
            - Stories are meant to be **happy and simple** — no scary themes.
            - A grown-up should stay nearby while a child uses the app.
            - The app also uses a **simple keyword gate** on captions and stories
              (not a full AI safety model).
            - First run may take a minute while models load.
            """
        )

    uploaded = st.file_uploader(
        "Choose a picture",
        type=["jpg", "jpeg", "png"],
        help="Pick a clear photo of animals, a park, or family fun.",
    )

    if uploaded is None:
        st.info("Please upload a picture to begin.")
        return

    try:
        image = Image.open(uploaded)
        image = image.convert("RGB")
    except Exception:
        st.error(
            "I could not open that file. Please try another JPG or PNG picture."
        )
        return

    st.image(image, caption="Your picture", use_container_width=True)

    if st.button("Generate Story", type="primary"):
        caption = None
        story = None

        with st.spinner(
            "Making your story... This may take a minute the first time."
        ):
            try:
                caption = caption_image(image)
            except Exception as err:
                st.error(
                    "Caption step failed. "
                    "Please wait and try again, or use a smaller JPG/PNG. "
                    f"Details: {type(err).__name__}: {err}"
                )
                return

            if not caption:
                st.warning(
                    "I could not describe that picture. "
                    "Please try a clearer photo and press Generate Story again."
                )
                return

            ok_caption, _ = is_kid_safe_text(caption)
            if not ok_caption:
                st.warning(
                    "That picture led to words that are not for little kids. "
                    "Please try a happier photo and press Generate Story again."
                )
                return

            try:
                story = generate_story(caption)
            except Exception as err:
                st.error(
                    "Story step failed. Please press Generate Story again. "
                    f"Details: {type(err).__name__}: {err}"
                )
                st.subheader("What I see")
                st.write(caption)
                return

            if not story or len(story.split()) < 50:
                st.warning(
                    "The story was too short. Please press Generate Story again."
                )
                return

            ok_story, _ = is_kid_safe_text(story)
            if not ok_story:
                st.warning(
                    "I made a story that is not gentle enough for little kids. "
                    "Please press Generate Story again, or try another picture."
                )
                return

            audio_bytes = None
            try:
                audio_bytes = text_to_speech(story)
            except Exception as err:
                st.error(
                    "Listening step failed (TTS). "
                    "Your caption and story are still shown below. "
                    f"Details: {type(err).__name__}: {err}"
                )

        st.subheader("What I see")
        st.write(caption)

        st.subheader("Your story")
        st.write(story)
        st.caption(f"Word count: {len(story.split())} (target 50–100)")

        if audio_bytes:
            st.subheader("Listen")
            st.audio(audio_bytes, format="audio/mp3")
            st.success("Done! You can upload another picture anytime.")
        else:
            st.info("Story is ready. Audio could not play this time—try Generate again.")


if __name__ == "__main__":
    main()
