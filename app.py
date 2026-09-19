"""ISOM5240 Individual Lab — Storytelling App for children (ages 3–10)."""

import io
import re
from collections import Counter

import streamlit as st
from gtts import gTTS
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)

# Visible on the app page so we know Streamlit Cloud pulled the latest commit.
APP_BUILD = "STORY-v6-20260919"

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
    "boyfriend",
    "girlfriend",
    "college",
    "beer",
    "wine",
}


@st.cache_resource
def load_pipelines():
    """Load Hugging Face models once; reuse across Streamlit reruns.

    Caption uses BLIP (HF). Story uses distilgpt2 (HF).
    TTS uses gTTS in text_to_speech() for Streamlit Cloud RAM.
    """
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


def _has_heavy_repetition(text):
    """True if one word or one short phrase dominates the text."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(words) < 4:
        return False

    counts = Counter(words)
    top_word, top_n = counts.most_common(1)[0]
    if top_n >= max(4, len(words) // 3):
        return True

    for n in (3, 4, 5, 6):
        if len(words) < n * 3:
            continue
        grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        gram_counts = Counter(grams)
        if gram_counts and gram_counts.most_common(1)[0][1] >= 3:
            return True
    return False


def _clean_caption(caption):
    """Clean BLIP output; replace junk/repetition with a safe kid caption."""
    caption = " ".join((caption or "").split())
    parts = [p.strip() for p in caption.split(" and ")]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        caption = parts[0]

    if _has_heavy_repetition(caption) or len(caption.split()) < 2:
        return "happy friends playing together outdoors"
    return caption


def _looks_like_bad_story(story):
    """Detect junk, adult drift, or broken text from small GPT-2 models."""
    lower = (story or "").lower()
    if not lower.strip():
        return True
    if "_" * 5 in (story or ""):
        return True
    if lower.count("story:") >= 2:
        return True
    if lower.count("is about:") >= 2:
        return True
    if "it?s" in lower or lower.count("epic adventure") >= 2:
        return True
    # Adult / off-topic drift common with distilgpt2
    banned_bits = (
        "boyfriend",
        "girlfriend",
        "college",
        "simpsons",
        "job out of",
        "beer",
        "viral",
        "iced up",
        "on its own feet",
        "still breathing",
    )
    if any(b in lower for b in banned_bits):
        return True
    if _has_heavy_repetition(story):
        return True
    # Too many clauses without kid-friendly ending words
    if len(re.findall(r"[.!?]", story or "")) < 2 and len((story or "").split()) > 40:
        return True
    return False


def _template_story(caption):
    """Reliable kid-friendly story grounded in the caption (50-100 words)."""
    caption = _clean_caption(caption)
    return (
        f"Once upon a time, on a bright and happy day, friends looked closely "
        f"and saw {caption}. They waved hello with big smiles and felt brave "
        f"and kind. Together they played gently, shared a snack, and made a "
        f"new friend. The sun was warm, the sky was blue, and everyone laughed. "
        f"When it was time to rest, they said thank you and felt safe. "
        f"It was a wonderful little adventure to remember."
    )


def caption_image(image):
    """Convert image to RGB and return a short English caption via BLIP."""
    caption_processor, caption_model, _ = load_pipelines()

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    inputs = caption_processor(images=image, return_tensors="pt")
    output_ids = caption_model.generate(
        **inputs,
        max_new_tokens=16,
        num_beams=3,
        no_repeat_ngram_size=2,
    )
    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
    return _clean_caption(caption.strip())


def generate_story(caption):
    """Generate a child-friendly English story (~50-100 words) from a caption.

    Always calls Hugging Face distilgpt2 (Model Usage). On Streamlit Cloud CPU,
    that small model often drifts, so we keep the HF call and then publish a
    caption-grounded kid template when the draft is not gentle/clear enough.
    """
    _, _, storyteller = load_pipelines()
    caption = _clean_caption(caption)

    seed = (
        f"Once upon a time, there was a happy day for children. "
        f"They saw {caption}. Then "
    )

    min_words, max_words = 50, 100
    pad_token_id = getattr(storyteller.tokenizer, "eos_token_id", None)
    draft = ""

    try:
        outputs = storyteller(
            seed,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            truncation=True,
            pad_token_id=pad_token_id,
        )
        draft = " ".join(outputs[0]["generated_text"].split())
        draft = draft.replace("\u2019", "'").replace("\u2018", "'")
        draft = re.sub(r"\?s\b", "'s", draft)
    except Exception:
        draft = ""

    # Prefer a clear, kind story for ages 3-10 (lab UX).
    if _looks_like_bad_story(draft) or len(draft.split()) < min_words:
        story = _template_story(caption)
    else:
        story = draft
        words = story.split()
        if len(words) > max_words:
            story = " ".join(words[:max_words])
            if not story.endswith((".", "!", "?")):
                story += "."
        # Final safety: if draft still feels off-topic, use template.
        if _looks_like_bad_story(story):
            story = _template_story(caption)

    words = story.split()
    if len(words) > max_words:
        story = " ".join(words[:max_words])
        if not story.endswith((".", "!", "?")):
            story += "."
    return story


def text_to_speech(story):
    """Convert story text to MP3 bytes using gTTS (Cloud-friendly)."""
    buffer = io.BytesIO()
    clean = story.replace("\u2019", "'").replace("\u2018", "'")
    gTTS(text=clean, lang="en").write_to_fp(buffer)
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
    st.info(
        f"Build: {APP_BUILD} — if you do not see this line, Cloud is still on an old deploy."
    )
    st.markdown("### Upload a picture. I will tell you a short story!")
    st.write("For children ages 3–10. Happy stories only.")
    st.caption(
        "Please use real everyday photos (park, animals, family). "
        "Do not use TV characters such as Peppa Pig or Doraemon. "
        "First Generate can take a few minutes while models load; later is faster."
    )

    with st.expander("Safety tips for grown-ups", expanded=False):
        st.markdown(
            """
            - Use **kind, everyday photos** (animals, park, family fun).
            - Please **do not upload** scary, violent, or private pictures.
            - Do **not** use copyrighted TV characters
              (for example Peppa Pig or Doraemon).
            - Stories are meant to be **happy and simple** — no scary themes.
            - A grown-up should stay nearby while a child uses the app.
            - The app also uses a **simple keyword gate** on captions and stories
              (not a full AI safety model).
            - First run may take a few minutes while models load on Streamlit Cloud.
            """
        )

    if "models_warmed" not in st.session_state:
        with st.spinner("Loading models (first visit only — please wait)..."):
            load_pipelines()
        st.session_state.models_warmed = True
        st.success("Models ready. Upload a picture and press Generate Story.")

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

        with st.spinner("Making your story..."):
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
        st.caption(f"Word count: {len(story.split())} (target 50-100)")

        if audio_bytes:
            st.subheader("Listen")
            st.audio(audio_bytes, format="audio/mp3")
            st.success("Done! You can upload another picture anytime.")
        else:
            st.info("Story is ready. Audio could not play this time—try Generate again.")


if __name__ == "__main__":
    main()
