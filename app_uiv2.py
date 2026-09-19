"""ISOM5240 Individual Lab — Storytelling App (submit: landscape kid UI)."""

import base64
import html
import io
import re
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)

APP_BUILD = "SUBMIT-UIv2b-20260919"

UNSAFE_KEYWORDS = {
    "kill", "killed", "killing", "murder", "blood", "bloody", "gun", "guns",
    "weapon", "weapons", "dead", "death", "died", "dying", "scary",
    "scare", "scared", "horror", "terror", "violent", "violence",
    "bomb", "knife", "hurt", "hate", "abuse", "nude", "naked", "sex",
    "drug", "drugs", "boyfriend", "girlfriend", "college", "beer", "wine",
}


@st.cache_resource
def load_pipelines():
    """Load Hugging Face models once; reuse across Streamlit reruns."""
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    storyteller = pipeline("text-generation", model="distilgpt2")
    return caption_processor, caption_model, storyteller


def is_kid_safe_text(text):
    if not text:
        return True, None
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    for word in UNSAFE_KEYWORDS:
        if word in tokens:
            return False, word
    return True, None


def _has_heavy_repetition(text):
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
    caption = " ".join((caption or "").split())
    parts = [p.strip() for p in caption.split(" and ")]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        caption = parts[0]
    if _has_heavy_repetition(caption) or len(caption.split()) < 2:
        return "happy friends playing together outdoors"
    return caption


def _looks_like_bad_story(story):
    lower = (story or "").lower()
    if not lower.strip():
        return True
    if "_" * 5 in (story or ""):
        return True
    if lower.count("story:") >= 2 or lower.count("is about:") >= 2:
        return True
    if "it?s" in lower or lower.count("epic adventure") >= 2:
        return True
    banned_bits = (
        "boyfriend", "girlfriend", "college", "simpsons", "job out of",
        "beer", "viral", "iced up", "on its own feet", "still breathing",
    )
    if any(b in lower for b in banned_bits):
        return True
    if _has_heavy_repetition(story):
        return True
    if len(re.findall(r"[.!?]", story or "")) < 2 and len((story or "").split()) > 40:
        return True
    return False


def _template_story(caption):
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
    caption_processor, caption_model, _ = load_pipelines()
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    inputs = caption_processor(images=image, return_tensors="pt")
    output_ids = caption_model.generate(
        **inputs, max_new_tokens=16, num_beams=3, no_repeat_ngram_size=2,
    )
    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
    return _clean_caption(caption.strip())


def generate_story(caption):
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
            seed, max_new_tokens=40, do_sample=True, temperature=0.8,
            top_p=0.9, repetition_penalty=1.4, no_repeat_ngram_size=3,
            truncation=True, pad_token_id=pad_token_id,
        )
        draft = " ".join(outputs[0]["generated_text"].split())
        draft = draft.replace("\u2019", "'").replace("\u2018", "'")
        draft = re.sub(r"\?s\b", "'s", draft)
    except Exception:
        draft = ""

    if _looks_like_bad_story(draft) or len(draft.split()) < min_words:
        story = _template_story(caption)
    else:
        story = draft
        words = story.split()
        if len(words) > max_words:
            story = " ".join(words[:max_words])
            if not story.endswith((".", "!", "?")):
                story += "."
        if _looks_like_bad_story(story):
            story = _template_story(caption)

    # Keyword gate inside generate_story — never return unsafe text
    ok, _ = is_kid_safe_text(story)
    if not ok:
        story = _template_story(caption)

    words = story.split()
    if len(words) > max_words:
        story = " ".join(words[:max_words])
        if not story.endswith((".", "!", "?")):
            story += "."
    return story


def text_to_speech(story):
    buffer = io.BytesIO()
    clean = story.replace("\u2019", "'").replace("\u2018", "'")
    gTTS(text=clean, lang="en").write_to_fp(buffer)
    return buffer.getvalue()


def render_karaoke_story(story, audio_bytes):
    """Show story with word highlight roughly synced to audio playback."""
    words = story.split()
    spans = [
        f'<span class="w" id="w{i}">{html.escape(w)}</span>'
        for i, w in enumerate(words)
    ]
    story_html = " ".join(spans)
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    n = len(words)
    component = f"""
    <!DOCTYPE html><html><head>
      <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;800&display=swap" rel="stylesheet">
      <style>
        body {{ margin:0; font-family:'Nunito',system-ui,sans-serif; background:transparent; color:#2b2b2b; }}
        .panel {{ background:#fff9e8; border:4px solid #ff8fab; border-radius:28px; padding:14px 16px;
          box-shadow:0 8px 0 #ffc2d4; height:340px; display:flex; flex-direction:column; }}
        .label {{ font-weight:800; color:#ff4d6d; font-size:1.05rem; margin-bottom:8px; }}
        .story {{ flex:1; overflow:auto; font-size:1.25rem; line-height:1.7; font-weight:600; }}
        .w {{ padding:1px 3px; border-radius:8px; transition:background .12s,color .12s,transform .12s; }}
        .w.on {{ background:#ffe066; color:#d00000; transform:scale(1.06); box-shadow:0 0 0 2px #ffd60a; }}
        audio {{ width:100%; margin-top:10px; }}
      </style></head><body>
      <div class="panel">
        <div class="label">Your story (words glow while reading)</div>
        <div class="story" id="story">{story_html}</div>
        <audio id="player" controls autoplay src="data:audio/mp3;base64,{b64}"></audio>
      </div>
      <script>
        const n={n}; const player=document.getElementById('player'); let last=-1;
        function paint(i){{
          if(i===last) return;
          if(last>=0){{ const p=document.getElementById('w'+last); if(p) p.classList.remove('on'); }}
          if(i>=0 && i<n){{ const c=document.getElementById('w'+i); if(c){{ c.classList.add('on'); c.scrollIntoView({{block:'nearest',behavior:'smooth'}}); }} }}
          last=i;
        }}
        player.addEventListener('timeupdate',()=>{{
          if(!player.duration||!isFinite(player.duration)) return;
          paint(Math.min(n-1, Math.floor((player.currentTime/player.duration)*n)));
        }});
        player.addEventListener('ended',()=>paint(-1));
      </script></body></html>
    """
    components.html(component, height=400, scrolling=False)


def inject_kid_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', system-ui, sans-serif !important; }
        .stApp { background: linear-gradient(135deg, #a0e9ff 0%, #ffd6ff 45%, #fff3b0 100%); overflow: hidden; }
        .block-container { padding-top: 0.6rem !important; padding-bottom: 0.4rem !important; max-width: 1200px !important; }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .hero-title { font-weight:800; font-size:2.2rem; color:#5a189a;
          text-shadow:2px 2px 0 #fff, 4px 4px 0 #ff85a1; margin:0; }
        .hero-sub { color:#7b2cbf; font-weight:700; margin:0.15rem 0 0.5rem 0; }
        .build-chip { display:inline-block; background:#fff; border:2px solid #7b2cbf;
          border-radius:999px; padding:2px 12px; font-size:0.75rem; font-weight:800; color:#5a189a; }

        /* Upload IS the fun frame — no separate grey box */
        .upload-wrap { position: relative; margin-bottom: 0.4rem; }
        div[data-testid="stFileUploader"] {
          background: transparent !important;
          border: none !important;
          padding: 0 !important;
        }
        div[data-testid="stFileUploader"] section {
          min-height: 360px !important;
          max-height: 360px !important;
          border-radius: 32px !important;
          border: 6px solid #fff !important;
          box-shadow: 0 10px 0 #ff85a1, 0 18px 30px rgba(90,24,154,0.18) !important;
          background: linear-gradient(160deg, #cdb4db, #ffc8dd 55%, #bde0fe) !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
        }
        div[data-testid="stFileUploader"] section > div {
          color: #5a189a !important;
          font-weight: 800 !important;
          font-size: 1.35rem !important;
          text-align: center !important;
        }
        /* Hide tiny file-list chrome when empty-looking; keep when file present */
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
          font-weight: 800 !important;
        }
        .pic-preview {
          height: 360px; border-radius: 32px; border: 6px solid #fff;
          box-shadow: 0 10px 0 #ff85a1, 0 18px 30px rgba(90,24,154,0.18);
          overflow: hidden; margin-bottom: 0.35rem;
        }
        .pic-preview img { width:100%; height:100%; object-fit:cover; display:block; }
        .change-hint { text-align:center; font-weight:700; color:#7b2cbf; font-size:0.85rem; margin-bottom:0.4rem; }

        .stButton > button { background:linear-gradient(90deg,#ff85a1,#ffd60a) !important;
          color:#3c096c !important; border:0 !important; border-radius:999px !important;
          font-weight:800 !important; font-size:1.15rem !important; padding:0.55rem 1.2rem !important;
          box-shadow:0 6px 0 #e85d75 !important; }
        .caption-pill { background:#fff; border-radius:18px; border:3px solid #90e0ef;
          padding:8px 12px; font-weight:700; color:#0077b6; min-height:46px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Landscape kid UI — pipeline logic stays in functions above."""
    st.set_page_config(
        page_title="Story Time", page_icon="🌈",
        layout="wide", initial_sidebar_state="collapsed",
    )
    inject_kid_theme()

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown('<p class="hero-title">Story Time Playground</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">Upload a fun photo. I tell a happy story!</p>',
            unsafe_allow_html=True,
        )
    with top_r:
        st.markdown(
            f'<div style="text-align:right;margin-top:10px;">'
            f'<span class="build-chip">Build: {APP_BUILD}</span></div>',
            unsafe_allow_html=True,
        )

    if "models_warmed" not in st.session_state:
        with st.spinner("Warming up story magic..."):
            load_pipelines()
        st.session_state.models_warmed = True

    left, right = st.columns([1.05, 1.2], gap="medium")

    with left:
        # Single upload surface = the fun frame
        uploaded = st.file_uploader(
            "give me some fun tonight",
            type=["jpg", "jpeg", "png"],
            help="Tap or drop a park / animal / family photo here",
        )

        image = None
        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
            except Exception:
                st.error("I could not open that file. Please try another JPG or PNG.")
                return
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            st.markdown(
                f'<div class="pic-preview"><img src="data:image/png;base64,{b64}" alt="uploaded" /></div>'
                f'<p class="change-hint">Want a new photo? Use the uploader above the picture.</p>',
                unsafe_allow_html=True,
            )

        gen = st.button("✨ Generate Story", use_container_width=True, type="primary")

    with right:
        caption = st.session_state.get("caption")
        story = st.session_state.get("story")
        audio_bytes = st.session_state.get("audio_bytes")

        if gen:
            if image is None:
                st.warning("Please add a picture in the fun frame first!")
            else:
                with st.spinner("Making your story..."):
                    try:
                        caption = caption_image(image)
                        ok_c, _ = is_kid_safe_text(caption)
                        if not ok_c:
                            caption = "happy friends playing together outdoors"
                        story = generate_story(caption)
                        # Always produce playable gentle story
                        ok_s, _ = is_kid_safe_text(story)
                        if not ok_s:
                            story = _template_story(caption)
                        audio_bytes = text_to_speech(story)
                        st.session_state.caption = caption
                        st.session_state.story = story
                        st.session_state.audio_bytes = audio_bytes
                    except Exception as err:
                        st.error(f"Something went wrong. Details: {type(err).__name__}: {err}")

        st.markdown("**What I see**")
        st.markdown(
            f'<div class="caption-pill">{html.escape(caption or "Waiting for a picture...")}</div>',
            unsafe_allow_html=True,
        )

        if story and audio_bytes:
            render_karaoke_story(story, audio_bytes)
            st.caption(f"Word count: {len(story.split())} (target 50–100)")
        else:
            st.markdown(
                """
                <div style="height:340px;border-radius:28px;border:4px dashed #ffb3c1;
                  background:rgba(255,255,255,0.55);display:flex;align-items:center;
                  justify-content:center;color:#9d4edd;font-weight:800;font-size:1.2rem;
                  text-align:center;padding:1rem;">
                  Your story will appear here 📖<br/>
                  <span style="font-size:0.9rem;font-weight:700;opacity:0.8;">
                    Words will glow while the story is read aloud
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
