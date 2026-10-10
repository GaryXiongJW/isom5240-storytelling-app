"""ISOM5240 Individual Lab — Storytelling App (submit: landscape kid UI)."""

import base64
import html
import io
import logging
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

APP_BUILD = "SUBMIT-STORY-v8-20261010"

UNSAFE_KEYWORDS = {
    "kill", "killed", "killing", "murder", "blood", "bloody", "gun", "guns",
    "weapon", "weapons", "dead", "death", "died", "dying", "scary",
    "scare", "scared", "horror", "terror", "violent", "violence",
    "bomb", "knife", "hurt", "hate", "abuse", "nude", "naked", "sex",
    "drug", "drugs", "boyfriend", "girlfriend", "college", "beer", "wine",
}

# Phrases distilgpt2 actually produced in earlier builds. Kept as a backstop
# after the per-sentence filter, because a tiny base model will still wander.
BANNED_BITS = (
    "boyfriend", "girlfriend", "college", "simpsons", "job out of",
    "beer", "viral", "iced up", "on its own feet", "still breathing",
    "iced tea", "chocolate milk", "came to their mouths",
    "received this gift", "those who had eaten", "from that corner",
)

OFF_TOPIC = ("tea", "coffee", "milk", "wine", "pizza", "burger", "salary")

# Distinct closers so a short-but-clean draft can reach 50 words without
# repeating a 3-word chunk (that would trip the repetition gate).
_CLOSERS = (
    "They used kind words, took turns, and felt safe together.",
    "Everyone smiled softly and stayed close until rest time.",
    "The day ended with a warm wave and a happy heart.",
    "Friends shared a slow game and looked after one another.",
    "Soft laughter stayed with them as the bright day grew quiet.",
)

# First matching place wins, so specific scenes stay ahead of "park".
# Each row is keywords, preposition, place, ground, air.
_SETTINGS = (
    (("beach", "ocean", "sea", "sand", "shore", "wave", "waves"), "at", "a sunny beach", "the warm sand", "the soft waves"),
    (("snow", "winter", "ski"), "in", "the snow", "the white snow", "the quiet air"),
    (("forest", "woods", "tree", "trees", "jungle"), "in", "a green forest", "the tall trees", "the leafy shade"),
    (("kitchen", "cake", "pizza", "cookie", "cupcake", "food"), "in", "a cozy kitchen", "the tidy table", "the warm room"),
    (("night", "moon", "star", "stars"), "on", "a calm evening", "the twinkling stars", "the kind moon"),
    (("mountain", "hill"), "on", "a gentle hill", "the open path", "a cool breeze"),
    (("lake", "river", "boat", "ship", "pool"), "by", "a calm lake", "the still water", "a light breeze"),
    (("sky", "cloud", "clouds", "balloon", "kite", "plane", "airplane"), "under", "a wide blue sky", "the light clouds", "a happy breeze"),
    (("street", "road", "city", "bus", "traffic", "sidewalk", "truck"), "on", "a friendly street", "the sunny sidewalk", "the bright day"),
    (("room", "bed", "sofa", "couch", "pillow", "indoor"), "in", "a cozy room", "the soft rug", "the quiet house"),
    (("park", "garden", "grass", "playground", "outdoor", "outdoors", "yard"), "at", "a sunny park", "the soft grass", "the open sky"),
)
_DEFAULT_SETTING = ("in", "a bright happy place", "the open space", "the gentle light")
_PEOPLE = (
    "boy", "girl", "child", "children", "kid", "kids", "baby",
    "family", "people", "person", "man", "woman", "group", "friend", "friends",
)

_SUBJECTS = (
    (("puppy", "dog"), "The dog wagged its tail and waited kindly."),
    (("kitten", "cat"), "The cat blinked slowly and purred."),
    (("bird", "birds"), "The bird sang a tiny happy song."),
    (("duck", "ducks"), "The duck paddled in a small happy line."),
    (("horse", "pony"), "The horse walked slowly and nodded."),
    (("rabbit", "bunny"), "The rabbit hopped in a tiny happy circle."),
    (("teddy", "bear"), "The bear gave a careful, friendly wave."),
    (("fish",), "The fish swished its tail and sparkled."),
    (("baby",), "The baby clapped and made a happy sound."),
    (("boy", "girl", "child", "children", "kid", "kids"), "The children held hands and giggled."),
    (("family", "people", "person", "man", "woman", "group", "friend", "friends"), "They smiled and stayed close together."),
    (("flower", "flowers"), "The flowers nodded in the light."),
    (("ball",), "The ball rolled slowly across the ground."),
    (("cupcake", "cake"), "Everyone looked at the cake and smiled."),
    (("bus",), "The bus waited calmly at the stop."),
    (("truck", "car"), "The car rested quietly for a moment."),
    (("ship", "boat"), "The boat rocked gently and stayed safe."),
    (("bicycle", "bike"), "The bike waited for a slow, happy ride."),
)
_DEFAULT_ACTION = "Everyone looked around with a kind smile."


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


def _has_word(text, word):
    return re.search(rf"\b{re.escape(word)}\b", (text or "").lower()) is not None


def _matches_any(text, words):
    return any(_has_word(text, word) for word in words)


def _pick_setting(caption):
    for keys, prep, place, ground, air in _SETTINGS:
        if _matches_any(caption, keys):
            return prep, place, ground, air
    return _DEFAULT_SETTING


def _lead(caption, variant):
    """Open in a way that stays grammatical for people and for objects."""
    if _matches_any(caption, _PEOPLE):
        if variant == 0:
            return f"Once upon a time, this happy story began with {caption}."
        if variant == 1:
            return f"One day, this happy story began with {caption}."
        return f"Once upon a time, on a bright day, this story began with {caption}."
    if variant == 0:
        return f"Once upon a time, children found {caption}."
    if variant == 1:
        return f"One day, a happy group noticed {caption}."
    return f"Once upon a time, there was a bright day with {caption}."


def _pick_action(caption):
    for keys, action in _SUBJECTS:
        if _matches_any(caption, keys):
            return action
    return _DEFAULT_ACTION


def _variant(caption):
    return sum(ord(char) for char in (caption or "").lower()) % 3


def _word_tokens(text):
    return set(re.findall(r"[a-z']+", (text or "").lower()))


def _sentence_ok(sentence, caption):
    """Keep one model sentence only when it is short, safe, and on topic."""
    text = " ".join((sentence or "").split())
    if not text or re.search(r"https?://|www\.|_{3,}|[{}]", text):
        return False
    words = text.split()
    if len(words) < 4 or len(words) > 32:
        return False
    lower = text.lower()
    if any(bit in lower for bit in BANNED_BITS):
        return False
    if lower.count("story:") or lower.count("is about:"):
        return False
    ok, _ = is_kid_safe_text(text)
    if not ok or _has_heavy_repetition(text):
        return False
    tokens = _word_tokens(text)
    if any(word in tokens for word in OFF_TOPIC):
        caption_tokens = _word_tokens(caption)
        if not any(word in caption_tokens for word in OFF_TOPIC):
            return False
    return True


def _trim_to_max(story, max_words):
    words = story.split()
    if len(words) <= max_words:
        return story
    trimmed = " ".join(words[:max_words])
    ends = list(re.finditer(r"[.!?]", trimmed))
    if ends:
        candidate = trimmed[: ends[-1].end()].strip()
        if len(candidate.split()) >= 50:
            return candidate
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    return trimmed


def _fit_word_count(story, min_words=50, max_words=100):
    """Pad with distinct closers or cut on a sentence boundary."""
    story = " ".join((story or "").split())
    if not story:
        return ""
    closer_i = 0
    while len(story.split()) < min_words and closer_i < len(_CLOSERS):
        if not story.endswith((".", "!", "?")):
            story += "."
        story = f"{story} {_CLOSERS[closer_i]}"
        closer_i += 1
    if len(story.split()) > max_words:
        story = _trim_to_max(story, max_words)
    return story


def _looks_like_bad_story(story, caption=""):
    """Reject nonsense / off-topic / mid-seed drafts from tiny LMs."""
    text = " ".join((story or "").split())
    lower = text.lower()
    if not lower.strip():
        return True
    if "_" * 5 in text:
        return True
    if lower.count("story:") >= 2 or lower.count("is about:") >= 2:
        return True
    if "it?s" in lower or lower.count("epic adventure") >= 2:
        return True
    # Mid-seed leftovers like "day for children. They saw..."
    if not re.match(r"^(once|one day|on a|there was|long ago)", lower):
        return True
    if any(bit in lower for bit in BANNED_BITS):
        return True
    if _has_heavy_repetition(text):
        return True
    if len(re.findall(r"[.!?]", text)) < 3:
        return True
    # Stay grounded in the caption
    stop = {
        "with", "their", "this", "that", "from", "they", "were",
        "have", "been", "into", "over", "under", "about", "group",
        "them", "then", "when", "what", "your", "some",
    }
    cap_words = [
        w for w in re.findall(r"[a-z]+", (caption or "").lower())
        if len(w) > 3 and w not in stop
    ]
    if cap_words and not any(w in lower for w in cap_words[:6]):
        return True
    tokens = _word_tokens(lower)
    if any(word in tokens for word in OFF_TOPIC):
        caption_tokens = _word_tokens(caption)
        if not any(word in caption_tokens for word in OFF_TOPIC):
            return True
    return False


def _template_story(caption):
    """Kid-friendly story whose place and character follow the caption.

    The old fallback always mentioned cool green grass, so a bus, a sofa,
    or a beach still became a park. The scene lexicons below keep the
    read-aloud in the same place as the picture.
    """
    caption = _clean_caption(caption)
    action = _pick_action(caption)
    prep, place, ground, air = _pick_setting(caption)
    where = f"{prep} {place}"
    variant = _variant(caption)
    lead = _lead(caption, variant)
    if variant == 0:
        story = (
            f"{lead} "
            f"{action} "
            f"They spent a gentle day {where}. "
            f"Friends stayed close, took turns, and used kind small words. "
            f"They liked {ground} and smiled at {air}. "
            f"Nobody was in a hurry. Everyone felt safe and glad. "
            f"When the day grew quiet, they waved goodbye. "
            f"It was a sweet adventure they would always remember."
        )
    elif variant == 1:
        story = (
            f"{lead} "
            f"{action} "
            f"They played slowly {where} and shared the fun. "
            f"The little ones looked around with wide, curious eyes. "
            f"{ground.capitalize()} felt nice, and {air} made them laugh softly. "
            f"They helped each other and said please and thank you. "
            f"At the end, every heart felt warm and proud. "
            f"It was a kind day they would remember."
        )
    else:
        story = (
            f"{lead} "
            f"{action} "
            f"{prep.capitalize()} {place}, everyone moved gently and stayed together. "
            f"They counted to three, then took a happy turn. "
            f"They noticed {ground} and enjoyed {air}. "
            f"Soft giggles filled the moment, and nobody felt left out. "
            f"When rest time came, they held a long friendly wave. "
            f"It was a calm, happy story to keep."
        )
    return _fit_word_count(story)


def _split_sentences(text):
    text = " ".join((text or "").split()).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for part in parts:
        part = part.strip(" ,;:-")
        if not part:
            continue
        if not re.search(r"[.!?]$", part):
            part += "."
        sentences.append(part)
    return sentences


def _bridge_first(sentence):
    """Attach the seed's unfinished clause so the first line stays grammatical."""
    sentence = sentence.strip()
    if sentence.lower().startswith("the children smiled and "):
        return sentence
    if sentence[:1].isalpha() and sentence[0].isupper():
        sentence = sentence[0].lower() + sentence[1:]
    return f"The children smiled and {sentence}"


def _polish_hf_draft(raw, seed, caption):
    """Keep a few clean continuation sentences; drop the rest of the draft."""
    text = " ".join((raw or "").split())
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\?s\b", "'s", text)
    seed_clean = " ".join(seed.split())
    if text.lower().startswith(seed_clean.lower()):
        cont = text[len(seed_clean):].strip()
    else:
        match = re.search(r"(?:smiled and |Then |then )(.+)$", text)
        cont = match.group(1).strip() if match else ""
    cont = cont.strip(" ,;:-")
    cont = re.sub(r"^(?:and|then)\s+", "", cont, flags=re.I)
    if len(cont.split()) < 4:
        return ""
    parts = _split_sentences(cont)
    if not parts:
        return ""
    parts[0] = _bridge_first(parts[0])
    kept = []
    for sentence in parts:
        if not _sentence_ok(sentence, caption):
            break
        kept.append(sentence)
        if len(kept) == 2:
            break
    if not kept:
        return ""
    story = (
        f"Once upon a time, there was a happy day for children. "
        f"They saw {caption}. "
        + " ".join(kept)
    )
    return " ".join(story.split())


def _sample_continuation(storyteller, seed):
    pad_token_id = getattr(storyteller.tokenizer, "eos_token_id", None)
    try:
        outputs = storyteller(
            seed,
            max_new_tokens=55,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            truncation=True,
            pad_token_id=pad_token_id,
        )
        return outputs[0]["generated_text"]
    except Exception:
        logging.getLogger("storytime").exception("distilgpt2 sample failed")
        return ""


def caption_image(image):
    """Hugging Face BLIP caption; returns a short cleaned English caption."""
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


def _usable_draft(draft, caption):
    if not draft or _looks_like_bad_story(draft, caption):
        return ""
    ok, _ = is_kid_safe_text(draft)
    if not ok:
        return ""
    fitted = _fit_word_count(draft)
    if not fitted or _looks_like_bad_story(fitted, caption):
        return ""
    ok, _ = is_kid_safe_text(fitted)
    if not ok:
        return ""
    return fitted


def generate_story(caption):
    """One distilgpt2 sample, then a caption-true story if that sample is unusable.

    distilgpt2 is a tiny base model, so most free-run text fails the safety
    gate. We keep at most two clean sentences from its continuation. If none
    survive, the fallback names the same place and character as the caption
    instead of pasting one park paragraph onto every photo.
    """
    _, _, storyteller = load_pipelines()
    caption = _clean_caption(caption)
    seed = (
        f"Once upon a time, there was a happy day for children. "
        f"They saw {caption}. The children smiled and "
    )
    draft = _polish_hf_draft(_sample_continuation(storyteller, seed), seed, caption)
    story = _usable_draft(draft, caption)
    if story:
        return story

    story = _template_story(caption)
    if _looks_like_bad_story(story, caption) or not is_kid_safe_text(story)[0]:
        story = _template_story("happy friends playing together outdoors")
    return _fit_word_count(story)


def text_to_speech(story):
    buffer = io.BytesIO()
    clean = story.replace("\u2019", "'").replace("\u2018", "'")
    gTTS(text=clean, lang="en").write_to_fp(buffer)
    return buffer.getvalue()


def render_karaoke_story(story, audio_bytes, caption=""):
    """Fixed-size story panel; auto-scroll words while audio plays."""
    words = story.split()
    spans = [
        f'<span class="w" id="w{i}">{html.escape(w)}</span>'
        for i, w in enumerate(words)
    ]
    story_html = " ".join(spans)
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    n = len(words)
    seen = ""
    if caption:
        safe_caption = html.escape(caption)
        seen = f'<div class="seen">I see: {safe_caption}</div>'
    # FRAME_H must match .pic-frame / .story-empty (380px)
    frame_h = 380
    component = f"""
    <!DOCTYPE html><html><head>
      <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@700;900&display=swap" rel="stylesheet">
      <style>
        html, body {{ margin:0; padding:0; height:{frame_h}px; overflow:hidden;
          font-family:'Nunito',system-ui,sans-serif; background:transparent; color:#2b2b2b; }}
        .panel {{ background:#fff9e8; border:6px solid #ff8fab; border-radius:32px;
          box-sizing:border-box; height:{frame_h}px; width:100%;
          padding:12px 14px 10px 14px; box-shadow:0 10px 0 #ffc2d4;
          display:flex; flex-direction:column; }}
        .labelrow {{ display:flex; align-items:center; justify-content:space-between;
          gap:8px; flex:0 0 auto; margin:0 0 2px 0; }}
        .label {{ font-weight:900; color:#ff4d6d; font-size:1.05rem; margin:0; }}
        .seen {{ font-weight:800; color:#7b2cbf; font-size:0.82rem; line-height:1.25;
          margin:0 0 6px 0; overflow:hidden; display:-webkit-box;
          -webkit-line-clamp:2; -webkit-box-orient:vertical; }}
        .dl {{ flex:0 0 auto; font-size:0.78rem; font-weight:900; color:#fff;
          background:#7b2cbf; border-radius:999px; padding:3px 10px; text-decoration:none; }}
        .story {{ flex:1 1 auto; min-height:0; overflow-y:auto; overflow-x:hidden;
          font-size:1.3rem; line-height:1.7; font-weight:700; scroll-behavior:smooth; }}
        .w {{ padding:1px 3px; border-radius:8px; transition:background .12s,color .12s,transform .12s; }}
        .w.on {{ background:#ffe066; color:#d00000; transform:scale(1.06); box-shadow:0 0 0 2px #ffd60a; }}
        audio {{ width:100%; height:32px; margin-top:8px; flex:0 0 auto; }}
      </style></head><body>
      <div class="panel">
        <div class="labelrow">
          <div class="label">Your story</div>
          <a class="dl" href="#" id="dl">Download</a>
        </div>
        {seen}
        <div class="story" id="story">{story_html}</div>
        <audio id="player" controls autoplay src="data:audio/mp3;base64,{b64}"></audio>
      </div>
      <script>
        const n={n}; const player=document.getElementById('player');
        const box=document.getElementById('story'); let last=-1;
        function paint(i){{
          if(i===last) return;
          if(last>=0){{ const p=document.getElementById('w'+last); if(p) p.classList.remove('on'); }}
          if(i>=0 && i<n){{
            const c=document.getElementById('w'+i);
            if(c){{
              c.classList.add('on');
              // Keep highlighted word near vertical center of story box
              const top = c.offsetTop - (box.clientHeight / 2) + (c.offsetHeight / 2);
              box.scrollTo({{ top: Math.max(0, top), behavior: 'smooth' }});
            }}
          }}
          last=i;
        }}
        player.addEventListener('timeupdate',()=>{{
          if(!player.duration||!isFinite(player.duration)) return;
          paint(Math.min(n-1, Math.floor((player.currentTime/player.duration)*n)));
        }});
        player.addEventListener('ended',()=>paint(-1));
        document.getElementById('dl').addEventListener('click', (e) => {{
          e.preventDefault();
          const a = document.createElement('a');
          a.href = player.src;
          a.download = 'story.mp3';
          a.click();
        }});
      </script></body></html>
    """
    # iframe height = frame only (no extra gap under the story box)
    components.html(component, height=frame_h, scrolling=False)



def inject_kid_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;900&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', system-ui, sans-serif !important; }
        .stApp { background: linear-gradient(135deg, #a0e9ff 0%, #ffd6ff 45%, #fff3b0 100%); overflow: hidden; }
        .block-container { padding-top: 0.4rem !important; padding-bottom: 0.3rem !important; max-width: 1180px !important; }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        .hero-wrap { text-align: center; margin: 0.2rem 0 0.85rem 0; }
        .hero-title { font-weight:900; font-size:3.1rem; line-height:1.1; color:#5a189a;
          text-shadow:3px 3px 0 #fff, 6px 6px 0 #ff85a1; margin:0; letter-spacing:-0.02em; }
        .hero-sub { color:#7b2cbf; font-weight:800; font-size:1.35rem; margin:0.45rem 0 0 0; }
        .build-chip { display:inline-block; background:#fff; border:2px solid #7b2cbf;
          border-radius:999px; padding:2px 12px; font-size:0.7rem; font-weight:800; color:#5a189a;
          margin-top:0.45rem; }

        /* Fixed picture frame (left) */
        .pic-frame {
          height: 380px; min-height: 380px; max-height: 380px;
          width: 100%; box-sizing: border-box;
          border-radius: 32px; border: 6px solid #fff;
          box-shadow: 0 10px 0 #ff85a1, 0 18px 30px rgba(90,24,154,0.18);
          overflow: hidden; margin-bottom: 0.55rem;
          background: linear-gradient(160deg, #cdb4db, #ffc8dd 55%, #bde0fe);
          display: flex; align-items: center; justify-content: center;
        }
        .pic-frame img { width:100%; height:100%; object-fit:cover; display:block; }
        .pic-empty { text-align:center; color:#5a189a; font-weight:900; font-size:1.35rem;
          padding:1rem; text-shadow:1px 1px 0 #fff; }

        /* Twin action pills under LEFT pic frame only — equal width, no overlap */
        .gen-status {
          text-align:center; font-weight:900; color:#5a189a; font-size:0.9rem;
          background:rgba(255,255,255,0.85); border-radius:999px; padding:0.3rem 0.7rem;
          margin:0.35rem 0 0 0; border:2px solid #ff85a1;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
          overflow: hidden !important;
        }
        .stButton { width: 100% !important; }
        .stButton > button {
          background: linear-gradient(90deg,#ff85a1,#ffd60a) !important;
          color:#3c096c !important; border:0 !important; border-radius:999px !important;
          font-weight:900 !important; font-size:1.05rem !important;
          height: 3.2rem !important; min-height: 3.2rem !important; max-height: 3.2rem !important;
          padding: 0 0.6rem !important;
          box-shadow: 0 6px 0 #e85d75 !important; margin: 0 !important;
          width: 100% !important;
        }
        /* Upload = blue twin pill (match Generate size) */
        div[data-testid="stFileUploader"] {
          background: transparent !important; border: none !important; padding: 0 !important;
          margin: 0 !important;
        }
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"] {
          min-height: 3.2rem !important; max-height: 3.2rem !important; height: 3.2rem !important;
          border-radius: 999px !important; border: 0 !important;
          box-shadow: 0 6px 0 #4895ef !important;
          background: linear-gradient(90deg, #4cc9f0, #4361ee) !important;
          display: flex !important; align-items: center !important; justify-content: center !important;
          padding: 0 0.5rem !important; overflow: hidden !important;
          gap: 0 !important;
        }
        /* Hide native "Browse files" / limit chrome; show one label */
        div[data-testid="stFileUploaderDropzoneInstructions"] {
          display: flex !important; align-items: center !important; justify-content: center !important;
          width: 100% !important;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] > div {
          display: flex !important; flex-direction: row !important; align-items: center !important;
          justify-content: center !important; gap: 0.35rem !important;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] span,
        div[data-testid="stFileUploaderDropzoneInstructions"] small,
        div[data-testid="stFileUploaderDropzoneInstructions"] p {
          font-size: 0 !important; line-height: 0 !important; color: transparent !important;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
          content: "📷 Upload Picture";
          font-size: 1.05rem !important; line-height: 1.2 !important;
          font-weight: 900 !important; color: #fff !important;
          white-space: nowrap;
        }
        div[data-testid="stFileUploader"] button {
          display: none !important;
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
          background: rgba(255,255,255,0.9) !important;
          border-radius: 14px !important;
          margin-top: 0.4rem !important;
          max-width: 100% !important;
        }

        .story-empty {
          height:380px; min-height:380px; max-height:380px; width:100%;
          box-sizing:border-box; border-radius:32px; border:6px solid #ff8fab;
          background:#fff9e8; display:flex; align-items:center;
          justify-content:center; color:#9d4edd; font-weight:900; font-size:1.35rem;
          text-align:center; padding:1.2rem; box-shadow:0 10px 0 #ffc2d4;
        }
        /* Keep iframe / empty story slot from shifting layout */
        iframe { border: none !important; }
        [data-testid="stVerticalBlockBorderWrapper"] { overflow: visible; }
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

    st.markdown(
        f"""
        <div class="hero-wrap">
          <p class="hero-title">Story Time Playground</p>
          <p class="hero-sub">Upload a fun photo. I tell a happy story!</p>
          <span class="build-chip">Build: {APP_BUILD}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "models_warmed" not in st.session_state:
        with st.spinner("Warming up story magic..."):
            load_pipelines()
        st.session_state.models_warmed = True

    left, right = st.columns(2, gap="large")

    with left:
        # 1) Fixed picture frame (always same size)
        image = st.session_state.get("image_pil")
        if image is not None:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            st.markdown(
                f'<div class="pic-frame"><img src="data:image/png;base64,{b64}" alt="uploaded" /></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pic-frame"><div class="pic-empty">'
                "Your photo goes here ✨<br/>"
                '<span style="font-size:0.95rem;opacity:0.85;">use Upload Picture below</span>'
                "</div></div>",
                unsafe_allow_html=True,
            )

        # 2+3) Under LEFT frame only: Generate | Upload side-by-side (equal)
        mid_l, mid_r = st.columns(2, gap="medium")
        with mid_l:
            gen = st.button("✨ Generate Story", use_container_width=True, type="primary")
        with mid_r:
            uploaded = st.file_uploader(
                "Upload Picture",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                help="Choose a park / animal / family photo",
                key="photo_uploader",
            )
        # Status sits under the Generate column (same width as that button)
        _s1, _s2 = st.columns(2, gap="medium")
        with _s1:
            gen_status = st.empty()
        with _s2:
            st.empty()

        if uploaded is not None:
            try:
                new_image = Image.open(uploaded).convert("RGB")
                st.session_state.image_pil = new_image
                image = new_image
                if st.session_state.get("last_upload_name") != uploaded.name:
                    st.session_state.last_upload_name = uploaded.name
                    st.session_state.caption = None
                    st.session_state.story = None
                    st.session_state.audio_bytes = None
                    st.rerun()
            except Exception:
                st.error("I could not open that file. Please try another JPG or PNG.")
                return

        # Generating status sits on the Generate Story button (left mid column)
        if gen:
            if image is None:
                gen_status.markdown(
                    '<div class="gen-status">Please upload a picture first!</div>',
                    unsafe_allow_html=True,
                )
            else:
                with gen_status.container():
                    with st.spinner("Making your story..."):
                        try:
                            caption = caption_image(image)
                            ok_c, _ = is_kid_safe_text(caption)
                            if not ok_c:
                                caption = "happy friends playing together outdoors"
                            story = generate_story(caption)
                            ok_s, _ = is_kid_safe_text(story)
                            if not ok_s:
                                story = _template_story(caption)
                            audio_bytes = text_to_speech(story)
                            st.session_state.caption = caption
                            st.session_state.story = story
                            st.session_state.audio_bytes = audio_bytes
                        except Exception as err:
                            logging.getLogger("storytime").exception(
                                "story generation failed"
                            )
                            st.error(
                                "The story magic hiccuped. Please try Generate Story again."
                            )
                            with st.expander("Details"):
                                st.code(f"{type(err).__name__}: {err}")

    with right:
        caption = st.session_state.get("caption")
        story = st.session_state.get("story")
        audio_bytes = st.session_state.get("audio_bytes")

        if story and audio_bytes:
            render_karaoke_story(story, audio_bytes, caption or "")
        else:
            st.markdown(
                '<div class="story-empty">'
                '<div><div style="color:#ff4d6d;font-weight:900;font-size:1.05rem;margin-bottom:0.6rem;">Your story</div>'
                "Waiting for a story 📖<br/>"
                '<span style="font-size:1rem;font-weight:800;opacity:0.8;">'
                "Words glow and scroll while it is read aloud</span></div></div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
