"""Story-quality checks that do not download Hugging Face weights."""

import unittest

import app_uiv2 as app


CAPTIONS = (
    "a dog sitting on the beach",
    "a red bus on the street",
    "a cat sitting on a sofa",
    "a bird in the forest",
    "children playing in the park",
    "a cake on a plate",
    "a boat on the lake",
    "a horse on a hill",
    "a family under the stars",
    "a kite in the sky",
    "happy friends playing together outdoors",
    "a yellow umbrella",
    "a pizza on a plate",
)


class _Tok:
    eos_token_id = 0


class _Teller:
    def __init__(self, tail):
        self.tokenizer = _Tok()
        self.tail = tail
        self.calls = 0

    def __call__(self, seed, **kwargs):
        self.calls += 1
        return [{"generated_text": f"{seed}{self.tail}"}]


class TemplateStoryTests(unittest.TestCase):
    def test_length_safety_and_stability(self):
        for caption in CAPTIONS:
            story = app._template_story(caption)
            words = story.split()
            self.assertGreaterEqual(len(words), 50, story)
            self.assertLessEqual(len(words), 100, story)
            self.assertFalse(app._looks_like_bad_story(story, caption), story)
            ok, flagged = app.is_kid_safe_text(story)
            self.assertTrue(ok, flagged)
            self.assertEqual(story, app._template_story(caption))

    def test_scene_follows_caption_not_the_park(self):
        beach = app._template_story("a dog sitting on the beach")
        street = app._template_story("a red bus on the street")
        sofa = app._template_story("a cat sitting on a sofa")
        park = app._template_story("children playing in the park")

        self.assertNotEqual(beach, street)
        self.assertNotIn("grass", beach.lower())
        self.assertIn("beach", beach.lower())
        self.assertIn("wagged", beach.lower())

        self.assertNotIn("grass", street.lower())
        self.assertIn("street", street.lower())
        self.assertIn("bus", street.lower())

        self.assertNotIn("grass", sofa.lower())
        self.assertNotIn("beach", sofa.lower())
        self.assertIn("in a cozy room", sofa.lower())
        self.assertIn("purred", sofa.lower())
        self.assertIn("sofa", sofa.lower())

        self.assertTrue("park" in park.lower() or "grass" in park.lower())
        self.assertNotIn("children found children", park.lower())
        self.assertIn("on a friendly street", street.lower())

    def test_short_text_is_extended_without_repeating_a_closer(self):
        short = "Once upon a time, children found a dog. They smiled. The dog sat."
        fitted = app._fit_word_count(short)
        self.assertGreaterEqual(len(fitted.split()), 50)
        self.assertLessEqual(len(fitted.split()), 100)
        self.assertEqual(fitted.count("took turns"), 1)

    def test_trim_keeps_a_sentence_boundary(self):
        sentence = "Once upon a time, children found a small red dog."
        long = " ".join([sentence] * 12)
        trimmed = app._trim_to_max(long, 100)
        self.assertLessEqual(len(trimmed.split()), 100)
        self.assertTrue(trimmed.endswith("."))


class ModelDraftTests(unittest.TestCase):
    def test_clean_continuation_is_kept(self):
        teller = _Teller("played a quiet game in the sun. They shared a smile.")
        app.load_pipelines = lambda: (None, None, teller)
        story = app.generate_story("a dog on the beach")
        self.assertEqual(teller.calls, 1)
        self.assertIn("quiet game", story)
        self.assertNotIn("grass", story.lower())
        self.assertGreaterEqual(len(story.split()), 50)
        self.assertLessEqual(len(story.split()), 100)
        self.assertTrue(app.is_kid_safe_text(story)[0])

    def test_unsafe_tail_is_cut_but_clean_sentence_stays(self):
        teller = _Teller("played a quiet game. A boyfriend found a gun.")
        app.load_pipelines = lambda: (None, None, teller)
        story = app.generate_story("a dog on the beach")
        self.assertIn("quiet game", story)
        self.assertNotIn("boyfriend", story.lower())
        self.assertNotIn("gun", story.lower())
        self.assertTrue(app.is_kid_safe_text(story)[0])

    def test_fully_unsafe_draft_falls_back_to_the_scene(self):
        teller = _Teller("a boyfriend at college found a gun and beer.")
        app.load_pipelines = lambda: (None, None, teller)
        story = app.generate_story("a dog on the beach")
        self.assertNotIn("boyfriend", story.lower())
        self.assertNotIn("gun", story.lower())
        self.assertNotIn("beer", story.lower())
        self.assertNotIn("college", story.lower())
        self.assertIn("beach", story.lower())
        self.assertIn("wagged", story.lower())
        self.assertGreaterEqual(len(story.split()), 50)
        self.assertLessEqual(len(story.split()), 100)


class SafetyGateTests(unittest.TestCase):
    def test_keyword_gate(self):
        self.assertFalse(app.is_kid_safe_text("They saw a gun.")[0])
        self.assertTrue(app.is_kid_safe_text("The dog wagged its tail.")[0])

    def test_repetition_gate(self):
        self.assertTrue(app._has_heavy_repetition("dog dog dog dog dog"))
        self.assertTrue(app._looks_like_bad_story("dog dog dog dog dog", "a dog"))


if __name__ == "__main__":
    unittest.main()
