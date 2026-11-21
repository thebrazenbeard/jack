"""Synthetic-only regression tests for JACK's independent analytic language model."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import jack_model


def synthetic_index():
    return {
        "schema": "JACK_EVIDENCE_INDEX_V1",
        "style": {"messages": 2, "avg_words": 6, "question_ratio": 0.0,
                  "exclamation_ratio": 0.0},
        "source_receipts": [{"source_id": "chat", "sha256": "a" * 64,
                            "kind": "chatgpt_export", "provenance": "user_role_in_export",
                            "records": 2, "sensitivity": "private"},
                           {"source_id": "medical", "sha256": "b" * 64,
                            "kind": "history_text", "provenance": "third_party_record",
                            "records": 1, "sensitivity": "restricted"}],
        "records": [
            {"id":"chat:0", "source_id":"chat", "class":"conversation",
             "provenance":"user_role_in_export", "sensitivity":"private",
             "text":"I repair the clock. I repair the radio."},
            {"id":"chat:1", "source_id":"chat", "class":"conversation",
             "provenance":"user_role_in_export", "sensitivity":"private",
             "text":"I question convenient answers."},
            {"id":"medical:0", "source_id":"medical", "class":"history",
             "provenance":"third_party_record", "sensitivity":"restricted",
             "text":"A clinician noted a history of trauma and recovery."}
        ]}


class AnalyticModelTests(unittest.TestCase):
    def test_closed_form_counts_are_derived_from_patrick_voice(self):
        model=jack_model.build_parameters(synthetic_index())
        self.assertEqual(model["schema"],"JACK_ANALYTIC_MODEL_V1")
        self.assertEqual(model["architecture"],"weighted_word_trigram")
        self.assertEqual(model["unigram_counts"]["repair"],2)
        self.assertEqual(model["transition_weights"]["i|repair"]["the"],1.0)
        self.assertNotIn("clinician",model["unigram_counts"])
        self.assertIn("clinician",model["history_weights"])
        self.assertEqual(model["backbone"],"none")
        self.assertEqual(model["optimizer_steps"],0)
        self.assertEqual(model["neural_weights"],"none")

    def test_lexical_weights_normalized_deterministically(self):
        a=jack_model.build_parameters(synthetic_index())
        b=jack_model.build_parameters(synthetic_index())
        self.assertEqual(a,b)
        self.assertAlmostEqual(sum(a["lexicon_weights"].values()),1.0,places=7)
        self.assertEqual(len(a["weights_sha256"]),64)
        self.assertEqual(a["weights_sha256"],b["weights_sha256"])

    def test_generation_is_self_contained_deterministic_and_not_history_voice(self):
        m=jack_model.build_parameters(synthetic_index())
        a=jack_model.generate(m,"What would you say about a clock?",seed=13,max_tokens=30)
        b=jack_model.generate(m,"What would you say about a clock?",seed=13,max_tokens=30)
        self.assertEqual(a,b)
        self.assertTrue(len(a.strip())>=1)
        self.assertNotIn("clinician",a.lower())
        self.assertNotIn("trauma",a.lower())

    def test_zero_direct_voice_refuses_false_personality_model(self):
        idx=synthetic_index()
        idx["records"]=[idx["records"][-1]]
        with self.assertRaisesRegex(ValueError,"Patrick-attributed"):
            jack_model.build_parameters(idx)

    def test_save_and_reload_only_in_private_vault(self):
        model=jack_model.build_parameters(synthetic_index())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"vault"
            root.mkdir()
            out=root/"model.json"
            jack_model.save_model(model,out,root)
            loaded=jack_model.load_model(out)
            self.assertEqual(loaded["weights_sha256"],model["weights_sha256"])
            with self.assertRaises(ValueError):
                jack_model.save_model(model,Path(d)/"outside.json",root)
            with self.assertRaises(FileExistsError):
                jack_model.save_model(model,out,root)
            self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(),
                             hashlib.sha256(out.read_bytes()).hexdigest())

    def test_rejects_untrusted_schema_and_nonsensical_prompt(self):
        with self.assertRaises(ValueError):
            jack_model.build_parameters({"schema":"UNTRUSTED","records":[]})
        model=jack_model.build_parameters(synthetic_index())
        self.assertEqual(jack_model.generate(model,"",seed=2,max_tokens=12),"")


if __name__=="__main__":
    unittest.main()
