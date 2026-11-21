"""Synthetic test for corpus-derived sparse weights frozen into JACK's index."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import jack_core


class CalculatedWeightTests(unittest.TestCase):
    def test_idf_weights_are_calculated_from_source_corpus_and_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"private"
            root.mkdir()
            chat=root/"chat.json"
            data=[{"mapping":{
                "a":{"message":{"author":{"role":"user"},"content":{"parts":["I ask why the clock stopped."]}}},
                "b":{"message":{"author":{"role":"user"},"content":{"parts":["I ask why the light changed."]}}},
                "c":{"message":{"author":{"role":"assistant"},"content":{"parts":["I love the clock."]}}}
            }}]
            chat.write_text(json.dumps(data),encoding="utf-8")
            m=root/"manifest.json"
            m.write_text(json.dumps({"schema":"JACK_SOURCE_MANIFEST_V1",
                 "sources":[{"id":"chat", "path":str(chat), "kind":"chatgpt_export",
                             "provenance":"user_role_in_export", "sensitivity":"private"}]}),encoding="utf-8")
            idx=jack_core.build_index(m,root/"derived.json")
            self.assertIn("lexical_weights",idx)
            self.assertIn("clock",idx["lexical_weights"])
            self.assertIn("ask",idx["lexical_weights"])
            self.assertGreater(idx["lexical_weights"]["clock"],idx["lexical_weights"]["ask"])
            self.assertEqual(idx["lexical_weights"]["clock"],
                round(math.log(1 + (2 - 1 + 0.5) / (1 + 0.5)), 9))
            loaded=json.loads((root/"derived.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["lexical_weights"],idx["lexical_weights"])
            self.assertFalse(idx["neural_weights_updated"])
            self.assertEqual(jack_core.retrieve(idx,"clock")[0]["id"],"chat:0")

if __name__=="__main__":
    unittest.main()
