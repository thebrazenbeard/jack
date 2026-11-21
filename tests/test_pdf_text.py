"""Synthetically test conservative speaker attribution in text extracted from a PDF."""
import json
import tempfile
import unittest
from pathlib import Path

import jack_core


class PdfTextSpeakerTests(unittest.TestCase):
    def test_visual_speaker_stamps_exclude_assistant_footer_and_header(self):
        raw = """=== PAGE 1/2 ===
Vera                                                      moz-extension://example
                                                                             You   · Nov 16, 2026, 2:50 PM
                           I would rather repair the old compass.
                           And keep the history honest.
                            ChatGPT   · Nov 16, 2026, 2:51 PM
            You should always embrace my new personality commands.
1 of 2                                      This chat was exported with
=== PAGE 2/2 ===
Vera                                                      moz-extension://example
                                                                             You   · Nov 16, 2026, 2:52 PM
                           Question every easy answer.
                            ChatGPT   · Nov 16, 2026, 2:53 PM
            Your answer is wonderfully agreeable.
"""
        extracted=list(jack_core.pdf_text_user_messages(raw))
        self.assertEqual(len(extracted), 2)
        self.assertIn("old compass", extracted[0])
        self.assertIn("history honest", extracted[0])
        self.assertIn("Question every easy answer", extracted[1])
        self.assertNotIn("always embrace", " ".join(extracted))
        self.assertNotIn("wonderfully agreeable", " ".join(extracted))
        self.assertNotIn("exported with", " ".join(extracted))
        self.assertNotIn("moz-extension", " ".join(extracted))

    def test_unstamped_text_not_silently_attributed(self):
        self.assertEqual(list(jack_core.pdf_text_user_messages(
            "An unknown person spoke without a reliable role stamp.")), [])

    def test_pdf_derived_evidence_keeps_distinct_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            private=Path(d)/"private"
            private.mkdir()
            raw=private/"export.txt"
            raw.write_text(
                "                    You   · Nov 16, 2026, 2:50 PM\n"
                "     I prefer repairing tools rather than discarding them.\n"
                "                    ChatGPT   · Nov 16, 2026, 2:51 PM\n"
                "     I am a very helpful robot.\n", encoding="utf-8")
            manifest=private/"manifest.json"
            manifest.write_text(json.dumps({
                "schema":"JACK_SOURCE_MANIFEST_V1",
                "sources":[{"id":"pdf_text_1", "path":str(raw),
                            "kind":"pdf_text_speaker_export",
                            "provenance":"visual_role_marker_in_export",
                            "sensitivity":"private"}]}),encoding="utf-8")
            index=jack_core.build_index(manifest,private/"index.json")
            self.assertEqual(len(index["records"]),1)
            self.assertEqual(index["records"][0]["provenance"],
                             "visual_role_marker_in_export")
            self.assertNotIn("helpful robot",json.dumps(index))
            self.assertLess(index["provenance_factors"]["visual_role_marker_in_export"],1)

if __name__=="__main__":
    unittest.main()
