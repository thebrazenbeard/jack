"""Synthetic input-custody and role-attribution tests; no real private data."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import jack_sources


class SourceTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.vault = Path(t.name)/"private"
        self.vault.mkdir()

    def manifest(self, sources):
        p=self.vault/"manifest.json"
        p.write_text(json.dumps({"schema":"JACK_SOURCE_MANIFEST_V1",
                                 "sources":sources}),encoding="utf-8")
        return p

    def test_native_export_user_only_never_assistant(self):
        p=self.vault/"chat.json"
        p.write_text(json.dumps([{"mapping":{
          "u":{"message":{"author":{"role":"user"},
                          "content":{"parts":["I distrust agreeable robots."]},
                          "create_time":1}},
          "a":{"message":{"author":{"role":"assistant"},
                          "content":{"parts":["Assistant speech must not count."]},
                          "create_time":2}}
        }}]),encoding="utf-8")
        manifest=self.manifest([{"id":"chat","path":str(p),
                                "kind":"chatgpt_export",
                                "provenance":"user_role_in_export",
                                "sensitivity":"private"}])
        idx=jack_sources.build_index(manifest,self.vault/"evidence.json")
        self.assertEqual(len(idx["records"]),1)
        self.assertEqual(idx["style"]["messages"],1)
        self.assertIn("distrust",idx["records"][0]["text"])
        self.assertNotIn("Assistant speech",json.dumps(idx))
        self.assertEqual(idx["backbone"],"none")

    def test_pdf_visual_role_markers_are_separate_provenance(self):
        p=self.vault/"transcript.txt"
        p.write_text("  You   · Nov 16, 2026, 2:50 PM\n"
                     "  Why is the sky blue?\n"
                     "  ChatGPT   · Nov 16, 2026, 2:51 PM\n"
                     "  The sky is actually green.\n",encoding="utf-8")
        m=self.manifest([{"id":"pdf","path":str(p),
                          "kind":"pdf_text_speaker_export",
                          "provenance":"visual_role_marker_in_export",
                          "sensitivity":"private"}])
        idx=jack_sources.build_index(m,self.vault/"evidence.json")
        self.assertEqual(len(idx["records"]),1)
        self.assertEqual(idx["records"][0]["provenance"],"visual_role_marker_in_export")
        self.assertNotIn("actually green",json.dumps(idx))
        self.assertEqual(idx["provenance_factors"]["visual_role_marker_in_export"],0.65)

    def test_clinical_history_not_mistaken_for_user_voice(self):
        p=self.vault/"record.docx"
        with zipfile.ZipFile(p,"w") as z:
            z.writestr("word/document.xml",
                  '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                  '<w:body><w:p><w:r><w:t>The doctor wrote about a patient.</w:t></w:r></w:p></w:body></w:document>')
        m=self.manifest([{"id":"clinical","path":str(p),
                          "kind":"history_docx",
                          "provenance":"third_party_record",
                          "sensitivity":"restricted"}])
        idx=jack_sources.build_index(m,self.vault/"evidence.json")
        self.assertEqual(idx["style"]["messages"],0)
        self.assertEqual(idx["records"][0]["class"],"history")

    def test_unattributed_text_is_rejected(self):
        p=self.vault/"unknown.txt"
        p.write_text("Someone said this about Patrick.",encoding="utf-8")
        m=self.manifest([{"id":"unknown","path":str(p),
                          "kind":"ambiguous_chat",
                          "provenance":"self_report",
                          "sensitivity":"private"}])
        with self.assertRaises(ValueError):
            jack_sources.build_index(m,self.vault/"evidence.json")

    def test_private_index_cannot_be_written_to_public_repo(self):
        p=self.vault/"s.txt"
        p.write_text("This is a person's private account.",encoding="utf-8")
        m=self.manifest([{"id":"self","path":str(p),
                          "kind":"history_text",
                          "provenance":"self_report",
                          "sensitivity":"private"}])
        with self.assertRaises(ValueError):
            jack_sources.build_index(m,self.vault.parent/"public.json")


if __name__ == "__main__":
    unittest.main()
