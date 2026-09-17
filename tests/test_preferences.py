from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mes_vision.operation.preferences import Preferences, DEFAULTS, validate
from mes_vision.i18n import validate_catalog, trf, tr, set_language, language, LANGUAGES, catalog, qt_button_text


class PreferenceTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_missing_file_defaults_without_writing(self):
        p=Preferences(self.root); self.assertEqual(p.value,DEFAULTS); self.assertFalse(p.path.exists())
    def test_roundtrip_and_immutable_input(self):
        p=Preferences(self.root); value=dict(DEFAULTS,language="en",text_scale=130,start_page=3)
        p.save(value); value["language"]="ko"
        self.assertEqual(Preferences(self.root).value["language"],"en")
    def test_invalid_options_rejected(self):
        for key,value in (("text_scale",999),("text_scale",True),("language","xx"),("start_page",5),("start_page",True),("remember_window",1),("window_geometry","not hex"),("schema_version",2)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): validate(dict(DEFAULTS,**{key:value}))
    def test_no_device_or_automatic_operation_keys_accepted(self):
        for key in ("auto_sort","vlm_enabled","camera_connected","robot_port"):
            with self.subTest(key=key),self.assertRaises(ValueError): validate(dict(DEFAULTS,**{key:True}))
    def test_corrupt_file_preserved_on_explicit_save(self):
        path=self.root/"preferences.json"; path.write_bytes(b"broken")
        p=Preferences(self.root); self.assertTrue(p.warning); self.assertEqual(p.value,DEFAULTS)
        p.save(DEFAULTS); backups=list(self.root.glob("preferences-invalid-*.json"))
        self.assertEqual(len(backups),1); self.assertEqual(backups[0].read_bytes(),b"broken")
    def test_unreadable_file_falls_back_without_overwrite(self):
        (self.root/"preferences.json").write_text("{}")
        with patch.object(Path,"read_bytes",side_effect=PermissionError("denied")):
            p=Preferences(self.root); self.assertTrue(p.warning); self.assertEqual(p.value,DEFAULTS)
    def test_atomic_write_failure_preserves_old_file_and_memory(self):
        p=Preferences(self.root); p.save(DEFAULTS); original=p.path.read_bytes()
        with patch.object(Path,"replace",side_effect=PermissionError("denied")),self.assertRaises(PermissionError): p.save(dict(DEFAULTS,language="en"))
        self.assertEqual(p.path.read_bytes(),original); self.assertEqual(p.value,DEFAULTS); self.assertFalse(list(self.root.glob("*.tmp")))
    def test_conflicting_window_does_not_overwrite(self):
        a,b=Preferences(self.root),Preferences(self.root); a.save(dict(DEFAULTS,language="en"))
        with self.assertRaises(ValueError): b.save(DEFAULTS)
    def test_geometry_merge_preserves_new_language(self):
        old=Preferences(self.root); new=Preferences(self.root); new.save(dict(DEFAULTS,language="en"))
        old.save_geometry("aabb"); self.assertEqual(Preferences(self.root).value["language"],"en")
        self.assertEqual(Preferences(self.root).value["window_geometry"],"aabb")
    def test_disabled_remember_and_corrupt_file_not_saved_on_close(self):
        p=Preferences(self.root); p.save(dict(DEFAULTS,remember_window=False)); p.save_geometry("aabb")
        self.assertEqual(Preferences(self.root).value["window_geometry"],"")
        p.path.write_bytes(b"broken"); p.save_geometry("aabb"); self.assertEqual(p.path.read_bytes(),b"broken")
    def test_all_catalog_placeholders_match(self): validate_catalog()
    def test_all_languages_persist_across_new_preferences_instance(self):
        for locale in LANGUAGES:
            with self.subTest(locale=locale):
                Preferences(self.root).save(dict(DEFAULTS,language=locale))
                self.assertEqual(Preferences(self.root).value["language"],locale)
    def test_chinese_thai_text_buttons_and_user_values(self):
        before=language()
        try:
            for locale,start,save in (("zh-CN","开始检查","保存"),("th","เริ่มตรวจสอบ","บันทึก")):
                with self.subTest(locale=locale):
                    set_language(locale)
                    self.assertEqual(tr("검사 시작"),start)
                    self.assertEqual(qt_button_text("&Save"),save)
                    self.assertEqual(qt_button_text("UNRECOGNIZED"),"UNRECOGNIZED")
                    text=trf("{v0}  ·  버전 {v1}",v0="정상 / ชิ้นงาน / 产品",v1=42)
                    self.assertIn("정상 / ชิ้นงาน / 产品",text)
                    self.assertIn("42",text)
                    self.assertEqual(tr("NG_UNKNOWN_CODE"),"NG_UNKNOWN_CODE")
        finally: set_language(before)
    def test_catalogs_have_equal_complete_coverage(self):
        for locale in LANGUAGES:
            self.assertEqual(set(catalog(locale)),set(catalog("en")))
    def test_invalid_language_does_not_change_active_language(self):
        before=language()
        for invalid in ("zh-TW","xx",None,[]):
            with self.assertRaises(ValueError): set_language(invalid)
            with self.assertRaises(ValueError): validate(dict(DEFAULTS,language=invalid))
            self.assertEqual(language(),before)
    def test_template_preserves_user_text_and_numeric_format(self):
        before=language()
        try:
            set_language("en")
            self.assertEqual(trf("{v0}  ·  버전 {v1}",v0="정상",v1=3),"정상  ·  Version 3")
            self.assertEqual(tr("검사 시작"),"Start inspection")
            self.assertEqual(tr("UNKNOWN_PROTOCOL_CODE"),"UNKNOWN_PROTOCOL_CODE")
        finally: set_language(before)


if __name__=="__main__": unittest.main()
