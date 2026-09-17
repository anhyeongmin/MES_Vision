import json,tempfile,unittest
from pathlib import Path
from mes_vision.operation.preferences import Preferences,DEFAULTS,validate


class ThemePreferencesTests(unittest.TestCase):
    def test_old_file_migrates_without_resetting_or_rewriting(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'preferences.json'
            old=dict(DEFAULTS,language='th',text_scale=130,start_page=3,window_geometry='aabb')
            old.pop('theme');raw=json.dumps(old).encode();path.write_bytes(raw)
            prefs=Preferences(temp)
            self.assertIsNone(prefs.warning)
            self.assertEqual(prefs.value,dict(old,theme='light'))
            self.assertEqual(path.read_bytes(),raw)

    def test_geometry_merge_preserves_theme(self):
        with tempfile.TemporaryDirectory() as temp:
            stale=Preferences(temp);current=Preferences(temp)
            current.save(dict(DEFAULTS,theme='dark'))
            stale.save_geometry('aabb')
            self.assertEqual(Preferences(temp).value['theme'],'dark')

    def test_invalid_theme_does_not_save(self):
        with tempfile.TemporaryDirectory() as temp:
            prefs=Preferences(temp);prefs.save(DEFAULTS);before=prefs.path.read_bytes()
            for value in ['auto','blue',None,[],True]:
                with self.assertRaises(ValueError):prefs.save(dict(DEFAULTS,theme=value))
            self.assertEqual(prefs.path.read_bytes(),before)


if __name__=='__main__':unittest.main()
