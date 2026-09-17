"""Display translations only. Device protocols, model classes and stored evidence keep their original values."""
import json
import os
from pathlib import Path
from string import Formatter

_language = "ko"
LANGUAGES = {"ko": "한국어", "en": "English", "zh-CN": "简体中文", "th": "ไทย"}
_catalogs = {}


def set_language(language):
    global _language
    if not isinstance(language, str) or language not in LANGUAGES: raise ValueError("Unsupported display language")
    _language = language


def language(): return _language


def catalog(locale="en"):
    if locale not in LANGUAGES: raise ValueError("Unsupported display language")
    if locale == "ko": return {source: source for source in catalog("en")}
    if locale not in _catalogs:
        _catalogs[locale] = json.loads((Path(__file__).parent / "locales" / (locale+".json")).read_text(encoding="utf-8"))
    return _catalogs[locale]


def render_text(value):
    return value.render() if isinstance(value, DisplayText) else value


def _render(recipe):
    kind, *args = recipe
    if kind == 'source':
        source, = args
        return catalog(_language).get(source, source) if _language != 'ko' else source
    if kind == 'join':
        separator, values = args
        return separator.join(render_text(value) for value in values)
    if kind == 'format':
        template, values, named = args
        return render_text(template).format(*(render_text(v) for v in values),
                                           **{k: render_text(v) for k, v in named.items()})
    if kind == 'slice':
        value, key = args
        return render_text(value)[key]
    raise ValueError('Unknown display text recipe')


class DisplayText(str):
    """A display string with its source and opaque user values retained.

    Qt bindings render this recipe again on language changes. JSON/log/protocol
    serialization remains an ordinary string; persisted evidence is not rewritten.
    Recipes contain data only, so worker results can also be pickled.
    """
    def __new__(cls, recipe):
        value = super().__new__(cls, _render(recipe))
        value.recipe = recipe
        return value

    def render(self): return _render(self.recipe)
    def __str__(self): return self.render()
    def __format__(self, spec): return format(self.render(), spec)
    def __reduce__(self): return (type(self), (self.recipe,))
    def __add__(self, other): return text_join('', (self, other))
    def __radd__(self, other): return text_join('', (other, self))
    def __getitem__(self, key): return DisplayText(('slice', self, key))
    def format(self, *values, **named): return DisplayText(('format', self, values, named))


def text_join(separator, values):
    values = tuple(values)
    if any(isinstance(value, DisplayText) for value in values):
        return DisplayText(('join', separator, values))
    return separator.join(values)


def tr(source):
    if isinstance(source, DisplayText): return DisplayText(source.recipe)
    return DisplayText(('source', source))


def trf(source, **values):
    # Format after translating the template; user-entered names and paths are never translated.
    return tr(source).format(**values)


def validate_catalog():
    def fields(text):
        return sorted((name, spec, conversion) for _, name, spec, conversion in Formatter().parse(text) if name is not None)
    sources = set(catalog("en"))
    for locale in LANGUAGES:
        translations = catalog(locale)
        if set(translations) != sources: raise ValueError("Translation coverage differs: "+locale)
        for source, translated in translations.items():
            if not isinstance(translated, str) or not translated.strip(): raise ValueError("Empty translation: "+locale+": "+source)
            if fields(source) != fields(translated): raise ValueError("Translation placeholders differ: "+locale+": "+source)


def install_display_font(app):
    """Use installed system fonts; font files are not redistributed with the program."""
    from PySide6.QtGui import QFont, QFontDatabase
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    # Explicitly load fonts as well as naming them: offscreen Qt has no Windows fallback database.
    if not getattr(app, '_mes_fonts_loaded', False):
        for name in ("malgun.ttf", "msyh.ttc", "LeelawUI.ttf"):
            path = fonts / name
            if path.is_file(): QFontDatabase.addApplicationFont(str(path))
        app._mes_fonts_loaded = True
    preferred = {
        "ko": ["Malgun Gothic", "Noto Sans CJK KR"],
        "en": ["Malgun Gothic", "Noto Sans"],
        "zh-CN": ["Microsoft YaHei", "Microsoft YaHei UI", "Noto Sans CJK SC"],
        "th": ["Leelawadee UI", "Noto Sans Thai", "Tahoma"],
    }[_language]
    font = QFont()
    font.setFamilies(preferred + ["Malgun Gothic", "Microsoft YaHei", "Leelawadee UI", "Noto Sans CJK KR", "Noto Sans CJK SC", "Noto Sans Thai", "sans-serif"])
    font.setPointSize(10)
    app.setFont(font)


_QT_BUTTONS = {
    "ko": dict(zip(
        ("OK", "Cancel", "Save", "Close", "Apply", "Reset", "Restore Defaults", "Yes", "No", "Open", "Discard", "Save All", "Yes to All", "No to All", "Abort", "Retry", "Ignore", "Help"),
        ("확인", "취소", "저장", "닫기", "적용", "초기화", "기본값 복원", "예", "아니요", "열기", "저장 안 함", "모두 저장", "모두 예", "모두 아니요", "중단", "다시 시도", "무시", "도움말"))),
    "zh-CN": dict(zip(
        ("OK", "Cancel", "Save", "Close", "Apply", "Reset", "Restore Defaults", "Yes", "No", "Open", "Discard", "Save All", "Yes to All", "No to All", "Abort", "Retry", "Ignore", "Help"),
        ("确定", "取消", "保存", "关闭", "应用", "重置", "恢复默认值", "是", "否", "打开", "不保存", "全部保存", "全是", "全否", "中止", "重试", "忽略", "帮助"))),
    "th": dict(zip(
        ("OK", "Cancel", "Save", "Close", "Apply", "Reset", "Restore Defaults", "Yes", "No", "Open", "Discard", "Save All", "Yes to All", "No to All", "Abort", "Retry", "Ignore", "Help"),
        ("ตกลง", "ยกเลิก", "บันทึก", "ปิด", "ใช้", "รีเซ็ต", "คืนค่าเริ่มต้น", "ใช่", "ไม่ใช่", "เปิด", "ไม่บันทึก", "บันทึกทั้งหมด", "ใช่ทั้งหมด", "ไม่ใช่ทั้งหมด", "ยุติ", "ลองอีกครั้ง", "ละเว้น", "ช่วยเหลือ"))),
}


def qt_button_text(source):
    return _QT_BUTTONS.get(_language, {}).get(source.replace("&", ""), source)


def install_qt_translator(app):
    from PySide6.QtCore import QTranslator
    class StandardButtons(QTranslator):
        def translate(self, context, sourceText, disambiguation=None, n=-1):
            return qt_button_text(sourceText)
    app._mes_translator = StandardButtons(app)
    app.installTranslator(app._mes_translator)
