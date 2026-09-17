"""Device-reported DirectShow ranges. Never infer support from get()==0."""
import ctypes as C
import platform
import uuid


def directshow_ranges(device_path):
    if platform.system() != 'Windows':
        raise RuntimeError('현재 촬영 조절 범위 조회는 Windows DirectShow에서 지원합니다.')
    from ctypes import wintypes as W
    class GUID(C.Structure):
        _fields_ = [('data', C.c_ubyte * 16)]
        def __init__(self, value):
            super().__init__(); self.data[:] = uuid.UUID(value).bytes_le
    ptr = C.c_void_p
    def call(obj, slot, args, *values):
        address = C.cast(obj, C.POINTER(C.POINTER(ptr))).contents[slot]
        result = C.WINFUNCTYPE(C.c_long, ptr, *args)(address)(obj, *values)
        if result < 0: raise OSError(f'DirectShow HRESULT {result & 0xffffffff:08x}')
        return result
    def release(obj):
        if obj: call(obj, 2, [])
    def query(obj, iid):
        out = ptr(); call(obj, 0, [C.POINTER(GUID), C.POINTER(ptr)], C.byref(GUID(iid)), C.byref(out)); return out
    ole = C.WinDLL('ole32'); initialized = ole.CoInitializeEx(None, 0)
    # OpenCV may already own an STA apartment on this worker thread.
    if initialized < 0 and initialized != -2147417850:
        raise OSError('COM initialization failed')
    enum = ptr(); monikers = ptr(); device = ptr(); control = ptr()
    try:
        hr = ole.CoCreateInstance(C.byref(GUID('62be5d10-60eb-11d0-bd3b-00a0c911ce86')), None, 1,
            C.byref(GUID('29840822-5b84-11d0-bd3b-00a0c911ce86')), C.byref(enum))
        if hr < 0: raise OSError('카메라 목록 조회 실패')
        call(enum, 3, [C.POINTER(GUID), C.POINTER(ptr), W.DWORD],
            C.byref(GUID('860bb310-5d01-11d0-bd3b-00a0c911ce86')), C.byref(monikers), 0)
        if not monikers: raise RuntimeError('카메라를 찾을 수 없습니다.')
        while True:
            moniker = ptr(); fetched = W.ULONG()
            if call(monikers, 3, [W.ULONG, C.POINTER(ptr), C.POINTER(W.ULONG)], 1, C.byref(moniker), C.byref(fetched)) != 0: break
            try:
                name = ptr()
                call(moniker, 20, [ptr, ptr, C.POINTER(ptr)], None, None, C.byref(name))
                try: matched = C.wstring_at(name).lower().removeprefix('@device:pnp:') == device_path.lower()
                finally: ole.CoTaskMemFree(name)
                if not matched: continue
                call(moniker, 8, [ptr, ptr, C.POINTER(GUID), C.POINTER(ptr)], None, None,
                    C.byref(GUID('56a86895-0ad4-11ce-b03a-0020af0ba770')), C.byref(device))
                break
            finally: release(moniker)
        if not device: raise RuntimeError('선택한 카메라의 제어 장치를 찾을 수 없습니다.')
        result = {}
        for name, iid, index in [('exposure','c6e13370-30ac-11d0-a18c-00a0c9118956',4),
                                 ('gain','c6e13360-30ac-11d0-a18c-00a0c9118956',9)]:
            try:
                control = query(device, iid)
                values = [W.LONG() for _ in range(5)]
                call(control, 3, [W.LONG]+[C.POINTER(W.LONG)]*5, index, *[C.byref(v) for v in values])
                low, high, step, default, flags = [v.value for v in values]
                if step <= 0 or high < low: raise ValueError('Invalid device range')
                result[name] = dict(min=low,max=high,step=step,default=default,manual=bool(flags & 2),auto=bool(flags & 1))
                current=W.LONG(); mode=W.LONG()
                call(control,5,[W.LONG,C.POINTER(W.LONG),C.POINTER(W.LONG)],index,C.byref(current),C.byref(mode))
                result[name].update(current=current.value,automatic=bool(mode.value & 1))
            except Exception as exc: result[name] = {'error':str(exc)}
            finally: release(control); control = ptr()
        return result
    finally:
        release(control); release(device); release(monikers); release(enum)
        if initialized in (0,1): ole.CoUninitialize()


def validate_controls(value, ranges):
    if type(value.get('auto_exposure')) is not bool: raise ValueError('자동 노출 설정 오류')
    exposure = ranges.get('exposure', {})
    if value['auto_exposure'] and not exposure.get('auto'): raise ValueError('자동 노출을 지원하지 않습니다.')
    for key in ('exposure','gain'):
        item = value.get(key)
        if key == 'exposure' and value['auto_exposure']: continue
        if key == 'gain' and item is None: continue
        limits = ranges.get(key,{})
        if not limits.get('manual'): raise ValueError(key+' 수동 제어 범위를 확인할 수 없습니다.')
        if type(item) not in (int,float) or not limits['min'] <= item <= limits['max']:
            raise ValueError(key+' 지원 범위를 벗어났습니다.')
        steps = (item-limits['min'])/limits['step']
        if abs(steps-round(steps)) > 1e-6: raise ValueError(key+' 지원 간격을 확인하세요.')
