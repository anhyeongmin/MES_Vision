"""Camera identity shared by preview, saved evidence and scan profiles."""
from urllib.parse import quote


def camera_uri(kind, identifier):
    kind = str(kind)
    if kind == 'd405': return 'realsense://' + identifier
    if kind == 'uvc': return 'uvc://' + quote(identifier, safe='')
    raise ValueError('Unsupported camera source')


def camera_driver(settings):
    # Old saved equipment is never silently reinterpreted as a USB camera.
    driver = settings.get('driver', 'd405')
    if driver not in ('uvc', 'd405'): raise ValueError('Unsupported camera driver')
    return driver


def matches_camera(frame, identifier, driver=None):
    kind = frame.source_kind.value
    return (frame.is_live and kind in ('d405', 'uvc') and (driver is None or driver == kind)
            and frame.source_uri == camera_uri(kind, identifier))
