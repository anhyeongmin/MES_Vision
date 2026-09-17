"""D405 color presets across USB modes; connection still resolves the actual RGB8 profile.

RealSense D400 August 2025 datasheet, tables 4-4 and 4-9.
These are candidates, not a claim about an attached camera's USB bandwidth/firmware.
"""
COLOR_PRESETS = {
    (1280, 720): (5, 10, 15, 30),
    (848, 480): (5, 10, 15, 30, 60, 90),
    (640, 480): (5, 15, 30),
    (640, 360): (5, 15, 30, 60, 90),
    (480, 270): (5, 15, 30, 60, 90),
    (424, 240): (5, 15, 30, 60, 90),
}

# INNOMAKER U20CAM-720P published candidates; actual negotiation is checked on open.
UVC_PRESETS = {
    'MJPG': {(1280,720):(30,), (800,600):(30,), (640,480):(30,), (320,240):(30,)},
    'YUY2': {(1280,720):(10,), (800,600):(20,), (640,480):(30,), (320,240):(30,)},
}
