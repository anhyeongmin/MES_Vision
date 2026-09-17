"""Synchronous, one-frame-at-a-time inputs; run reads outside the future UI thread."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4
import re

import av
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class SourceKind(StrEnum):
    IMAGE = "image"
    FOLDER = "folder"
    VIDEO = "video"
    D405 = "d405"
    UVC = "uvc"


class InputStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    FRAME = "frame"
    END = "end"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Frame:
    frame_id: str
    session_id: str
    sequence: int
    source_kind: SourceKind
    source_uri: str
    read_at_utc: datetime
    rgb: np.ndarray
    captured_at_utc: datetime | None = None
    media_time_seconds: float | None = None
    encoded_size: tuple[int, int] | None = None
    transformations: tuple[str, ...] = ()
    coordinate_space: str = "input_rgb_pixels"
    is_live: bool = False
    host_read_completed_monotonic: float | None = None

    acquisition_identity: str | None = None
    acquisition_source: dict | None = None

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    def metadata(self) -> dict[str, Any]:
        """JSON-ready metadata, never serializes the large pixel buffer."""
        return {
            **({"acquisition_identity": self.acquisition_identity, "acquisition_source": self.acquisition_source} if self.acquisition_identity else {}),
            "frame_id": self.frame_id, "session_id": self.session_id,
            "sequence": self.sequence, "source_kind": self.source_kind.value,
            "source_uri": self.source_uri,
            "read_at_utc": self.read_at_utc.isoformat(),
            "captured_at_utc": self.captured_at_utc.isoformat() if self.captured_at_utc else None,
            "media_time_seconds": self.media_time_seconds,
            "width": self.width, "height": self.height,
            "encoded_size": self.encoded_size,
            "transformations": self.transformations,
            "coordinate_space": self.coordinate_space,
            "color_format": "RGB", "dtype": "uint8", "is_live": self.is_live,
            "host_read_completed_monotonic": self.host_read_completed_monotonic,
        }


@dataclass(frozen=True, slots=True)
class InputEvent:
    status: InputStatus
    frame: Frame | None = None
    code: str | None = None
    message: str = ""


class InputFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class InputSource(ABC):
    """Single-consumer source. END/ERROR are terminal until a new instance is created.

    Every non-FRAME event has frame=None. close() releases resources; use a context
    manager even when processing fails. No seeking, looping, pacing or robot commands.
    """

    def __init__(self, kind: SourceKind):
        self.kind = kind
        self.session_id = uuid4().hex
        self.sequence = 0
        self.event = InputEvent(InputStatus.CREATED)

    @property
    def status(self) -> InputStatus:
        return self.event.status

    def open(self) -> InputEvent:
        if self.status != InputStatus.CREATED:
            return self.event
        try:
            self._open()
            self.event = InputEvent(InputStatus.READY, message="입력 준비 완료")
        except (InputFailure, OSError, ValueError, av.error.FFmpegError) as exc:
            self._fail(exc)
        return self.event

    def read(self) -> InputEvent:
        if self.status == InputStatus.CREATED:
            self.open()
        if self.status not in (InputStatus.READY, InputStatus.FRAME):
            return self.event
        try:
            data = self._read()
            if data is None:
                self._release()
                self.event = InputEvent(InputStatus.END, message="입력이 끝났습니다")
            else:
                rgb, uri, encoded_size, transforms, media_time = data
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise InputFailure("INVALID_RGB", "입력이 RGB uint8 형식이 아닙니다")
                if not rgb.shape[0] or not rgb.shape[1]:
                    raise InputFailure("EMPTY_FRAME", "빈 프레임입니다")
                # Own a separate contiguous buffer: decoder reuse cannot alter old frames.
                rgb = np.array(rgb, dtype=np.uint8, order="C", copy=True)
                rgb.setflags(write=False)
                frame = Frame(
                    frame_id=f"{self.session_id}:{self.sequence}",
                    session_id=self.session_id, sequence=self.sequence,
                    source_kind=self.kind, source_uri=uri,
                    read_at_utc=datetime.now(timezone.utc), rgb=rgb,
                    encoded_size=encoded_size, transformations=transforms,
                    media_time_seconds=media_time,
                )
                self.sequence += 1
                self.event = InputEvent(InputStatus.FRAME, frame=frame, message="프레임 수신")
        except (InputFailure, OSError, ValueError, av.error.FFmpegError) as exc:
            self._fail(exc)
        return self.event

    def _fail(self, exc: Exception) -> None:
        self._release()
        code = exc.code if isinstance(exc, InputFailure) else "DECODE_ERROR"
        status = InputStatus.NOT_CONFIGURED if code == "D405_NOT_CONFIGURED" else InputStatus.ERROR
        self.event = InputEvent(status, code=code, message=str(exc))

    def close(self) -> InputEvent:
        self._release()
        self.event = InputEvent(InputStatus.CLOSED, message="입력을 닫았습니다")
        return self.event

    def __enter__(self) -> InputSource:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @abstractmethod
    def _open(self) -> None: ...

    @abstractmethod
    def _read(self) -> tuple | None: ...

    def _release(self) -> None:
        pass


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def _image(path: Path) -> tuple:
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise InputFailure("MULTI_FRAME_IMAGE", f"다중 페이지/애니메이션 이미지는 지원하지 않습니다: {path}")
            # Pillow can expose 16-bit RGB PNG/TIFF as mode RGB after truncation.
            # Check encoded depth before converting, not only the Pillow mode.
            if image.format == "PNG":
                with path.open("rb") as encoded:
                    header = encoded.read(25)
                if len(header) >= 25 and header[24] > 8:
                    raise InputFailure("UNSUPPORTED_COLOR", f"16비트 PNG 입력은 지원하지 않습니다: {path}")
            if image.format == "TIFF":
                depths = image.tag_v2.get(258, (8,))
                if isinstance(depths, int):
                    depths = (depths,)
                if any(depth > 8 for depth in depths):
                    raise InputFailure("UNSUPPORTED_COLOR", f"고비트 TIFF 입력은 지원하지 않습니다: {path}")
            encoded_size = image.size
            if image.mode not in {"RGB", "RGBA", "L", "LA", "P", "1"}:
                raise InputFailure("UNSUPPORTED_COLOR", f"8비트 RGB/회색조 입력이 필요합니다 ({image.mode}): {path}")
            transforms = []
            orientation = image.getexif().get(274, 1)
            image = ImageOps.exif_transpose(image)
            if orientation in range(2, 9):
                transforms.append(f"exif_orientation_{orientation}")
            if "A" in image.getbands() or "transparency" in image.info:
                if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                    raise InputFailure("TRANSPARENT_IMAGE", f"배경이 확정된 불투명 이미지가 필요합니다: {path}")
            if image.mode != "RGB":
                transforms.append(f"{image.mode}_to_RGB")
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            return rgb, path.as_uri(), encoded_size, tuple(transforms), None
    except (Image.DecompressionBombError, UnidentifiedImageError) as exc:
        raise InputFailure("INVALID_IMAGE", f"이미지를 읽을 수 없습니다: {path}: {exc}") from exc


class ImageSource(InputSource):
    def __init__(self, path: str | Path):
        super().__init__(SourceKind.IMAGE)
        self.path = Path(path).expanduser().resolve()

    def _open(self) -> None:
        if not self.path.is_file():
            raise InputFailure("FILE_NOT_FOUND", f"이미지 파일이 없습니다: {self.path}")

    def _read(self) -> tuple | None:
        return _image(self.path) if self.sequence == 0 else None


def _natural_key(path: Path) -> tuple:
    text = path.as_posix().casefold()
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in re.split(r"(\d+)", text)), path.as_posix()


class FolderSource(InputSource):
    def __init__(self, path: str | Path, *, recursive: bool = False):
        super().__init__(SourceKind.FOLDER)
        self.path = Path(path).expanduser().resolve()
        self.recursive = recursive
        self.files: tuple[Path, ...] = ()

    def _open(self) -> None:
        if not self.path.is_dir():
            raise InputFailure("FOLDER_NOT_FOUND", f"폴더가 없습니다: {self.path}")
        paths = self.path.rglob("*") if self.recursive else self.path.iterdir()
        self.files = tuple(sorted(
            (p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=_natural_key,
        ))
        if not self.files:
            raise InputFailure("EMPTY_FOLDER", f"지원되는 이미지 파일이 없습니다: {self.path}")

    def _read(self) -> tuple | None:
        if self.sequence == len(self.files):
            return None
        return _image(self.files[self.sequence])


class VideoSource(InputSource):
    def __init__(self, path: str | Path):
        super().__init__(SourceKind.VIDEO)
        self.path = Path(path).expanduser().resolve()
        self._container = None
        self._frames = None

    def _open(self) -> None:
        if not self.path.is_file():
            raise InputFailure("FILE_NOT_FOUND", f"영상 파일이 없습니다: {self.path}")
        # A file handle keeps input local; disallow external playlist/resource access.
        self._file = self.path.open("rb")
        try:
            self._container = av.open(self._file, options={"protocol_whitelist": "file"}, io_open=self._deny_external)
            if not self._container.streams.video:
                raise InputFailure("NO_VIDEO_STREAM", f"영상 트랙이 없습니다: {self.path}")
            stream = self._container.streams.video[0]
            stream.codec_context.options = {"err_detect": "explode"}
            self._frames = iter(self._container.decode(stream))
        except Exception:
            self._release()
            raise

    @staticmethod
    def _deny_external(*_: Any) -> None:
        raise OSError("외부 영상 리소스/재생 목록은 지원하지 않습니다")

    def _read(self) -> tuple | None:
        try:
            decoded = next(self._frames)
        except StopIteration:
            if self.sequence == 0:
                raise InputFailure("EMPTY_VIDEO", f"읽을 수 있는 영상 프레임이 없습니다: {self.path}")
            return None
        seconds = float(decoded.pts * decoded.time_base) if decoded.pts is not None and decoded.time_base is not None else None
        if any(component.bits > 8 for component in decoded.format.components):
            raise InputFailure("UNSUPPORTED_COLOR", f"8비트 SDR 영상이 필요합니다 ({decoded.format.name}): {self.path}")
        transforms = (f"{decoded.format.name}_to_RGB", "encoded_video_orientation")
        return decoded.to_ndarray(format="rgb24"), self.path.as_uri(), (decoded.width, decoded.height), transforms, seconds

    def _release(self) -> None:
        self._frames = None
        if self._container is not None:
            self._container.close()
            self._container = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None


class D405Source(InputSource):
    """Configured RealSense color input. No-argument use remains explicitly unconfigured."""

    def __init__(self, settings=None, *, factory=None):
        super().__init__(SourceKind.D405)
        self.settings=settings; self.factory=factory; self.camera=None

    def _open(self) -> None:
        if self.settings is None: raise InputFailure("D405_NOT_CONFIGURED", "D405 연결·촬영 설정이 준비되지 않았습니다")
        try:
            from mes_vision.operation.camera import D405Camera
            self.camera=(self.factory or D405Camera)(self.settings); self.camera.open()
        except Exception as exc: raise InputFailure("D405_CONNECT_ERROR",str(exc)) from exc

    def _read(self) -> None:
        raise InputFailure("D405_READ_CONTRACT", "D405는 실시간 프레임 읽기 경로를 사용합니다")

    def read(self):
        import time
        if self.status==InputStatus.CREATED: self.open()
        if self.status not in {InputStatus.READY,InputStatus.FRAME}: return self.event
        try:
            started=time.monotonic()
            while time.monotonic()-started<2:
                frame=self.camera.read()
                if frame is not None:
                    self.event=InputEvent(InputStatus.FRAME,frame=frame,message="D405 프레임 수신"); return self.event
            raise InputFailure("D405_FRAME_TIMEOUT","D405 영상 수신 시간이 초과됐습니다")
        except Exception as exc: self._fail(exc)
        return self.event

    def _release(self):
        if self.camera is not None:
            try: self.camera.close()
            finally: self.camera=None
