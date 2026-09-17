from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.inputs import D405Source, FolderSource, ImageSource, InputStatus, VideoSource


def write_video(path: Path) -> None:
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=5)
        stream.width, stream.height, stream.pix_fmt = 80, 48, "bgr0"
        for index in range(3):
            array = np.zeros((48, 80, 3), dtype=np.uint8)
            array[:, :, index] = 220
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 5)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mes-입력-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, name="사진.png", mode="RGB", color=(211, 57, 13)):
        path = self.root / name
        Image.new(mode, (37, 19), color).save(path)
        return path

    def source(self, source):
        self.addCleanup(source.close)
        return source

    def test_rgb_pixels_original_size_identity_and_timestamps(self):
        source = self.source(ImageSource(self.image()))
        event = source.read()
        self.assertEqual(event.status, InputStatus.FRAME)
        frame = event.frame
        self.assertEqual((frame.width, frame.height), (37, 19))
        np.testing.assert_array_equal(frame.rgb[0, 0], [211, 57, 13])
        self.assertEqual(frame.encoded_size, (37, 19))
        self.assertIsNone(frame.captured_at_utc)
        self.assertIsNone(frame.media_time_seconds)
        self.assertIsNotNone(frame.read_at_utc.tzinfo)
        self.assertFalse(frame.is_live)
        self.assertFalse(frame.rgb.flags.writeable)
        self.assertEqual(frame.coordinate_space, "input_rgb_pixels")
        json.dumps(frame.metadata())

    def test_terminal_end_never_contains_previous_frame(self):
        source = self.source(ImageSource(self.image()))
        first = source.read().frame
        for _ in range(3):
            event = source.read()
            self.assertEqual(event.status, InputStatus.END)
            self.assertIsNone(event.frame)
        np.testing.assert_array_equal(first.rgb[0, 0], [211, 57, 13])

    def test_new_source_has_new_frame_id(self):
        path = self.image()
        a, b = self.source(ImageSource(path)), self.source(ImageSource(path))
        self.assertNotEqual(a.read().frame.frame_id, b.read().frame.frame_id)

    def test_close_is_idempotent_and_prevents_reopen(self):
        source = ImageSource(self.image())
        source.read()
        source.close()
        self.assertEqual(source.close().status, InputStatus.CLOSED)
        self.assertEqual(source.open().status, InputStatus.CLOSED)
        self.assertIsNone(source.read().frame)

    def test_context_releases_source_on_processing_error(self):
        source = ImageSource(self.image())
        with self.assertRaises(RuntimeError):
            with source:
                source.read()
                raise RuntimeError("processing failed")
        self.assertEqual(source.status, InputStatus.CLOSED)

    def test_missing_file_and_folder_are_errors_not_end(self):
        for source in [ImageSource(self.root / "missing.png"), VideoSource(self.root / "missing.mp4"), FolderSource(self.root / "missing")]:
            self.source(source)
            self.assertEqual(source.read().status, InputStatus.ERROR)
            self.assertIsNone(source.read().frame)

    def test_folder_natural_order_and_frozen_membership(self):
        for name in ["part10.png", "part2.png", "part1.png"]:
            self.image(name)
        (self.root / "notes.txt").write_text("ignored")
        source = self.source(FolderSource(self.root))
        source.open()
        self.image("part3.png")
        actual = [source.read().frame.source_uri for _ in range(3)]
        expected = [(self.root / name).as_uri() for name in ["part1.png", "part2.png", "part10.png"]]
        self.assertEqual(actual, expected)
        self.assertEqual(source.read().status, InputStatus.END)

    def test_recursive_folder_is_explicit(self):
        child = self.root / "nested"
        child.mkdir()
        Image.new("RGB", (8, 9)).save(child / "one.PNG")
        plain = self.source(FolderSource(self.root))
        self.assertEqual(plain.read().code, "EMPTY_FOLDER")
        recursive = self.source(FolderSource(self.root, recursive=True))
        self.assertEqual(recursive.read().status, InputStatus.FRAME)

    def test_folder_corruption_stops_without_skipping_or_stale_frame(self):
        self.image("1.png")
        (self.root / "2.png").write_bytes(b"not an image")
        self.image("3.png")
        source = self.source(FolderSource(self.root))
        self.assertEqual(source.read().status, InputStatus.FRAME)
        error = source.read()
        self.assertEqual(error.status, InputStatus.ERROR)
        self.assertEqual(error.code, "INVALID_IMAGE")
        self.assertIsNone(error.frame)
        self.assertEqual(source.read(), error)

    def test_removed_folder_member_is_error(self):
        path = self.image()
        source = self.source(FolderSource(self.root))
        source.open()
        path.unlink()
        self.assertEqual(source.read().status, InputStatus.ERROR)

    def test_grayscale_is_explicit_rgb_conversion(self):
        source = self.source(ImageSource(self.image(mode="L", color=137)))
        frame = source.read().frame
        np.testing.assert_array_equal(frame.rgb[0, 0], [137, 137, 137])
        self.assertIn("L_to_RGB", frame.transformations)

    def test_exif_orientation_keeps_consistent_pixel_coordinates(self):
        path = self.root / "rotated.jpg"
        image = Image.new("RGB", (12, 7), "red")
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
        frame = self.source(ImageSource(path)).read().frame
        self.assertEqual((frame.width, frame.height), (7, 12))
        self.assertEqual(frame.encoded_size, (12, 7))
        self.assertIn("exif_orientation_6", frame.transformations)

    def test_transparency_is_not_silently_composited(self):
        source = self.source(ImageSource(self.image(mode="RGBA", color=(255, 0, 0, 0))))
        self.assertEqual(source.read().code, "TRANSPARENT_IMAGE")

    def test_opaque_alpha_is_accepted(self):
        source = self.source(ImageSource(self.image(mode="RGBA", color=(255, 0, 0, 255))))
        self.assertEqual(source.read().status, InputStatus.FRAME)

    def test_16bit_image_is_not_silently_quantized(self):
        path = self.root / "high-depth.tiff"
        Image.fromarray(np.full((9, 11), 40000, dtype=np.uint16)).save(path)
        self.assertEqual(self.source(ImageSource(path)).read().code, "UNSUPPORTED_COLOR")

    def test_multipage_image_is_not_silently_reduced_to_first_page(self):
        path = self.root / "pages.tiff"
        Image.new("RGB", (9, 11)).save(path, save_all=True, append_images=[Image.new("RGB", (9, 11), "red")])
        self.assertEqual(self.source(ImageSource(path)).read().code, "MULTI_FRAME_IMAGE")

    def test_16bit_rgb_png_is_not_silently_quantized_by_pillow(self):
        import struct
        import zlib
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        path = self.root / "rgb16.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                         chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)) +
                         chunk(b"IDAT", zlib.compress(b"\0" + struct.pack(">HHH", 50000, 15000, 30000))) +
                         chunk(b"IEND", b""))
        self.assertEqual(self.source(ImageSource(path)).read().code, "UNSUPPORTED_COLOR")

    def test_mp4_decodes_all_frames(self):
        path = self.root / "movie.mp4"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("mpeg4", rate=5)
            stream.width, stream.height, stream.pix_fmt = 80, 48, "yuv420p"
            for index in range(3):
                frame = av.VideoFrame.from_ndarray(np.full((48, 80, 3), 30 + index * 70, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        source = self.source(VideoSource(path))
        frames = [source.read().frame for _ in range(3)]
        self.assertTrue(all(frame is not None for frame in frames))
        self.assertLess(float(frames[0].rgb.mean()), float(frames[1].rgb.mean()))
        self.assertLess(float(frames[1].rgb.mean()), float(frames[2].rgb.mean()))
        self.assertEqual(source.read().status, InputStatus.END)

    def test_10bit_video_is_not_silently_quantized(self):
        path = self.root / "high-depth.mkv"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("ffv1", rate=5)
            stream.width, stream.height, stream.pix_fmt = 80, 48, "yuv420p10le"
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 80, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        self.assertEqual(self.source(VideoSource(path)).read().code, "UNSUPPORTED_COLOR")

    def test_video_full_decode_rgb_order_pts_and_buffer_lifetime(self):
        path = self.root / "영상.mkv"
        write_video(path)
        source = self.source(VideoSource(path))
        frames = [source.read().frame for _ in range(3)]
        self.assertTrue(all(frame is not None for frame in frames))
        for index, frame in enumerate(frames):
            expected = [0, 0, 0]
            expected[index] = 220
            np.testing.assert_array_equal(frame.rgb[0, 0], expected)
            self.assertEqual((frame.width, frame.height), (80, 48))
            self.assertAlmostEqual(frame.media_time_seconds, index * 0.2, places=5)
            self.assertIsNone(frame.captured_at_utc)
        self.assertEqual(len({frame.frame_id for frame in frames}), 3)
        self.assertEqual(source.read().status, InputStatus.END)
        self.assertIsNone(source.read().frame)
        # Windows rename confirms the file handle was released at EOF.
        path.rename(self.root / "released.mkv")

    def test_corrupt_video_is_error(self):
        path = self.root / "broken.mp4"
        path.write_bytes(b"invalid movie")
        source = self.source(VideoSource(path))
        self.assertEqual(source.read().status, InputStatus.ERROR)
        self.assertIsNone(source.read().frame)
        path.rename(self.root / "released.mp4")

    def test_audio_only_is_error(self):
        import wave
        path = self.root / "audio.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"\0\0" * 800)
        self.assertEqual(self.source(VideoSource(path)).read().code, "NO_VIDEO_STREAM")

    def test_d405_is_explicitly_unconfigured(self):
        source = self.source(D405Source())
        self.assertEqual(source.read().status, InputStatus.NOT_CONFIGURED)
        self.assertEqual(source.read().code, "D405_NOT_CONFIGURED")
        self.assertIsNone(source.read().frame)

    def test_cli_records_end_and_preview_without_inspection(self):
        path = self.image()
        output = self.root / "result"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/check_input.py"), "image", str(path), "--limit", "2", "--output", str(output)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[0])["message"], "입력 준비 완료")
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["stop_reason"], "end")
        self.assertEqual(summary["frames_read"], 1)
        self.assertFalse(summary["inspection_performed"])
        with Image.open(output / "first-frame.png") as image:
            np.testing.assert_array_equal(np.array(image)[0, 0], [211, 57, 13])

    def test_cli_limit_is_not_misreported_as_eof(self):
        path = self.image()
        output = self.root / "result"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/check_input.py"), "image", str(path), "--limit", "1", "--output", str(output)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["stop_reason"], "limit_reached")

    def test_cli_error_exit_code(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/check_input.py"), "d405"], capture_output=True)
        self.assertEqual(result.returncode, 1, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
