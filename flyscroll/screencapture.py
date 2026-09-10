"""macOS ScreenCaptureKit window capture for live Shorts frames.

Requires System Settings → Privacy & Security → Screen Recording for the
app running Python (Terminal, iTerm, etc.).
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable

import numpy as np
import objc
from CoreMedia import CMSampleBufferGetImageBuffer, CMTimeMake
from Foundation import NSDate, NSObject, NSRunLoop
from PIL import Image
from Quartz import (
    CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetHeight,
    CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress,
    CVPixelBufferUnlockBaseAddress,
    kCVPixelBufferLock_ReadOnly,
)
from ScreenCaptureKit import (
    SCContentFilter,
    SCShareableContent,
    SCStream,
    SCStreamConfiguration,
    SCStreamOutputTypeScreen,
)

# kCVPixelFormatType_32BGRA
_PIXEL_BGRA = 1111970369

_CHROME_OWNERS = (
    "Google Chrome for Testing",
    "Chromium",
    "Google Chrome",
    "Chrome",
)


def _pixel_buffer_to_rgb(pixel) -> np.ndarray | None:
    if pixel is None:
        return None
    CVPixelBufferLockBaseAddress(pixel, kCVPixelBufferLock_ReadOnly)
    try:
        w = int(CVPixelBufferGetWidth(pixel))
        h = int(CVPixelBufferGetHeight(pixel))
        bpr = int(CVPixelBufferGetBytesPerRow(pixel))
        base = CVPixelBufferGetBaseAddress(pixel)
        if not base or w <= 0 or h <= 0:
            return None
        if isinstance(base, int):
            addr = base
        else:
            try:
                addr = int(ctypes.cast(base, ctypes.c_void_p).value or 0)
            except Exception:
                try:
                    addr = int(base)
                except Exception:
                    return None
        if addr == 0:
            return None
        raw = ctypes.string_at(addr, bpr * h)
        row = np.frombuffer(raw, dtype=np.uint8).reshape(h, bpr)
        bgra = row[:, : w * 4].reshape(h, w, 4)
        return np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
    finally:
        CVPixelBufferUnlockBaseAddress(pixel, kCVPixelBufferLock_ReadOnly)


class _StreamOutput(NSObject):
    """SCStreamOutput: copies BGRA frames into a shared numpy buffer."""

    def initWithCallback_(self, callback: Callable[[np.ndarray], None]):
        self = objc.super(_StreamOutput, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, of_type):
        if of_type != SCStreamOutputTypeScreen:
            return
        try:
            pixel = CMSampleBufferGetImageBuffer(sample_buffer)
            rgb = _pixel_buffer_to_rgb(pixel)
            if rgb is not None:
                self._callback(rgb)
        except Exception:
            return


class _StreamDelegate(NSObject):
    def initWithOwner_(self, owner):
        self = objc.super(_StreamDelegate, self).init()
        if self is None:
            return None
        self._owner = owner
        return self

    def stream_didStopWithError_(self, stream, error):
        owner = getattr(self, "_owner", None)
        if owner is not None:
            owner._on_stream_error(error)


def _wait_shareable_content(timeout: float = 8.0):
    box: dict = {"content": None, "error": None, "done": False}

    def handler(content, error):
        box["content"] = content
        box["error"] = error
        box["done"] = True

    SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
        False, True, handler
    )
    deadline = time.time() + timeout
    while not box["done"] and time.time() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
    if not box["done"]:
        raise TimeoutError("Timed out waiting for ScreenCaptureKit shareable content")
    if box["error"] is not None:
        msg = str(box["error"])
        if "3801" in msg or "declined" in msg.lower() or "TCC" in msg:
            raise PermissionError(
                "Screen Recording permission is required for the app that hosts "
                "this Python process.\n"
                "If you launched flyscroll from Cursor's terminal, enable **Cursor** "
                "(not only Terminal) under:\n"
                "  System Settings → Privacy & Security → Screen & System Audio Recording\n"
                "Then fully quit Cursor (Cmd+Q) and reopen, or run the same command in "
                "Terminal.app / iTerm instead."
            )
        raise RuntimeError(f"ScreenCaptureKit shareable content failed: {box['error']}")
    return box["content"]


def find_chrome_window(
    prefer_title_substr: str | None = None,
    owner_pid: int | None = None,
    retries: int = 25,
    delay: float = 0.2,
):
    """Return an SCWindow for Playwright/Chrome, waiting briefly for it to appear."""
    last_err: BaseException | None = None
    last_candidates: list[tuple[int, str, str]] = []
    for _ in range(retries):
        try:
            content = _wait_shareable_content()
            windows = list(content.windows() or [])
            candidates = []
            for w in windows:
                try:
                    app = w.owningApplication()
                    owner = str(app.applicationName() if app is not None else "")
                except Exception:
                    owner = ""
                if not any(key.lower() in owner.lower() for key in _CHROME_OWNERS):
                    continue
                try:
                    title = str(w.title() or "")
                except Exception:
                    title = ""
                try:
                    pid = int(app.processID()) if app is not None else 0
                except Exception:
                    pid = 0
                candidates.append((w, owner, title, pid))
            last_candidates = [(c[3], c[1], c[2]) for c in candidates]
            if not candidates:
                time.sleep(delay)
                continue
            if owner_pid is not None:
                ranked = [c for c in candidates if c[3] == int(owner_pid)]
                if ranked:
                    window = ranked[0][0]
                    _activate_window(window)
                    return window
                time.sleep(delay)
                continue
            if prefer_title_substr:
                pref = prefer_title_substr.lower()
                ranked = [c for c in candidates if pref in c[2].lower()]
                if ranked:
                    window = ranked[0][0]
                    _activate_window(window)
                    return window
                # A preferred title identifies the exact Playwright window.
                # Falling back here silently captures unrelated/blank Chrome.
                time.sleep(delay)
                continue

            def area(item):
                try:
                    f = item[0].frame()
                    return float(f.size.width * f.size.height)
                except Exception:
                    return 0.0

            candidates.sort(key=area, reverse=True)
            window = candidates[0][0]
            _activate_window(window)
            return window
        except PermissionError:
            raise
        except Exception as exc:
            last_err = exc
        time.sleep(delay)
    raise RuntimeError(
        "Could not find the Playwright Chromium window for ScreenCaptureKit "
        f"(pid={owner_pid}, candidates={last_candidates}, error={last_err})"
    )


def _activate_window(window) -> None:
    """Best-effort: keep the hardware-composited browser surface on the active Space."""
    try:
        from AppKit import (
            NSApplicationActivateAllWindows,
            NSApplicationActivateIgnoringOtherApps,
            NSRunningApplication,
        )

        app = window.owningApplication()
        pid = int(app.processID()) if app is not None else 0
        running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if running is not None:
            running.activateWithOptions_(
                NSApplicationActivateAllWindows | NSApplicationActivateIgnoringOtherApps
            )
            time.sleep(0.15)
    except Exception:
        pass


class WindowStreamer:
    """Continuous ScreenCaptureKit stream of one window → latest RGB frame."""

    def __init__(self, target_w: int = 360, target_h: int = 640, fps: float = 20.0):
        self.target_w = target_w
        self.target_h = target_h
        self.fps = fps
        self._lock = threading.Lock()
        self._frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        self._have_frame = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._stream = None
        self._output = None
        self._delegate = None
        self._filter = None
        self._window = None

    def start(self, window) -> None:
        self._window = window
        self._thread = threading.Thread(target=self._run, name="flyscroll-sck", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15.0):
            self.stop()
            raise TimeoutError("ScreenCaptureKit stream did not start in time")
        if self._error is not None:
            raise RuntimeError(f"ScreenCaptureKit failed to start: {self._error}") from self._error

    def _on_frame(self, rgb: np.ndarray) -> None:
        try:
            img = Image.fromarray(rgb, mode="RGB")
            iw, ih = img.size
            target_aspect = self.target_w / self.target_h
            src_aspect = iw / max(ih, 1)
            if src_aspect > target_aspect:
                new_w = max(1, int(ih * target_aspect))
                left = (iw - new_w) // 2
                img = img.crop((left, 0, left + new_w, ih))
            else:
                new_h = max(1, int(iw / target_aspect))
                top = (ih - new_h) // 2
                img = img.crop((0, top, iw, top + new_h))
            img = img.resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
            # ScreenCaptureKit can emit zero-filled startup frames before the
            # selected window's IOSurface is ready. Never promote those to the
            # visual stream or a dead capture looks like valid black video.
            if arr.size == 0 or int(arr.max()) <= 2:
                return
            with self._lock:
                self._frame = arr
                self._have_frame = True
        except Exception:
            return

    def _on_stream_error(self, error) -> None:
        if error is not None:
            self._error = RuntimeError(str(error))

    def _run(self) -> None:
        try:
            filt = SCContentFilter.alloc().initWithDesktopIndependentWindow_(self._window)
            self._filter = filt
            config = SCStreamConfiguration.alloc().init()
            try:
                frame = self._window.frame()
                w = max(2, int(frame.size.width))
                h = max(2, int(frame.size.height))
            except Exception:
                w, h = 420, 780
            config.setWidth_(w)
            config.setHeight_(h)
            config.setScalesToFit_(True)
            config.setShowsCursor_(False)
            config.setPixelFormat_(_PIXEL_BGRA)
            config.setQueueDepth_(3)
            fps = max(5.0, float(self.fps))
            config.setMinimumFrameInterval_(CMTimeMake(1, int(fps)))
            try:
                config.setIgnoreFramingSingleWindow_(True)
            except Exception:
                pass

            self._delegate = _StreamDelegate.alloc().initWithOwner_(self)
            self._output = _StreamOutput.alloc().initWithCallback_(self._on_frame)
            stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                filt, config, self._delegate
            )
            self._stream = stream
            added = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output, SCStreamOutputTypeScreen, None, None
            )
            if added is False:
                raise RuntimeError("addStreamOutput failed")

            started = {"done": False, "error": None}

            def on_start(error):
                started["error"] = error
                started["done"] = True

            stream.startCaptureWithCompletionHandler_(on_start)
            deadline = time.time() + 8.0
            while not started["done"] and time.time() < deadline:
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
            if not started["done"]:
                raise TimeoutError("startCapture timed out")
            if started["error"] is not None:
                err = str(started["error"])
                if "3801" in err or "declined" in err.lower() or "TCC" in err:
                    raise PermissionError(
                        "Screen Recording permission is required for the app that hosts "
                        "this Python process.\n"
                        "If you launched flyscroll from Cursor's terminal, enable **Cursor** "
                        "under System Settings → Privacy & Security → Screen & System Audio "
                        "Recording, then Cmd+Q Cursor and reopen — or run from Terminal.app."
                    )
                raise RuntimeError(f"startCapture failed: {started['error']}")
            self._ready.set()

            while not self._stop.is_set():
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._teardown_stream()

    def _teardown_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        done = {"ok": False}

        def on_stop(_error):
            done["ok"] = True

        try:
            stream.stopCaptureWithCompletionHandler_(on_stop)
            deadline = time.time() + 3.0
            while not done["ok"] and time.time() < deadline:
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
        except Exception:
            pass

    def latest_frame(self) -> np.ndarray:
        with self._lock:
            return self._frame

    def wait_first_frame(self, timeout: float = 6.0) -> np.ndarray:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._have_frame:
                    return self._frame
            time.sleep(0.05)
        if self._error is not None:
            raise RuntimeError(f"ScreenCaptureKit stopped before a usable frame: {self._error}")
        raise RuntimeError(
            "ScreenCaptureKit produced only blank frames for the selected Chromium "
            "window. Keep the Shorts window visible and restart flyscroll. If it "
            "persists, revoke/re-enable Screen Recording for the app running "
            "flyscroll, then fully quit and reopen that app."
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
