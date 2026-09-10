"""Live YouTube Shorts feed via Playwright + ScreenCaptureKit (opt-in, no login).

Playwright opens public Shorts in headed Chromium and presses ArrowDown on scroll.
ScreenCaptureKit streams the Chromium window into the fly retina (no CDP screenshots,
so the video window should not blink from capture).

Requires:
  pip install -e ".[shorts]"
  PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium
  System Settings → Privacy & Security → Screen Recording → enable Terminal
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid

import numpy as np
from PIL import Image


DEFAULT_SHORTS_URL = "https://www.youtube.com/shorts"
_TARGET_W = 360
_TARGET_H = 640


def _ensure_browsers_path() -> None:
    """Use package-local browsers (``PLAYWRIGHT_BROWSERS_PATH=0``) when unset."""
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"


def _require_playwright():
    _ensure_browsers_path()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'YouTube Shorts mode needs Playwright. Install with:\n'
            '  pip install -e ".[shorts]"\n'
            "  PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium"
        ) from exc
    return sync_playwright


def _dismiss_consent(page) -> None:
    """Best-effort cookie / consent dismissal; headed mode lets the user click if this fails."""
    selectors = [
        'button:has-text("Accept all")',
        'button:has-text("Accept All")',
        'button:has-text("I agree")',
        'button:has-text("Got it")',
        'button[aria-label="Accept all"]',
        'tp-yt-paper-button:has-text("Accept all")',
        "#introAgreeButton",
        "form[action*='consent'] button",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=400):
                loc.click(timeout=800)
                page.wait_for_timeout(400)
                return
        except Exception:
            continue
    # Regional consent variants do not share stable selectors. Prefer an
    # affirmative action by visible text after the known selectors fail.
    try:
        page.evaluate(
            """() => {
                const words = ['accept all', 'accept', 'i agree', 'agree', 'continue'];
                const controls = [...document.querySelectorAll(
                    'button, input[type="submit"], tp-yt-paper-button'
                )];
                const target = controls.find((el) => {
                    const text = (el.innerText || el.value || el.getAttribute('aria-label') || '')
                        .trim().toLowerCase();
                    return words.some((word) => text === word || text.includes(word));
                });
                if (target) target.click();
            }"""
        )
        page.wait_for_timeout(500)
    except Exception:
        pass


def _wait_through_consent(page, timeout_ms: int = 10_000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        _dismiss_consent(page)
        try:
            if page.locator("video").count() > 0:
                return
        except Exception:
            pass
        page.wait_for_timeout(350)


def _active_video_signature(page) -> str:
    """YouTube video ID for verifying that Shorts actually advanced."""
    try:
        return str(
            page.evaluate(
                """() => {
                    const path = location.pathname.match(/^\\/shorts\\/([^/?#]+)/);
                    if (path) return path[1];
                    const active = document.querySelector(
                        'ytd-reel-video-renderer[is-active], [data-video-id][is-active]'
                    );
                    const attr = active?.getAttribute('video-id')
                        || active?.getAttribute('data-video-id');
                    if (attr) return attr;
                    const link = active?.querySelector('a[href*="/shorts/"]')
                        || document.querySelector('a[href*="/shorts/"][aria-current="page"]');
                    return link?.href?.match(/\\/shorts\\/([^/?#]+)/)?.[1] || '';
                }"""
            )
        )
    except Exception:
        return ""


def _active_video_metrics(page) -> dict[str, float]:
    try:
        result = page.evaluate(
            """() => {
                const videos = [...document.querySelectorAll('video')]
                    .filter((v) => {
                        const r = v.getBoundingClientRect();
                        return r.width > 100 && r.height > 100;
                    })
                    .sort((a, b) => {
                        const ar = a.getBoundingClientRect();
                        const br = b.getBoundingClientRect();
                        return (br.width * br.height) - (ar.width * ar.height);
                    });
                const v = videos[0];
                return {
                    currentTime: Number(v?.currentTime || 0),
                    duration: Number(v?.duration || 0),
                };
            }"""
        )
        return {
            "watch_seconds": max(0.0, float(result.get("currentTime", 0.0))),
            "duration_seconds": max(0.0, float(result.get("duration", 0.0))),
        }
    except Exception:
        return {"watch_seconds": 0.0, "duration_seconds": 0.0}


def _ensure_unmuted(page) -> None:
    """Keep the active Shorts video audible across player replacements."""
    try:
        page.evaluate(
            """() => {
                for (const video of document.querySelectorAll('video')) {
                    video.muted = false;
                    video.defaultMuted = false;
                    video.volume = 1;
                }
            }"""
        )
    except Exception:
        pass


class ShortsReel:
    """Duck-types ``Reel``: live buffer instead of a fixed frame list."""

    def __init__(self, feed: "YouTubeShortsFeed"):
        self._feed = feed
        self.reel_id = f"shorts-{uuid.uuid4().hex[:8]}"
        self.title = "YouTube Shorts"
        self.fps = 20.0
        self.index = 0
        self.frames: list[np.ndarray] = []

    @property
    def frame(self) -> np.ndarray:
        return self._feed.latest_frame()

    def advance(self) -> np.ndarray:
        self.index += 1
        return self._feed.capture(force=False)

    @property
    def duration_seconds(self) -> float:
        return float("inf")


class YouTubeShortsFeed:
    """Duck-types ``Feed``: Playwright navigates; ScreenCaptureKit supplies frames."""

    def __init__(
        self,
        url: str = DEFAULT_SHORTS_URL,
        headless: bool = False,
        width: int = _TARGET_W,
        height: int = _TARGET_H,
    ):
        if headless:
            # ScreenCaptureKit needs an on-screen window.
            print(
                "note: --shorts-headless ignored; ScreenCaptureKit needs a visible Chromium window",
                flush=True,
            )
        self.url = url
        self.headless = False
        self.width = width
        self.height = height
        self._lock = threading.RLock()
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._closed = False
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._streamer = None
        self._capture_cdp = None
        self._last_capture_at = 0.0
        self._last_audio_check_at = 0.0
        self._last_media_check_at = 0.0
        self.capture_backend = "starting"
        self.last_scroll_error = ""
        self.last_watch_metrics = {
            "watch_seconds": 0.0,
            "duration_seconds": 0.0,
        }
        self._current_watch_metrics = dict(self.last_watch_metrics)
        self._media_last_time = 0.0
        self._media_loops = 0
        self.index = 0
        self._reel = ShortsReel(self)
        self.reels = [self._reel]
        try:
            self._start()
        except BaseException:
            # Close synchronous Playwright objects while its event loop is still
            # alive; deferring this to __del__ can produce un-awaited coroutines.
            self.close()
            raise

    def _start(self) -> None:
        from flyscroll.screencapture import WindowStreamer, find_chrome_window

        sync_playwright = _require_playwright()
        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch(
                headless=False,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )
        except Exception as exc:
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg.lower():
                raise RuntimeError(
                    "Playwright Chromium is missing. From the project venv run:\n"
                    "  PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium"
                ) from exc
            raise

        self._context = self._browser.new_context(
            viewport={"width": 420, "height": 780},
            device_scale_factor=1,
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            locale="en-US",
        )
        self._page = self._context.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded", timeout=60_000)
        _wait_through_consent(self._page)
        try:
            self._page.wait_for_selector("video", timeout=15_000)
        except Exception:
            pass
        self._page.wait_for_timeout(600)
        _ensure_unmuted(self._page)
        self._page.bring_to_front()
        initial_video_id = _active_video_signature(self._page)
        if initial_video_id:
            self._reel.reel_id = f"youtube-{initial_video_id}"
        self._reel.title = "YouTube Shorts #1"
        self._reset_media_tracker()

        # Get the browser's OS pid from Chromium itself. Window title matching is
        # retained only as a fallback because YouTube can rewrite document.title.
        browser_pid = None
        try:
            cdp = self._browser.new_browser_cdp_session()
            process_info = cdp.send("SystemInfo.getProcessInfo")
            browser_processes = [
                p for p in process_info.get("processInfo", []) if p.get("type") == "browser"
            ]
            if browser_processes:
                browser_pid = int(browser_processes[0]["id"])
            cdp.detach()
        except Exception:
            browser_pid = None

        capture_title = f"FlyScroll Capture {uuid.uuid4().hex[:10]}"
        self._page.evaluate("(title) => { document.title = title; }", capture_title)
        try:
            window = find_chrome_window(
                prefer_title_substr=None if browser_pid is not None else capture_title,
                owner_pid=browser_pid,
            )
            self._streamer = WindowStreamer(
                target_w=self.width, target_h=self.height, fps=20.0
            )
            self._streamer.start(window)
            self._frame = self._streamer.wait_first_frame(timeout=2.5)
            self.capture_backend = "screencapturekit"
        except Exception as exc:
            if self._streamer is not None:
                try:
                    self._streamer.stop()
                except Exception:
                    pass
                self._streamer = None
            self._start_compositor_capture()
            print(
                "ScreenCaptureKit window capture unavailable; using Chromium "
                f"compositor capture ({exc}).",
                flush=True,
            )

    def _start_compositor_capture(self) -> None:
        """Fallback that captures Chromium's compositor without touching the window."""
        if self._context is None or self._page is None:
            raise RuntimeError("Cannot start compositor capture before Chromium is ready")
        self._capture_cdp = self._context.new_cdp_session(self._page)
        self.capture_backend = "chromium_compositor"
        self._frame = self._capture_compositor_frame()
        if self._frame.size == 0 or int(self._frame.max()) <= 2:
            raise RuntimeError("Chromium compositor also returned a blank frame")

    def _capture_compositor_frame(self) -> np.ndarray:
        if self._capture_cdp is None:
            return self._frame
        shot = self._capture_cdp.send(
            "Page.captureScreenshot",
            {
                "format": "jpeg",
                "quality": 55,
                "fromSurface": True,
                "captureBeyondViewport": False,
            },
        )
        raw = base64.b64decode(shot["data"])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        iw, ih = img.size
        target_aspect = self.width / self.height
        src_aspect = iw / max(ih, 1)
        if src_aspect > target_aspect:
            new_w = max(1, int(ih * target_aspect))
            left = (iw - new_w) // 2
            img = img.crop((left, 0, left + new_w, ih))
        else:
            new_h = max(1, int(iw / target_aspect))
            top = (ih - new_h) // 2
            img = img.crop((0, top, iw, top + new_h))
        img = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    @property
    def current(self) -> ShortsReel:
        return self._reel

    @property
    def current_watch_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._current_watch_metrics)

    def latest_frame(self) -> np.ndarray:
        with self._lock:
            now = time.monotonic()
            if (
                self._page is not None
                and now - self._last_audio_check_at >= 1.0
            ):
                _ensure_unmuted(self._page)
                self._last_audio_check_at = now
            if (
                self._page is not None
                and now - self._last_media_check_at >= 0.25
            ):
                self._update_media_progress()
                self._last_media_check_at = now
            if self._streamer is not None:
                self._frame = self._streamer.latest_frame()
            elif self._capture_cdp is not None:
                # The neural loop is slower than video playback; sample only the
                # newest compositor frame and avoid duplicate captures per tick.
                if now - self._last_capture_at >= 0.075:
                    frame = self._capture_compositor_frame()
                    if frame.size and int(frame.max()) > 2:
                        self._frame = frame
                    self._last_capture_at = now
            return self._frame

    def capture(self, force: bool = False) -> np.ndarray:
        # Stream pushes continuously; just read the latest buffer.
        return self.latest_frame()

    def scroll(self) -> ShortsReel:
        with self._lock:
            if self._closed or self._page is None:
                return self._reel
            _wait_through_consent(self._page, timeout_ms=2_500)
            before = _active_video_signature(self._page)
            self._update_media_progress()
            watched_metrics = dict(self._current_watch_metrics)
            self.last_scroll_error = ""
            changed = False
            attempts = (
                self._scroll_with_wheel,
                self._scroll_with_next_button,
                self._scroll_with_keyboard,
            )
            for attempt in attempts:
                try:
                    attempt()
                    after = self._wait_for_video_change(before, timeout_seconds=1.0)
                    if after:
                        changed = True
                        break
                except Exception:
                    continue
            if not changed:
                self.last_scroll_error = (
                    f"YouTube remained on video {before or 'unknown'} after all next actions"
                )
                print(
                    f"warning: {self.last_scroll_error}; skip was not recorded",
                    flush=True,
                )
                return self._reel
            _ensure_unmuted(self._page)
            self.last_watch_metrics = watched_metrics
            self.index += 1
            self._reel.reel_id = f"youtube-{after}"
            self._reel.title = f"YouTube Shorts #{self.index + 1}"
            self._reel.index = 0
            self._reset_media_tracker()
        return self._reel

    def _reset_media_tracker(self) -> None:
        self._media_last_time = 0.0
        self._media_loops = 0
        self._last_media_check_at = 0.0
        self._current_watch_metrics = {
            "watch_seconds": 0.0,
            "duration_seconds": 0.0,
        }

    def _update_media_progress(self) -> None:
        if self._page is None:
            return
        metrics = _active_video_metrics(self._page)
        current = float(metrics["watch_seconds"])
        duration = float(metrics["duration_seconds"])
        if duration <= 0:
            return
        # YouTube loops Shorts by resetting currentTime to zero. Accumulate the
        # completed pass so a full watch cannot later appear as 2–3 seconds.
        if (
            self._media_last_time >= 0.60 * duration
            and current <= 0.25 * duration
            and current + 2.0 < self._media_last_time
        ):
            self._media_loops += 1
        cumulative = self._media_loops * duration + current
        self._media_last_time = current
        self._current_watch_metrics = {
            "watch_seconds": min(cumulative, duration),
            "raw_watch_seconds": cumulative,
            "duration_seconds": duration,
        }

    def _wait_for_video_change(self, before: str, timeout_seconds: float) -> str:
        assert self._page is not None
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self._page.wait_for_timeout(150)
            after = _active_video_signature(self._page)
            if after and after != before:
                return after
        return ""

    def _scroll_with_wheel(self) -> None:
        assert self._page is not None
        self._page.mouse.move(self.width * 0.5, self.height * 0.72)
        self._page.mouse.wheel(0, self.height * 0.9)

    def _scroll_with_next_button(self) -> None:
        assert self._page is not None
        selectors = (
            "button[aria-label*='Next']",
            "#navigation-button-down button",
            "ytd-shorts #navigation-button-down",
        )
        for selector in selectors:
            target = self._page.locator(selector).first
            if target.count() and target.is_visible(timeout=250):
                target.click(timeout=700)
                return
        raise RuntimeError("No visible Shorts next button")

    def _scroll_with_keyboard(self) -> None:
        assert self._page is not None
        self._page.keyboard.press("ArrowDown")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._streamer is not None:
                try:
                    self._streamer.stop()
                except Exception:
                    pass
                self._streamer = None
            if self._capture_cdp is not None:
                try:
                    self._capture_cdp.detach()
                except Exception:
                    pass
                self._capture_cdp = None
            for obj in (self._context, self._browser):
                try:
                    if obj is not None:
                        obj.close()
                except Exception:
                    pass
            try:
                if self._pw is not None:
                    self._pw.stop()
            except Exception:
                pass
            self._page = None
            self._context = None
            self._browser = None
            self._pw = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
