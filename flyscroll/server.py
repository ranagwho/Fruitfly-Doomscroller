"""Read-only spectator HTTP server for the doomscroll loop.

Binds 127.0.0.1 only. GET /state and /health. No remote control surface.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from flyscroll.feed import Feed, load_feed
from flyscroll.session import DoomscrollSession, SessionConfig

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).with_name("ui")

latest: dict = {"status": "starting"}
stop = threading.Event()
session: DoomscrollSession | None = None
feed_handle = None


def run_loop(args):
    global latest, session, feed_handle
    if getattr(args, "shorts", False):
        from flyscroll.shorts import YouTubeShortsFeed

        feed_handle = YouTubeShortsFeed(
            url=args.shorts_url,
            headless=args.shorts_headless,
        )
        feed = feed_handle
        n_reels = "live-shorts"
    else:
        feed = Feed(
            load_feed(
                Path(args.reels) if args.reels else None,
                seed=args.seed,
                n_reels=args.n_reels,
            )
        )
        feed_handle = feed
        n_reels = len(feed.reels)
    config = SessionConfig(
        scale=args.scale,
        frame_ms=args.frame_ms,
        learning=not args.no_learning,
        interest_threshold=args.threshold,
        min_watch_seconds=args.min_watch,
        max_watch_seconds=args.max_watch,
        seed=args.seed,
        shorts=bool(getattr(args, "shorts", False)),
        boredom_dwell_seconds=getattr(args, "boredom_dwell", 1.6),
        peak_fraction=getattr(args, "peak_fraction", 0.40),
    )
    session = DoomscrollSession(feed, config)
    print(
        json.dumps(
            {
                "status": "running",
                "run_id": session.run_id,
                "port": args.port,
                "scale": args.scale,
                "reels": n_reels,
                "shorts": bool(getattr(args, "shorts", False)),
            }
        ),
        flush=True,
    )
    try:
        while not stop.is_set():
            began = time.monotonic()
            state = session.step()
            latest = {k: v for k, v in state.items()}
            target = args.frame_ms / 1000.0
            elapsed = time.monotonic() - began
            if elapsed < target:
                time.sleep(target - elapsed)
    finally:
        closer = getattr(feed_handle, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            body = json.dumps({"ok": True, "status": latest.get("status")}).encode()
            return self._send(200, body, "application/json")
        if path == "/state":
            body = json.dumps(latest).encode()
            return self._send(200, body, "application/json")
        if path == "/anatomy":
            if session is None:
                return self._send(503, b'{"status":"starting"}', "application/json")
            body = json.dumps(session.anatomy).encode()
            return self._send(200, body, "application/json")
        if path in ("/", "/index.html"):
            html = (UI / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        if path == "/app.css":
            return self._send(200, (UI / "app.css").read_bytes(), "text/css")
        if path == "/app.js":
            return self._send(200, (UI / "app.js").read_bytes(), "application/javascript")
        if path == "/cloud.js":
            return self._send(200, (UI / "cloud.js").read_bytes(), "application/javascript")
        self._send(404, b"not found", "text/plain")


def _init_macos_graphics() -> None:
    """Initialize AppKit/CGS on the main thread before ScreenCaptureKit touches CG."""
    import sys

    if sys.platform != "darwin":
        return
    from AppKit import NSApplication

    # Required before any CGS/ScreenCaptureKit call or macOS aborts with
    # CGS_REQUIRE_INIT (did_initialize) from a background thread.
    NSApplication.sharedApplication()


def _serve_http(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Spectator at http://127.0.0.1:{port}/", flush=True)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--scale", choices=["full", "visual"], default="visual")
    parser.add_argument("--reels", type=str, default="", help="Directory of local videos")
    parser.add_argument("--n-reels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-ms", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=14.0)
    parser.add_argument("--min-watch", type=float, default=0.35, help="Scroll refractory seconds")
    parser.add_argument(
        "--max-watch",
        type=float,
        default=5.0,
        help="Local-feed safety timeout; Shorts use actual video completion",
    )
    parser.add_argument(
        "--boredom-dwell",
        type=float,
        default=1.6,
        help="Seconds phasic novelty must stay low before a bored scroll",
    )
    parser.add_argument(
        "--peak-fraction",
        type=float,
        default=0.40,
        help="Bored when phasic < this fraction of the reel's peak novelty",
    )
    parser.add_argument("--no-learning", action="store_true")
    parser.add_argument(
        "--shorts",
        action="store_true",
        help="Watch live YouTube Shorts via Playwright (no login)",
    )
    parser.add_argument(
        "--shorts-url",
        type=str,
        default="https://www.youtube.com/shorts",
        help="Shorts start URL",
    )
    parser.add_argument(
        "--shorts-headless",
        action="store_true",
        help="Hide the Chromium window (spectator still shows captures)",
    )
    args = parser.parse_args(argv)
    if not args.reels:
        args.reels = ""

    def handle_sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # ScreenCaptureKit + Playwright need the macOS graphics stack initialized
    # on the main thread; run the doomscroll loop there and HTTP in the background.
    if args.shorts:
        _init_macos_graphics()
        server = _serve_http(args.port)

        def http_loop():
            try:
                while not stop.is_set():
                    server.handle_request()
            finally:
                server.server_close()

        http_thread = threading.Thread(target=http_loop, name="flyscroll-http", daemon=True)
        http_thread.start()
        try:
            run_loop(args)
        finally:
            stop.set()
            http_thread.join(timeout=2.0)
        return

    thread = threading.Thread(target=run_loop, args=(args,), daemon=True)
    thread.start()
    server = _serve_http(args.port)
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        stop.set()
        thread.join(timeout=8.0)
        server.server_close()


if __name__ == "__main__":
    main()
