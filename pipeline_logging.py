#!/usr/bin/env python3
"""
Unified file + console logging for the VisDrone training pipeline.
==================================================================

Both entry points — ``train_and_export.py`` and
``tune_hyperparameters.py`` — emit a lot of output: our own progress
messages, Ultralytics' training / export chatter, and the stdout/stderr
of several *child processes* (the GPU-free TFLite export, Arm Vela, and
STEdgeAI for ``train_and_export.py``; Ray Tune workers for
``tune_hyperparameters.py``). This module gives every run one
timestamped log file under ``logs/`` that captures **all** of it, with
the child-process output interleaved into the same file in the order it
actually happened.

How the "everything, in order" guarantee works
-----------------------------------------------
``setup_logging`` installs a file-descriptor-level *tee*: it redirects
the process's stdout/stderr descriptors (1 and 2) into an OS pipe and
starts a single reader thread that copies every byte to (a) the original
console and (b) the log file. Because the redirection happens at the
*fd* level, anything that writes to fd 1/2 is captured — C-extension
prints (TensorFlow, PyTorch), Ultralytics' logger, bare ``print``
calls, and child processes that inherit those descriptors. A single
reader thread is the only writer to the log file, so output from the
main process and from subprocesses can never interleave mid-line or land
out of order.

For the child processes we launch ourselves we go one better:
``stream_subprocess`` captures the child's merged stdout/stderr through a
pipe and re-emits it line-by-line through the parent's logger, so each
child line is timestamped and tagged (``[export] …``, ``[vela] …``)
while still flowing through the same tee into the same file.

If the fd-level tee cannot be installed (e.g. stdout is not backed by a
real descriptor), ``setup_logging`` falls back to ordinary
console + file ``logging`` handlers. ``stream_subprocess`` still funnels
every child line through the logger in that mode, so subprocess output
stays sequentially integrated either way.

stdlib-only on purpose: this module is imported at startup by both CLIs
(and re-imported inside the isolated export subprocess), so it must not
drag in any heavy or optional dependency.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

# CSI escape sequences (colours, cursor moves) emitted by Ultralytics /
# tqdm. Stripped from the *file* copy so the log stays plain text; the
# console copy keeps them so interactive runs look normal.
_ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]")

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _Tee:
    """Redirect fds 1/2 into a pipe and fan the bytes out to console + file."""

    def __init__(self, log_path: Path) -> None:
        # Unbuffered binary file so the log is tailable in real time — a
        # training run lasts hours and we don't want a stale tail.
        self._file = open(log_path, "ab", buffering=0)

        # Flush whatever is still sitting in Python's buffers to the real
        # console before we move the descriptors out from under it.
        sys.stdout.flush()
        sys.stderr.flush()

        # Keep copies of the real console so the reader thread can echo to
        # it and so we can restore the descriptors on shutdown.
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)

        self._read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)  # fds 1 & 2 now hold the only write ends

        # Deliver each line through the pipe promptly.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._pump, name="log-tee", daemon=True
        )
        self._thread.start()

    def _pump(self) -> None:
        while True:
            try:
                chunk = os.read(self._read_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            # Console: verbatim (keep colours / live progress bars).
            try:
                os.write(self._saved_out, chunk)
            except OSError:
                pass
            # File: strip ANSI and turn carriage-return redraws into plain
            # lines so progress bars don't smear into one unreadable line.
            clean = _ANSI_RE.sub(b"", chunk)
            clean = clean.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            try:
                self._file.write(clean)
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # Restoring the real console onto fds 1/2 closes the pipe's write
        # ends, which is what gives the reader thread its EOF.
        try:
            os.dup2(self._saved_out, 1)
            os.dup2(self._saved_err, 2)
        except OSError:
            pass

        self._thread.join(timeout=5)

        for fd in (self._read_fd, self._saved_out, self._saved_err):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            self._file.flush()
            self._file.close()
        except OSError:
            pass


_active_tee: _Tee | None = None


def setup_logging(
    logger_name: str,
    *,
    log_dir: str | os.PathLike = "logs",
    prefix: str | None = None,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """
    Configure unified console + file logging for an entry-point script.

    Creates ``<log_dir>/<prefix>_<timestamp>.log``, installs the fd-level
    tee (see the module docstring), and returns the configured logger plus
    the log-file path. Call :func:`shutdown_logging` (or rely on the
    registered ``atexit`` hook) to flush and restore the console.
    """
    global _active_tee

    prefix = prefix or logger_name
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{prefix}_{stamp}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # If logging is re-initialised in the same process, tear the old tee
    # down first so we don't leak descriptors / threads.
    if _active_tee is not None:
        _active_tee.close()
        _active_tee = None

    try:
        _active_tee = _Tee(log_path)
    except Exception:
        # Fall back to plain handlers. We lose verbatim capture of
        # C-extension / subprocess output that bypasses our logger, but
        # stream_subprocess() still funnels every child line through the
        # logger, so the file stays complete for the cases that matter.
        _active_tee = None
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(file_handler)
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(console)
        # Capture Ultralytics' own logger into the file too, since without
        # the tee we cannot intercept its stdout writes.
        _attach_file_handler("ultralytics", file_handler)
    else:
        # With the tee active a single stdout handler suffices: its output
        # flows through the tee to *both* console and file.
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)

    atexit.register(shutdown_logging)

    logger.info("=" * 72)
    logger.info("  Logging to %s", log_path)
    logger.info("  Command : %s", " ".join(sys.argv))
    logger.info("  Workdir : %s", os.getcwd())
    logger.info("  PID     : %d", os.getpid())
    logger.info("=" * 72)
    return logger, log_path


def _attach_file_handler(name: str, handler: logging.Handler) -> None:
    lg = logging.getLogger(name)
    if handler not in lg.handlers:
        lg.addHandler(handler)


def setup_child_logging(
    logger_name: str, *, level: int = logging.INFO
) -> logging.Logger:
    """
    Minimal stdout logging for a subprocess child.

    The parent relays the child's stdout through :func:`stream_subprocess`,
    adding its own timestamp + tag, so the child must emit *plain* lines
    (no timestamp of its own) to avoid double-stamping. Returns the
    configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def stream_subprocess(
    cmd: list[str],
    *,
    logger: logging.Logger,
    tag: str,
    env: dict | None = None,
    cwd: str | None = None,
) -> int:
    """
    Run ``cmd``, relaying its merged stdout/stderr into ``logger`` one line
    at a time, and return the exit code.

    stderr is merged into stdout so the child's output keeps its natural
    ordering, and each line is prefixed with ``[tag]`` so it stays
    attributable in the unified log. Because the relay runs in the
    parent's calling thread, child output is sequentially integrated with
    the parent's own log lines.
    """
    logger.info("[%s] $ %s", tag, " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
        bufsize=1,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None
    with proc.stdout:
        for line in proc.stdout:
            logger.info("[%s] %s", tag, line.rstrip("\n"))
    rc = proc.wait()
    emit = logger.info if rc == 0 else logger.error
    emit("[%s] process exited with code %d", tag, rc)
    return rc


def shutdown_logging() -> None:
    """Flush the log file and restore the original console (idempotent)."""
    global _active_tee
    if _active_tee is not None:
        _active_tee.close()
        _active_tee = None
