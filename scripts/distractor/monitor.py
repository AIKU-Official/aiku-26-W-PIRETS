import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional


def _has_flag(args: List[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _build_cmd(
    python_exec: str,
    script_path: str,
    eval_args: List[str],
    checkpoint_dir: Path,
    heartbeat_file: Path,
) -> List[str]:
    cmd = [python_exec, script_path]
    passthrough = list(eval_args)

    if not _has_flag(passthrough, "--resume-encoding"):
        cmd.append("--resume-encoding")
    if not _has_flag(passthrough, "--encoding-checkpoint-dir"):
        cmd.extend(["--encoding-checkpoint-dir", str(checkpoint_dir)])
    if not _has_flag(passthrough, "--heartbeat-file"):
        cmd.extend(["--heartbeat-file", str(heartbeat_file)])

    cmd.extend(passthrough)
    return cmd


def _stream_output(pipe) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            print(line, end="", flush=True)
    finally:
        pipe.close()


def _terminate(proc: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _read_heartbeat(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", default="evaluate_distractor_dpr.py")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--check-interval", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--checkpoint-dir", default="outputs/dpr_encoding_checkpoints")
    parser.add_argument("--heartbeat-file", default="outputs/dpr_encoding_heartbeat.json")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    eval_args = list(args.eval_args)
    if eval_args and eval_args[0] == "--":
        eval_args = eval_args[1:]

    checkpoint_dir = Path(args.checkpoint_dir)
    heartbeat_file = Path(args.heartbeat_file)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

    restart_count = 0

    while True:
        if heartbeat_file.exists():
            heartbeat_file.unlink()

        cmd = _build_cmd(args.python, args.script, eval_args, checkpoint_dir, heartbeat_file)
        print(f"[monitor] launch #{restart_count + 1}: {' '.join(cmd)}", flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        t = threading.Thread(target=_stream_output, args=(proc.stdout,), daemon=True)
        t.start()

        launched_at = time.time()
        last_progress = launched_at
        encoding_seen = False
        encoding_started_once = False
        stalled = False

        while proc.poll() is None:
            hb = _read_heartbeat(heartbeat_file)
            now = time.time()
            if hb is not None:
                stage = str(hb.get("stage", ""))
                if stage.startswith("encoding:"):
                    encoding_seen = True
                    encoding_started_once = True
                    try:
                        hb_mtime = heartbeat_file.stat().st_mtime
                    except OSError:
                        hb_mtime = now
                    if hb_mtime > last_progress:
                        last_progress = hb_mtime
                elif stage == "encoding_done":
                    encoding_seen = False
                    last_progress = now

            if encoding_seen:
                if now - last_progress > args.timeout:
                    stalled = True
                    print(
                        f"[monitor] no encoding progress for {args.timeout:.1f}s; restarting...",
                        flush=True,
                    )
                    _terminate(proc)
                    break
            else:
                # Startup timeout is only for the phase before encoding begins.
                # After encoding starts (and especially during recall/metrics), do not timeout.
                if (not encoding_started_once) and (now - launched_at > args.startup_timeout):
                    stalled = True
                    print(
                        f"[monitor] startup timeout {args.startup_timeout:.1f}s exceeded; restarting...",
                        flush=True,
                    )
                    _terminate(proc)
                    break

            time.sleep(args.check_interval)

        t.join(timeout=2.0)
        code = proc.poll()

        if not stalled:
            if code == 0:
                print("[monitor] completed successfully", flush=True)
                return 0
            print(f"[monitor] process exited with code {code}", flush=True)
            return int(code if code is not None else 1)

        restart_count += 1
        if restart_count > args.max_restarts:
            print(f"[monitor] reached max restarts ({args.max_restarts})", flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
