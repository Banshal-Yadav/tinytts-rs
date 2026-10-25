"""Simulate a token stream feeding stream_play.py --stdin-stream."""

import subprocess
import sys
import time

TEXT = (
    "This is incremental streaming. Each sentence is spoken the moment it completes, "
    "not after the whole reply is written. The producer reads one character at a time "
    "from standard input. When it sees a period followed by a space, it flushes. "
    "That means the first sentence is already playing while the rest is still arriving. "
    "For a caller, this is the difference between waiting three seconds and waiting "
    "thirty milliseconds."
)

CMD = [
    sys.executable,
    "scripts/user/stream_play.py",
    "--stdin-stream",
    "--no-play",
]


def main() -> None:
    proc = subprocess.Popen(
        CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    t0 = time.perf_counter()
    for ch in TEXT:
        proc.stdin.write(ch)
        proc.stdin.flush()
        time.sleep(0.004)  # ~250 chars/sec, roughly a fast token stream
    proc.stdin.close()

    for line in proc.stdout:
        elapsed = time.perf_counter() - t0
        print(f"[{elapsed:6.2f}s] {line.rstrip()}")

    proc.wait()
    print(f"\ntotal: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
