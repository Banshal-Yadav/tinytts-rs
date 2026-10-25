"""Measure RSS at each stage: baseline, after model load, after 10s of audio synthesis."""

import ctypes
import ctypes.wintypes as wt
import time

from paradee import SAMPLE_RATE, Paradee

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

TEXT = (
    "The history of speech synthesis is a long and surprisingly stubborn one. "
    "For most of the twentieth century, the best a machine could do was imitate a human voice "
    "in the crudest possible way, buzzing through a bank of analog filters and hoping the "
    "listener would fill in the gaps. The results were intelligible, barely, and they sounded "
    "like a robot trapped inside a pipe."
)


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wt.DWORD]
psapi.GetProcessMemoryInfo.restype = wt.BOOL
k32.GetCurrentProcess.restype = wt.HANDLE


def mem():
    c = PROCESS_MEMORY_COUNTERS_EX()
    c.cb = ctypes.sizeof(c)
    ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return c.WorkingSetSize, c.PeakWorkingSetSize, c.PrivateUsage


def mb(n):
    return n / (1024 * 1024)


def report(label, base=None):
    ws, peak, priv = mem()
    delta = f"  (+{mb(ws - base):6.1f} MB)" if base is not None else ""
    print(f"{label:<28} WS {mb(ws):7.1f} MB   peak {mb(peak):7.1f} MB   private {mb(priv):7.1f} MB{delta}")
    return ws


print(f"python {__import__('sys').version.split()[0]}")
base = report("baseline (interpreter)")

t0 = time.perf_counter()
tts = Paradee(threads=4)
t_load = time.perf_counter() - t0
after_load = report("after model load", base)
print(f"{'':<28} load time {t_load * 1000:.0f} ms")

t0 = time.perf_counter()
audio = tts(TEXT)
t_synth = time.perf_counter() - t0
dur = len(audio) / SAMPLE_RATE
after_synth = report("after synthesis", base)
print(f"{'':<28} {dur:.2f}s audio in {t_synth * 1000:.0f} ms  (RTF {dur / t_synth:.1f}x)")

# second call: warm path, no session load
t0 = time.perf_counter()
tts(TEXT)
t_warm = time.perf_counter() - t0
report("after 2nd synthesis", base)
print(f"{'':<28} warm synth {t_warm * 1000:.0f} ms")

print()
print(f"model load cost      {mb(after_load - base):6.1f} MB")
print(f"synthesis cost       {mb(after_synth - after_load):6.1f} MB")
print(f"total over baseline  {mb(after_synth - base):6.1f} MB")
