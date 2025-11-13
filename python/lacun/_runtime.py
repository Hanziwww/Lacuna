import os
import multiprocessing

_default_threads = max(1, multiprocessing.cpu_count())
_current_threads = _default_threads

def set_num_threads(n: int) -> None:
    global _current_threads
    if n and n > 0:
        _current_threads = int(n)
        os.environ["RAYON_NUM_THREADS"] = str(_current_threads)

def get_num_threads() -> int:
    # If user set env externally, honor it
    env = os.environ.get("RAYON_NUM_THREADS")
    if env:
        try:
            return max(1, int(env))
        except Exception:
            return _current_threads
    return _current_threads
