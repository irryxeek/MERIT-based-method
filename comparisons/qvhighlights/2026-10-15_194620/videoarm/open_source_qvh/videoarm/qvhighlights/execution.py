"""Bounded cross-query scheduling; every query retains its sequential tool loop."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


class PauseRequested(BaseException):
    """Pause at a request boundary without classifying it as model failure."""


def run_bounded(rows, workers, stop_event, run_one, on_complete):
    if workers < 1:
        raise ValueError("workers must be positive")
    iterator = iter(rows)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qvh-query") as pool:
        active = {}

        def submit_next():
            if stop_event.is_set():
                return False
            row = next(iterator, None)
            if row is None:
                return False
            active[pool.submit(run_one, row)] = row
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                on_complete(row, future.result())
                submit_next()
