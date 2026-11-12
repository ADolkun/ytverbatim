import os
import subprocess
import sys

from ytverbatim import sysmem


def test_footprint_counts_pages_the_process_touched() -> None:
    before = sysmem.footprint(os.getpid())
    held = b"x" * (64 << 20)
    after = sysmem.footprint(os.getpid())
    assert before is not None
    assert after is not None
    assert after.current - before.current >= 60 << 20
    assert after.peak >= after.current
    assert held


def test_footprint_of_an_exited_process_is_none() -> None:
    child = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, check=True)
    assert sysmem.footprint(int(child.stdout)) is None


def test_machine_readings_are_plausible() -> None:
    assert 0 <= sysmem.available_percent() <= 100
    assert sysmem.physical_memory_bytes() >= 1 << 30
