"""Machine and per-process memory readings on macOS, through ctypes into libSystem.

There is no fallback for other platforms: `ps` RSS leaves out compressed pages, so it under-reads exactly the process
the memory watch exists to stop, and MLX itself only runs on Apple Silicon macOS.
"""

import ctypes
import errno
import functools
import sys
from dataclasses import dataclass

from ytverbatim.errors import DependencyError

_RUSAGE_INFO_V4 = 4
_UNSUPPORTED = "the memory watch needs macOS (proc_pid_rusage and kern.memorystatus_level)"


class _RusageInfoV4(ctypes.Structure):
    # Field order of `struct rusage_info_v4` in <sys/resource.h>; only the footprint fields are read.
    _fields_ = (
        ("ri_uuid", ctypes.c_uint8 * 16),
        *(
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
            )
        ),
    )


@dataclass(frozen=True)
class Footprint:
    """A process's physical footprint now and the most it has reached, in bytes.

    Unlike RSS, the footprint counts compressed and swapped pages, so a process squeezed by memory pressure still shows
    its full size.
    """

    current: int
    peak: int


@functools.cache
def _libc() -> ctypes.CDLL:
    if sys.platform != "darwin":
        raise DependencyError(_UNSUPPORTED)
    lib = ctypes.CDLL(None, use_errno=True)
    try:
        lib.proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        lib.proc_pid_rusage.restype = ctypes.c_int
        lib.sysctlbyname.argtypes = (
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        lib.sysctlbyname.restype = ctypes.c_int
    except AttributeError as exc:
        raise DependencyError(_UNSUPPORTED) from exc
    return lib


def footprint(pid: int) -> Footprint | None:
    """Return the footprint of `pid`, or None once it has exited."""
    info = _RusageInfoV4()
    if _libc().proc_pid_rusage(pid, _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        if code == errno.ESRCH:
            return None
        raise DependencyError(f"the memory watch cannot read the footprint of process {pid}: errno {code}")
    return Footprint(info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)


def available_percent() -> int:
    """Percent of memory the kernel counts as available (`kern.memorystatus_level`); low means pressure."""
    return _sysctl("kern.memorystatus_level", ctypes.c_int()).value


def physical_memory_bytes() -> int:
    return _sysctl("hw.memsize", ctypes.c_uint64()).value


def _sysctl[T: ctypes.c_int | ctypes.c_uint64](name: str, value: T) -> T:
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _libc().sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        code = ctypes.get_errno()
        raise DependencyError(f"{_UNSUPPORTED}: sysctl {name} failed with errno {code}")
    return value
