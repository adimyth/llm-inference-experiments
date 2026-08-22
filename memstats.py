"""Host RAM and Metal VRAM around a block of work.

torch.mps.current_allocated_memory is what tensors hold; driver_allocated_memory is
what Metal has reserved from the system, which is the number that decides whether
you are about to swap.
"""

from dataclasses import dataclass

import psutil
import torch

GB = 1024**3


@dataclass
class MemSnapshot:
    rss_gb: float
    sys_used_gb: float
    sys_avail_gb: float
    swap_used_gb: float
    mps_alloc_gb: float
    mps_driver_gb: float

    @classmethod
    def take(cls):
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        mps_a = mps_d = 0.0
        if torch.backends.mps.is_available():
            mps_a = torch.mps.current_allocated_memory() / GB
            mps_d = torch.mps.driver_allocated_memory() / GB
        return cls(
            rss_gb=psutil.Process().memory_info().rss / GB,
            sys_used_gb=vm.used / GB,
            sys_avail_gb=vm.available / GB,
            swap_used_gb=sw.used / GB,
            mps_alloc_gb=mps_a,
            mps_driver_gb=mps_d,
        )

    def line(self, label):
        return (
            f"{label:<18} rss {self.rss_gb:5.2f} GB | mps alloc {self.mps_alloc_gb:5.2f} "
            f"GB | mps driver {self.mps_driver_gb:5.2f} GB | sys avail "
            f"{self.sys_avail_gb:5.1f} GB | swap {self.swap_used_gb:4.2f} GB"
        )

    def as_dict(self):
        return self.__dict__.copy()
