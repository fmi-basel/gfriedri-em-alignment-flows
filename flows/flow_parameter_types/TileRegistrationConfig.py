from typing import List

from pydantic import BaseModel


class MeshIntegrationConfig(BaseModel):
    dt: float = 0.001
    gamma: float = 0.0
    k0: float = 0.01
    k: float = 0.1
    stride: int = 20
    num_iters: int = 1000
    max_iters: int = 20000
    stop_v_max: float = 0.001
    dt_max: float = 100.0
    prefer_orig_order: bool = True
    start_cap: float = 1.0
    final_cap: float = 10.0
    remove_drift: bool = True


class RegistrationConfig(BaseModel):
    overlaps_x: List[int] = [200, 300, 400]
    overlaps_y: List[int] = [200, 300, 400]
    min_overlap: int = 20
    patch_size: List[int] = [80, 80]
    batch_size: int = 8000
    min_peak_ratio: float = 1.4
    min_peak_sharpness: float = 1.4
    max_deviation: float = 5.0
    max_magnitude: float = 0.0
    min_patch_size: int = 10
    max_gradient: float = -1.0
    reconcile_flow_max_deviation: float = -1.0
