from pydantic import BaseModel


class AcquisitionConfig(BaseModel):
    sbem_root_dir: str = ""
    acquisition: str = "run_0"
    tile_grid: str = "g0001"
    thickness: float = 25
    resolution_xy: float = 11
    tile_width: int = 3072
    tile_height: int = 2304
    tile_overlap: int = 220


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
    overlaps_x: list[int] = [200, 300, 400]
    overlaps_y: list[int] = [200, 300, 400]
    min_overlap: int = 20
    min_range: list[int] = [10, 100, 0]
    patch_size: list[int] = [80, 80]
    batch_size: int = 8000
    min_peak_ratio: float = 1.4
    min_peak_sharpness: float = 1.4
    max_deviation: int = 5
    max_magnitude: int = 0
    min_patch_size: int = 10
    max_gradient: float = -1.0
    reconcile_flow_max_deviation: float = -1.0


class WarpConfig(BaseModel):
    margin: int = 20
    use_clahe: bool = True
    kernel_size: int = 1024
    clip_limit: float = 0.01
    nbins: int = 256
    warp_parallelism: int = 5
