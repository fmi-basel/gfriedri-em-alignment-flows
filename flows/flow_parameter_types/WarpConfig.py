from pydantic import BaseModel


class WarpConfig(BaseModel):
    stride: int = 20
    margin: int = 20
    use_clahe: bool = True
    kernel_size: int = 1024
    clip_limit: float = 0.01
    nbins: int = 256
