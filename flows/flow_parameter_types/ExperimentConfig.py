from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    exp_path: str = "/path/to/experiment.yaml"
    sample_name: str = None
    acquisition: str = None
    start_section_num: int = None
    end_section_num: int = None
    tile_grid_num: int = None
