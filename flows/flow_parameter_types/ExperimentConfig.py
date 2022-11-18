from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    exp_path: str = "/path/to/experiment.yaml"
    sample_name: str = "Sample"
    acquisition: str = "run"
    start_section_num: int = 1
    end_section_num: int = 10
    tile_grid_num: int = 1
