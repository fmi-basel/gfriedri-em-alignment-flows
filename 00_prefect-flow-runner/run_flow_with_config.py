import argparse

import yaml
from prefect.deployments import run_deployment


def start_flow_with_config(flow_name: str, config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_deployment(
        name=flow_name,
        parameters=dict(config),
        timeout=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_name")
    parser.add_argument("--config_file")
    args = parser.parse_args()

    start_flow_with_config(
        flow_name=args.flow_name,
        config_path=args.config_file,
    )
