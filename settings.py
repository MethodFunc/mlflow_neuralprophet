import argparse
from pathlib import Path
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature


def args_setting():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.estimator = 'nprophet'
    args.path = 'input data path (file and folder support)'
    args.col_name = ['WS', 'WD']

    # hyperopt setting
    args.eval = 10

    args.csvs = './csv_output'

    if not Path(args.csvs).exists():
        Path(args.csvs).mkdir(parents=True, exist_ok=True)

    return args


def mlflow_setting():
    custom_env_ = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["neuralprophet"],
        additional_conda_channels=None,
    )

    input_schema = Schema([
        ColSpec("datetime", "ds"),
        ColSpec("double", "WS"),
        ColSpec("double", "WD"),
        ColSpec("double", "y"),
    ])

    output_schmea = Schema(
        [ColSpec("double", "ActivePower")]
    )

    signature = ModelSignature(inputs=input_schema, outputs=output_schmea)

    return custom_env_, signature
