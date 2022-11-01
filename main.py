import os
import mlflow
from pathlib import Path

from bin.loader import load_dataframe, split_dataframe
from bin.automl_tool import hyperopt_fit
from settings import args_setting, mlflow_setting
from bin.models import NeuralModel
from bin.registers import model_register
from bin.wrapper import NeuralProphetWrapper

# Tracking URI setting
os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://dev:dev1234@localhost:5432/mlflow_test'
# sftp server
ARTIFACT_URI = "sftp://dev:dev1234@localhost:14022/mlflow/"


def main(path, col_name):
    dataframe = load_dataframe(path, col_name)
    train_data, validation_data = split_dataframe(dataframe)
    best_params = hyperopt_fit(train_data, validation_data, args)

    model_name = Path(path).name.split('.')[0]

    try:
        mlflow.create_experiment(model_name, artifact_location=ARTIFACT_URI)
    except mlflow.exceptions.MlflowException:
        print(f'{model_name} is exists')
    mlflow.set_experiment(model_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            'n_forecasts': 24,
            'n_lags': 24})
        mlflow.log_params(best_params)

        run_id = run.info.run_id
        model = NeuralModel(**best_params, n_forecasts=24, n_lags=24)
        model = model.add_lagged_regressor(names=col_name, normalize=True)

        result = model.fit(train_data, freq='h', validation_df=validation_data)
        input_example = train_data.sample(n=1)

        if not Path(f'{args.csvs}/{model_name}').exists():
            Path(f'{args.csvs}/{model_name}').mkdir(parents=True, exist_ok=True)

        result.to_csv(f'{args.csvs}/{model_name}/{run_id}.csv')
        mlflow.log_artifact(f'{args.csvs}/{model_name}/{run_id}.csv')

        mlflow.pyfunc.log_model("model",
                                python_model=NeuralProphetWrapper(model),
                                signature=signature)

    model_register(model_name, ARTIFACT_URI, run_id, args.path)


if __name__ == '__main__':
    args = args_setting()
    custom_env, signature = mlflow_setting()

    if Path(args.path).is_file():
        main(args.path, args.col_name)
    elif Path(args.path).is_dir():
        file_list = os.listdir(args.path)
        fullpath_list = [os.path.join(args.path, file) for file in file_list]
        fullpath_list.sort()
        for path in fullpath_list:
            path = Path(path)
            if path.suffix == '.csv':
                print(f'{path.name.split(".")[0]} Start')
                main(path, args.col_name)

