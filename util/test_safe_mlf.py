# добавь импорты все
import os
import mlflow
import mlflow.sklearn


def log_afc_model1(afc):
    TRACKING_SERVER_HOST = "127.0.0.1"
    TRACKING_SERVER_PORT = 5003
    TRACKING_SERVER_URL = (
        f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
    )
    EXPERIMENT_NAME = "autofeat_classifier"
    RUN_NAME = "autofeat_classifier_run_1"

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

    mlflow.set_tracking_uri(TRACKING_SERVER_URL)
    mlflow.set_registry_uri(TRACKING_SERVER_URL)

    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment_id:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        experiment_id = mlflow.get_experiment_by_name(
            EXPERIMENT_NAME
        ).experiment_id

    artifact_path = "afc"

    with mlflow.start_run(
        run_name=RUN_NAME, experiment_id=experiment_id
    ) as run:
        run_id = run.info.run_id

        afc_info = mlflow.sklearn.log_model(afc, artifact_path=artifact_path)
