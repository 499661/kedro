from mlflow.tracking import MlflowClient

def latest_model_version(model_name: str) -> int:
    client = MlflowClient()
    versions = [v.version for v in client.get_registered_model(name=model_name).latest_versions if v.status == 'READY']

    if versions:
        return max(versions)
    else:
        raise Exception(
            f"There is no READY model with name {model_name}")

def model_uri(model_name: str, env: str) -> str:

    if env in ['Staging', 'Production']:
        return "models:/{}/{}".format(model_name, env)
    else:
        latest_version = latest_model_version(model_name)
        return "models:/{}/{}".format(model_name, latest_version)
