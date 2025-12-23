import mlflow
from mlflow.tracking import MlflowClient

def register_model(run_id: str, name: str, artifact_path: str = "model") -> tuple[str, int]:
    if not run_id:
        raise ValueError("run_id is required")
    if not name:
        raise ValueError("name is required")

    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, name)
    return mv.name, int(mv.version)


def promote(name: str, version: int, stage: str = "Production", archive: bool = True) -> None:
    MlflowClient().transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive,
    )


def model_uri(name: str, stage: str = "Production") -> str:
    return f"models:/{name}/{stage}"


def register_and_promote(run_id: str, name: str, artifact_path: str = "model", stage: str = "Production") -> dict:
    reg_name, ver = register_model(run_id, name, artifact_path=artifact_path)
    promote(reg_name, ver, stage=stage, archive=True)
    return {"name": reg_name, "version": ver, "stage": stage, "uri": model_uri(reg_name, stage)}
