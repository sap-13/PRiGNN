
from dataclasses import dataclass, asdict
import jsonlines
import mlflow
import pathlib
from typing import Any, Dict, Optional

@dataclass
class TrialResult:
    round_no: int
    metric_name: str
    metric_value: float
    meta: Dict[str, Any]
    architecture_config: Dict[str, Any]
    code: str
    code_hash: str
    cost: Optional[float] = None

class ResultStore:
    def __init__(
        self,
        root: pathlib.Path | str = "results",
        experiment_name: str = "prignn-search",
    ):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_file = self.root / "trials.jsonl"
        self.writer = jsonlines.open(self.log_file, mode="a")

        mlflow.set_tracking_uri(f"file://{self.root.absolute()}/mlflow")
        mlflow.set_experiment(experiment_name)

    def log_trial(self, tr: TrialResult) -> None:
        """
        Logs a trial to MLflow and the local jsonlines file.
        """
        with mlflow.start_run(run_name=f"round_{tr.round_no:03d}"):
            mlflow.log_params(tr.meta)
            mlflow.log_param("round_no", tr.round_no)
            mlflow.log_param("code_hash", tr.code_hash)
            mlflow.log_metric(tr.metric_name, tr.metric_value)
            if tr.cost is not None:
                mlflow.log_metric("cost_usd", tr.cost)

            code_artifact_path = self.root / f"code_round_{tr.round_no:03d}.py"
            code_artifact_path.write_text(tr.code, encoding="utf-8")
            mlflow.log_artifact(str(code_artifact_path))
            code_artifact_path.unlink()

            self.writer.write(asdict(tr))

def hash_code(code_str: str) -> str:
    """Computes a SHA256 hash of a string."""
    import hashlib
    return hashlib.sha256(code_str.encode("utf-8")).hexdigest()

