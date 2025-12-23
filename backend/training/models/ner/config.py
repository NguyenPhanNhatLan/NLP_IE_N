from dataclasses import dataclass
import mlflow.pyfunc
import pandas as pd
from typing import Any, Dict, List
from training.models.ner.features import build_features

@dataclass
class CRFConfig:
    # CRF hyperparams (lbfgs thường tốt)
    algorithm: str = "lbfgs"
    c1: float = 0.1          # L1 regularization
    c2: float = 0.1          # L2 regularization
    max_iterations: int = 200
    all_possible_transitions: bool = True

    # feature window
    window: int = 2          # nhìn trái/phải 2 token


class NerCrfPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, window: int = 2):
        self.window = int(window)
        self.crf = None

    def load_context(self, context):
        import joblib
        self.crf = joblib.load(context.artifacts["crf_joblib"])

    def _extract_tokens_batch(self, model_input: Any):
        if isinstance(model_input, pd.DataFrame):
            if "tokens" not in model_input.columns:
                raise TypeError('PyFunc input DataFrame must have column "tokens".')
            tokens_batch = model_input["tokens"].tolist()

        # Series of list[str]
        elif isinstance(model_input, pd.Series):
            tokens_batch = model_input.tolist()

        # list / tuple
        elif isinstance(model_input, (list, tuple)):
            if len(model_input) == 0:
                tokens_batch = []
            else:
                # single sample: ["a","b"] -> wrap
                if isinstance(model_input[0], str):
                    tokens_batch = [list(model_input)]
                else:
                    # batch: [["a","b"], ...]
                    tokens_batch = [list(x) for x in model_input]

        else:
            raise TypeError(
                "Unsupported input for NerCrfPyFunc.predict. "
                "Use DataFrame with column 'tokens' or list tokens."
            )

        # validate
        for i, row in enumerate(tokens_batch):
            if not isinstance(row, list) or (row and not isinstance(row[0], str)):
                raise TypeError(f"Row {i} in tokens_batch must be List[str]. Got: {type(row)}")
        return tokens_batch

    def predict(self, context, model_input):
        tokens_batch = self._extract_tokens_batch(model_input)
        X_feats = build_features(tokens_batch, window=self.window)
        y_pred = self.crf.predict(X_feats)  # List[List[str]]
        return y_pred