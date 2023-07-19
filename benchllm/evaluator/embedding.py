import numpy as np
import openai

from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator


class EmbeddingEvaluator(Evaluator):
    def __init__(self, *, engine: str = "text-similarity-davinci-001", threshold: float = 0.9, workers: int = 1):
        super().__init__(workers=workers)
        self._engine = engine
        self._threshold = threshold

    def evaluate_prediction(self, prediction: Prediction) -> list[Evaluator.Candidate]:
        output_embedding = get_embedding(prediction.output, engine=self._engine)
        candidates = []
        for expected in prediction.test.expected:
            expected_embedding = get_embedding(expected, engine=self._engine)
            similarity = cosine_similarity(output_embedding, expected_embedding)
            candidates.append(
                Evaluator.Candidate(
                    prediction=prediction.output,
                    expected=expected,
                    score=similarity,
                    passed=similarity > self._threshold,
                )
            )
        return candidates


# these also exist in openai.embeddings_utils but have additional dependencies
def get_embedding(text: str, engine: str, **kwargs) -> list[float]:
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], engine=engine, **kwargs)["data"][0]["embedding"]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
