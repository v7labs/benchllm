import json
from pathlib import Path
from typing import Optional

from benchllm.data_types import Evaluation, Prediction
from benchllm.evaluator import Evaluator
from benchllm.input_types import Json
from benchllm.listener import EvaluatorListener


class MemoryCache(Evaluator):
    """Caches the results of the evaluator in memory"""

    def __init__(self, evaluator: Evaluator):
        super().__init__(workers=evaluator.workers)
        self._data: dict = {}
        self._evaluator = evaluator
        self._num_cache_misses = 0
        self._num_cache_hits = 0

    def _key(self, answer1: Json, answer2: Json) -> str:
        key1, key2 = json.dumps([answer1, answer2]), json.dumps([answer2, answer1])
        return key1 if key1 < key2 else key2

    def lookup(self, answer1: Json, answer2: Json) -> Optional[bool]:
        return self._data.get(self._key(answer1, answer2), None)

    def store(self, answer1: Json, answer2: Json, value: bool) -> None:
        key = self._key(answer1, answer2)
        self._data[key] = value

    def evaluate_prediction(self, prediction: Prediction) -> Optional[Evaluator.Match]:
        uncached_expectations = []
        for expected in prediction.test.expected:
            lookup = self.lookup(expected, prediction.output)
            if lookup is None:
                uncached_expectations.append(expected)
            elif lookup:
                # If we find a positive match we can stop comparing and just return.
                # For negative matches we still need to check the other expected answers.
                self._num_cache_hits += 1
                return Evaluator.Match(prediction=prediction.output, expected=expected)

        # If all expectations were found in the cache but were negative matches,
        # we increment the cache hits counter and return None as there's no match.
        if not uncached_expectations:
            self._num_cache_hits += 1
            return None

        self._num_cache_misses += 1
        # set prediction.test.expected to only the ones that were not cached
        prediction = Prediction(**prediction.dict())
        prediction.test.expected = uncached_expectations
        result = self._evaluator.evaluate_prediction(prediction)
        if result:
            self.store(result.expected, result.prediction, True)
        else:
            for expected in prediction.test.expected:
                self.store(expected, prediction.output, False)
        return result

    @property
    def num_cache_hits(self) -> int:
        return self._num_cache_hits

    @property
    def num_cache_misses(self) -> int:
        return self._num_cache_misses


class FileCache(MemoryCache, EvaluatorListener):
    """Caches the results of the evaluator in a json file"""

    def __init__(self, evaluator: Evaluator, path: Path):
        super().__init__(evaluator)
        self._path = path
        self.add_listener(self)
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                cache = json.loads(self._path.read_text(encoding="UTF-8"), parse_int=str)
                if cache["version"] != "1":
                    raise ValueError("Unsupported cache version")
                self._data = cache["entries"]
            except Exception:
                print(f"Failed to load cache file {self._path}")
                self._data = {}

    def _save(self) -> None:
        cache = {"entries": self._data, "version": "1"}
        self._path.write_text(json.dumps(cache, indent=4), encoding="UTF-8")

    def evaluate_ended(self, evaluations: list[Evaluation]) -> None:
        self._save()
