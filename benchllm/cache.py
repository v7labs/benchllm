import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from benchllm.data_types import Evaluation, Prediction
from benchllm.evaluator import Evaluator
from benchllm.input_types import Json
from benchllm.listener import EvaluatorListener


class MemoryValue(BaseModel):
    passed: bool
    score: float


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

    def lookup(self, answer1: Json, answer2: Json) -> Optional[MemoryValue]:
        result = self._data.get(self._key(answer1, answer2), None)
        if result:
            return MemoryValue(**result)
        return None

    def store(self, answer1: Json, answer2: Json, value: MemoryValue) -> None:
        key = self._key(answer1, answer2)
        self._data[key] = value.dict()

    def evaluate_prediction(self, prediction: Prediction) -> list[Evaluator.Candidate]:
        uncached_expectations = []
        candidates = []
        for expected in prediction.test.expected:
            lookup = self.lookup(expected, prediction.output)
            if lookup is None:
                uncached_expectations.append(expected)
            else:
                candidates.append(Evaluator.Candidate(prediction=prediction.output, expected=expected, **lookup.dict()))

        # If any of the cached candidates passed, we return them.
        if any([candidate.passed for candidate in candidates]):
            self._num_cache_hits += 1
            return candidates

        # If all expectations were found in the cache but were negative matches,
        # we increment the cache hits counter and return None as there's no match.
        if not uncached_expectations:
            self._num_cache_hits += 1
            return candidates

        self._num_cache_misses += 1
        # set prediction.test.expected to only the ones that were not cached
        prediction = Prediction(**prediction.dict())
        prediction.test.expected = uncached_expectations
        candidates = self._evaluator.evaluate_prediction(prediction)
        for candidate in candidates:
            self.store(candidate.expected, candidate.prediction, MemoryValue(**candidate.dict()))
        return candidates

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
