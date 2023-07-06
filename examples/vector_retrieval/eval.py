import sys
from pathlib import Path

import benchllm

current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir))
from utils import initiate_test_faiss


@benchllm.test(suite=".")
def run(input: str):
    qa = initiate_test_faiss()
    resp = qa.run(input)
    return resp
