import benchllm
from benchllm import SimilarityInput
from benchllm.similarity import semantically_similar


@benchllm.test(suite=".")
def run(input: SimilarityInput):
    return semantically_similar(input.prompt_1, input.prompt_2)
