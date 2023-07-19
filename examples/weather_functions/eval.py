from forecast import run

import benchllm


@benchllm.test()
def eval(input: str):
    return run(input)
