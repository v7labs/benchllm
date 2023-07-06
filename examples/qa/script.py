from benchllm import StringMatchEvaluator, Test, Tester

tests = [
    Test(input="What's 1+1?", expected=["2", "It's 2"]),
    Test(input="What's Obama's first name?", expected=["Barack"]),
]

tester = Tester(lambda _: 2)
tester.add_tests(tests)
predictions = tester.run()

evaluator = StringMatchEvaluator()
evaluator.load(predictions)
report = evaluator.run()

print(report)
