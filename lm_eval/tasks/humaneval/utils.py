import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def sanitize_response(response: str) -> str:
    for end_sequence in ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]:
        if end_sequence in response:
            response = response[: response.index(end_sequence)]

    return "    " + response.strip()


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + sanitize_response(r) for r in resp] for resp, doc in zip(resps, docs)]
