def doc_to_text(doc):
    idx = doc["sentence"].index("_")
    return doc["sentence"][:idx].strip()


def doc_to_target(doc):
    idx = doc["sentence"].index("_")
    idx_of_correct_option = {"1": 0, "2": 1}[doc["answer"]]
    correct_option = [doc["option1"], doc["option2"]][idx_of_correct_option]
    return correct_option + doc["sentence"][idx+1:]
