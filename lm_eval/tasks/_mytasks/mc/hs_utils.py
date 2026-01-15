import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            **doc,
            "ctx": preprocess(doc["activity_label"] + ": " + ctx),
            "activity_label": preprocess(doc["activity_label"]),
            "endings": [preprocess(ending) for ending in doc["endings"]],
        }
        return out_doc

    return dataset.map(_process_doc)
