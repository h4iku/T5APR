"""Script to locally preprocess the training data downloaded from:
https://github.com/lin-tan/CoCoNut-Artifact/releases/tag/training_data_1.0.0
"""

import pygments
from configs import (
    cache_dir,
    coconut_c2005,
    coconut_java2006,
    coconut_javascript2010,
    coconut_preprocessed_dir,
    coconut_python2010,
)
from datasets import (
    Dataset,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, String


def remove_comments(code: str, lexer) -> str:
    """Remove comments from given code snippet"""

    lexed_code = pygments.lex(code, lexer)
    comment_stripped_code = []
    for ttype, tvalue in lexed_code:
        if not (ttype in Comment or ttype in String.Doc):
            comment_stripped_code.append(tvalue)

    return "".join(comment_stripped_code).strip()


def batch_remove_comments(examples, lexer):
    """Remove comments from rem and add fields"""

    rem = [ex for ex in examples["rem"]]
    add = [ex for ex in examples["add"]]

    return {
        "rem": [remove_comments(ex, lexer) for ex in rem],
        "add": [remove_comments(ex, lexer) for ex in add],
    }


def ready_deduplicate(examples):
    """Removing whitespace tokens"""

    inputs = ["".join(ex.split()) for ex in examples["rem"]]
    targets = ["".join(ex.split()) for ex in examples["add"]]
    contexts = ["".join(ex.split()) for ex in examples["context"]]

    columns = {"src": inputs, "tgt": targets, "ctx": contexts}
    return columns


def filter_unchanged(example):
    """Filter unchanged and empty samples"""

    return example["src"] != example["tgt"]


def main():
    coconut_datasets = [
        coconut_python2010,
        coconut_c2005,
        coconut_javascript2010,
        coconut_java2006,
    ]
    columns = ["rem", "add", "context"]

    for lang_dataset in coconut_datasets:
        data_columns = []
        for column in columns:
            dataset = load_dataset(
                "text",
                data_files=f"{lang_dataset/column}.txt",
                split="train",
                cache_dir=cache_dir,
                features=Features({column: Value("string")}),
            )
            data_columns.append(dataset)

        raw_dataset = concatenate_datasets(data_columns, axis=1)
        print(raw_dataset)

        comment_stripped_dataset = raw_dataset.map(
            lambda ex: batch_remove_comments(
                ex, get_lexer_by_name(lang_dataset.parts[-2])
            ),
            batched=True,
            num_proc=4,
        )

        ready_dedup_dataset = comment_stripped_dataset.map(
            ready_deduplicate, batched=True, num_proc=4
        )
        ready_dedup_dataset.set_format("pandas")
        df = ready_dedup_dataset[:]
        df.drop_duplicates(subset=["src", "tgt", "ctx"], inplace=True)
        dedup_dataset = Dataset.from_pandas(df, preserve_index=False)
        print(dedup_dataset)

        filtered_dataset = dedup_dataset.filter(filter_unchanged)
        filtered_dataset = filtered_dataset.remove_columns(["src", "tgt", "ctx"])
        print(filtered_dataset)

        filtered_dataset.to_json(
            str(coconut_preprocessed_dir / f"{''.join(lang_dataset.parts[-2:])}.jsonl")
        )


if __name__ == "__main__":
    main()
