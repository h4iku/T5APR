import json
import shutil
from collections import ChainMap
from pathlib import Path

import pandas as pd

from .configs import d4j1_gen_dir, d4j2_gen_dir, d4j_gen_dir


def generate_data(bug_hunks: ChainMap[str, list], gen_dir: Path) -> None:
    """Generates input data for model evaluation"""

    def prepare(hunk: str) -> str:
        lines_concat = " ".join([line.strip() for line in hunk.splitlines()])
        return lines_concat.strip()

    with (
        open(gen_dir / "rem.txt", "w", encoding="utf-8") as remfile,
        open(gen_dir / "add.txt", "w", encoding="utf-8") as addfile,
        open(gen_dir / "context.txt", "w", encoding="utf-8") as ctxfile,
    ):
        for program, hunks in bug_hunks.items():
            remfile.writelines(prepare(h["removed_lines"]) + "\n" for h in hunks)
            addfile.writelines(prepare(h["added_lines"]) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h["source_context"]) + "\n" for h in hunks)


def get_ids(number_of_bugs: int) -> list[int]:
    return list(range(1, number_of_bugs + 1))


d4j1_projects: dict[str, list[int]] = {
    "Chart": get_ids(26),
    "Closure": get_ids(133),
    "Lang": get_ids(65),
    "Math": get_ids(106),
    "Mockito": get_ids(38),
    "Time": get_ids(27),
}
d4j1_bugids = [f"{proj} {num}" for proj, nums in d4j1_projects.items() for num in nums]

beam_size: int = 100

target_dirs: list[str] = ["outputs-java", "outputs-multi"]
target_files: list[str] = [
    f"sequences_{beam_size}.jsonl",
    f"final_candidates_{beam_size}.jsonl",
    f"plausible_candidates_{beam_size}.jsonl",
    f"validated_reranked_candidates_{beam_size}.jsonl",
]

# Move metadata files
meta_filename = "Defects4J.jsonl"
with open(d4j_gen_dir / meta_filename) as d4j_meta_file:
    d4j_meta = [json.loads(line) for line in d4j_meta_file]

d4j1_meta = [b for b in d4j_meta if [*b][0] in d4j1_bugids]
d4j2_meta = [b for b in d4j_meta if [*b][0] not in d4j1_bugids]

assert not any(b for b in d4j1_meta if b in d4j2_meta)

d4j1_gen_dir.mkdir()
d4j2_gen_dir.mkdir()
with (
    open(d4j1_gen_dir / meta_filename, "w", encoding="utf-8") as d4j1_meta_file,
    open(d4j2_gen_dir / meta_filename, "w", encoding="utf-8") as d4j2_meta_file,
):
    print(*[json.dumps(b) for b in d4j1_meta], sep="\n", file=d4j1_meta_file)
    print(*[json.dumps(b) for b in d4j2_meta], sep="\n", file=d4j2_meta_file)

generate_data(ChainMap(*d4j1_meta[::-1]), d4j1_gen_dir)
generate_data(ChainMap(*d4j2_meta[::-1]), d4j2_gen_dir)

# Move other files
for target_dir in target_dirs:
    # Move sate-sate
    d4j_save_state_dir = d4j_gen_dir / target_dir / "save-state"
    d4j_states = list(d4j_save_state_dir.iterdir())

    d4j1_states = [s for s in d4j_states if s.stem in d4j1_bugids]
    d4j2_states = [s for s in d4j_states if s.stem not in d4j1_bugids]
    d4j1_save_state_dir = d4j1_gen_dir / target_dir / "save-state"
    d4j1_save_state_dir.mkdir(parents=True, exist_ok=False)
    d4j2_save_state_dir = d4j2_gen_dir / target_dir / "save-state"
    d4j2_save_state_dir.mkdir(parents=True, exist_ok=False)
    for state in d4j1_states:
        shutil.copy(state, d4j1_save_state_dir)
    for state in d4j2_states:
        shutil.copy(state, d4j2_save_state_dir)

    for target_file in target_files:
        # Read data
        d4j_df = pd.read_json(
            d4j_gen_dir / target_dir / target_file, orient="records", lines=True
        )

        d4j1_df = d4j_df.loc[d4j_df["bugid"].isin(d4j1_bugids)]
        d4j2_df = d4j_df.loc[~d4j_df["bugid"].isin(d4j1_bugids)]

        assert not (set(d4j1_df["bugid"].unique()) & set(d4j2_df["bugid"].unique()))

        # Write data
        d4j1_df.to_json(
            d4j1_gen_dir / target_dir / target_file, orient="records", lines=True
        )
        d4j2_df.to_json(
            d4j2_gen_dir / target_dir / target_file, orient="records", lines=True
        )
