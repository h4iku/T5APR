import shutil

from git import Repo
from tqdm import tqdm

from .configs import cloned_models_dir, models_root

repo_name = "codet5-small-t5apr-multi"
clone_dir = cloned_models_dir / repo_name
checkpoints_dir = models_root / repo_name


def copy_files(copy_dir):
    needed_files = [
        "config.json",
        "special_tokens_map.json",
        "training_args.bin",
        "merges.txt",
        "tokenizer.json",
        "vocab.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
    ]

    copy_dir.mkdir(parents=True, exist_ok=False)
    for file in needed_files:
        shutil.copy(
            clone_dir / file,
            copy_dir,
        )


repo_url = f"https://huggingface.co/h4iku/{repo_name}"
if clone_dir.exists():
    repo = Repo(clone_dir)
else:
    repo = Repo.clone_from(repo_url, clone_dir)

default_branch = repo.heads[0].name
repo.git.checkout(default_branch)
for commit in tqdm(repo.iter_commits()):
    checkpoint_step = commit.message.split()[-1]
    if checkpoint_step.isdigit():
        repo.git.checkout(commit.hexsha)
        copy_dir = checkpoints_dir / f"checkpoint-{checkpoint_step}"
        copy_files(copy_dir)

repo.git.checkout(default_branch)
