from configs import cache_dir, models_root
from huggingface_hub import HfApi

repo_name = "codet5-small-t5apr-multi"
repo_id = f"h4iku/{repo_name}"
checkpoints_dir = models_root / repo_name

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

api = HfApi()
commits = api.list_repo_commits(repo_id)

for commit in commits:
    checkpoint_step = commit.title.split()[-1]
    if checkpoint_step.isdigit():
        download_dir = checkpoints_dir / f"checkpoint-{checkpoint_step}"
        api.snapshot_download(
            repo_id=repo_id,
            revision=commit.commit_id,
            cache_dir=cache_dir,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            allow_patterns=needed_files,
        )
        break
