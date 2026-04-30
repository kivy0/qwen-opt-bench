import argparse
import json
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from src.config import Config
from src.logger import setup_logger
from src.seed import set_seed


def main(args):
    config = Config(_yaml_file=args.config)

    if args.baseline:
        # For base model
        model_name_or_path = config.model.name_or_path
        eval_path = Path("runs") / config.experiment_name / "evaluation_baseline"
    else:
        # For tuned model
        checkpoint_path = (
            Path(args.checkpoint)
            if args.checkpoint is not None
            else Path("runs") / config.experiment_name
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )

    eval_path = checkpoint_path / "evaluation"
    eval_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(eval_path)

    logger.info(f"Experiment: {config.experiment_name}")
    if args.baseline:
        logger.info(f"Running BASELINE evaluation for model: {model_name_or_path}")
    else:
        logger.info(f"Running checkpoint evaluation: {model_name_or_path}")

    logger.info(f"Tasks: {config.evaluation.tasks}")

    set_seed(seed=config.seed, deterministic=config.deterministic)

    lm = HFLM(
        pretrained=model_name_or_path,
        tokenizer=model_name_or_path,
        batch_size=config.evaluation.batch_size,
        device=config.evaluation.device,
        trust_remote_code=True,
    )

    results = simple_evaluate(
        model=lm,
        tasks=config.evaluation.tasks,
        num_fewshot=config.evaluation.num_fewshot,
        limit=config.evaluation.limit,
        log_samples=config.evaluation.log_samples,
    )

    logger.info("Evaluation finished.")
    logger.info("\n" + make_table(results))

    output_path = eval_path / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Config path")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to saved model checkpoint. Default: runs/<experiment_name>",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the baseline model (from config.model.name_or_path) instead of local checkpoint.",
    )
    args = parser.parse_args()

    main(args)
