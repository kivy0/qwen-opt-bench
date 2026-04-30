import argparse
import json
from pathlib import Path

import yaml
from transformers import get_constant_schedule_with_warmup

from src.build_optimizer import build_optimizer
from src.config import Config
from src.data import get_dataloader, prepare_dataset
from src.logger import MetricsLogger, setup_logger
from src.model import build_model, build_tokenizer
from src.seed import set_seed
from src.trainer import Trainer


def main(args):
    config = Config(_yaml_file=args.config)

    experiment_path = Path("runs") / config.experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(experiment_path)

    logger.info(f"Experement: {config.experiment_name}")
    logger.info("Config:")
    logger.info(config.model_dump_json(indent=4))

    # save config
    config_path = experiment_path / "config.yaml"
    config_dict = json.loads(config.model_dump_json())
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, indent=4, sort_keys=False)

    set_seed(seed=config.seed, deterministic=config.deterministic)

    model = build_model(**config.model.model_dump())
    tokenizer = build_tokenizer(name_or_path=config.model.name_or_path)

    dataset = prepare_dataset(
        hf_dataset_name=config.dataset.hf_dataset_name,
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
    )

    optimizer = build_optimizer(model, config)
    train_dataloader = get_dataloader(
        dataset,
        tokenizer=tokenizer,
        seed=config.seed,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=config.training.num_warmup_steps
    )

    metrics_logger = MetricsLogger(experiment_path)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        metrics_logger=metrics_logger,
        **config.training.model_dump(exclude=["batch_size", "num_warmup_steps"]),
    )

    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(experiment_path)
    tokenizer.save_pretrained(experiment_path)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Config path.")
    args = parser.parse_args()

    main(args)
