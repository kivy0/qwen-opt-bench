from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class AdamWConfig(BaseModel):
    type: Literal["adamw"] = "adamw"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool | None = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None


class MuonConfig(BaseModel):
    lr: float = 1e-3
    momentum: float = 0.95
    nesterov: bool = True
    weight_decay: float = 0.1
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    eps: float = 1e-7
    ns_steps: int = 5
    adjust_lr_fn: str | None = None


class MoonlightMuonConfig(BaseModel):
    type: Literal["moonlight_muon"] = "moonlight_muon"
    adamw: AdamWConfig = Field(default_factory=AdamWConfig)
    muon: MuonConfig = Field(default_factory=MuonConfig)


class HybridOptimizerConfig(BaseModel):
    type: Literal["hybrid"] = "hybrid"
    muon_layers_limit: int = 12

    muon: MuonConfig = Field(default_factory=MuonConfig)
    adamw_2d: AdamWConfig = Field(default_factory=AdamWConfig)
    adamw_other: AdamWConfig = Field(default_factory=AdamWConfig)


OptimizerConfig = Annotated[
    AdamWConfig | MoonlightMuonConfig,
    Field(discriminator="type"),
]


class DatasetConfig(BaseModel):
    hf_dataset_name: str = "Elriggs/openwebtext-100k"
    max_length: int = 512


class ModelConfig(BaseModel):
    name_or_path: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True


class TrainingConfig(BaseModel):
    num_steps: int = 1000
    batch_size: int = 8
    grad_accum_steps: int = 4
    num_warmup_steps: int = 500
    device: str = "cpu"


class EvaluationConfig(BaseModel):
    tasks: list[str] = ["hellaswag"]
    batch_size: int | str = 8
    num_fewshot: int = 0
    limit: int | None = None
    device: str = "cuda"
    log_samples: bool = False


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/base.yaml",
        yaml_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    seed: int = 42
    deterministic: bool = True

    experiment_name: str = "baseline"

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = Field(default_factory=AdamWConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        custom_yaml = init_settings.init_kwargs.get("_yaml_file")
        yaml_files = ["configs/base.yaml", *([custom_yaml] if custom_yaml else [])]

        return (YamlConfigSettingsSource(settings_cls, yaml_file=yaml_files),)
