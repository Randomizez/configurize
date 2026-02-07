from configurize import Config, Ref


class MyLoggerConfig(Config):
    log_dir: str
    backend = "loguru"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = "./"

    def build_logger(self):
        from loguru import logger

        logger.add(self.log_dir)
        return logger


class ModelConfig(Config):
    in_channels: int
    out_channels: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = 32
        self.out_channels = 64

    def build_model(self):
        from torch.nn import Linear

        return Linear(self.in_channels, self.out_channels)


class Trainer:
    def __init__(self, cfg: "Exp"):
        self.cfg = cfg

    def train(self):
        import torch
        from tqdm import tqdm

        logger = self.cfg.logger_cfg.build_logger()
        model = self.cfg.model_cfg.build_model()

        for i in tqdm(range(self.cfg.trainer_cfg.train_iters)):
            data = torch.ones((10, 32))
            loss = model(data).sum()
            logger.info(f"loss={loss}")


class TrainerConfig(Config):
    train_iters: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_iters = 10

    def get_trainer_cls(self):
        return Trainer


class Exp(Config):
    logger_cfg = MyLoggerConfig
    model_cfg = ModelConfig
    trainer_cfg = TrainerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger_cfg.log_dir = "./log_dir/"

    def run(self):
        TrainerCls = self.trainer_cfg.get_trainer_cls()
        trainer = TrainerCls(cfg=self)
        trainer.train()


if __name__ == "__main__":
    Exp().run()
