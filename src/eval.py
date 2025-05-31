from common.pipeline import Pipeline


class EvalPipeline(Pipeline):
    def __call__(self):
        ckpt_path = self.config.get("ckpt_path")
        if ckpt_path is None:
            raise RuntimeError("No `ckpt_path` found in config, unable to eval")
        self.trainer.test(
            self.model, datamodule=self.datamodule, ckpt_path=ckpt_path, verbose=False
        )


if __name__ == "__main__":
    EvalPipeline.start("eval")
