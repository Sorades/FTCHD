from common.pipeline import Pipeline


class FitPipeline(Pipeline):
    def __call__(self):
        ckpt_path = self.config.get("ckpt_path")

        self.trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
        ckpt_path = getattr(self.trainer.checkpoint_callback, "best_model_path")
        assert ckpt_path is not None, "No best model path found"
        self.wandb_logger.log_hyperparams({"saved_ckpt_path": ckpt_path})

        if self.config.get("test"):
            self.trainer.test(
                self.model,
                datamodule=self.datamodule,
                ckpt_path=ckpt_path,
                verbose=False,
            )


if __name__ == "__main__":
    FitPipeline.start("fit")
