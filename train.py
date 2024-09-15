import argparse
import yaml
import os

import torch

# import transformers
# import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# import wandb
##############################
from utils import data_pipeline, utils
from model.model import Model

##############################


if __name__ == "__main__":

    # baseline_config 설정 불러오기
    with open("./config/config.yaml", encoding="utf-8") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # experiments 폴더 내부에 실험 폴더 생성
    # 폴더 이름 : 실험 날짜 - 실험 시간 - user
    experiment_path = utils.create_experiment_folder(CFG)

    # dataloader / model 설정
    dataloader = data_pipeline.Dataloader(CFG)
    model = Model(CFG)

    # 텐서보드 테스트
    logger = TensorBoardLogger("tb_logs", name="test1")

    # trainer 인스턴스 생성
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=CFG["train"]["max_epoch"],
        log_every_n_steps=1,
        logger=logger,
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    ## datamodule에서 train_dataloader와 val_dataloader를 호출

    ## Dataloader 내부에 val_dataloader 부분을 수정해서
    ## valid set을 바꿀 수 있음

    trainer.test(model=model, datamodule=dataloader)
    ## datamodule에서 test_dataloader 호출
    ## predict_path로 설정된 test.csv가 사용된다

    # 학습된 모델 저장 (experiment_folder 안에 model.pt로 저장)
    torch.save(model, os.path.join(experiment_path, "model.pt"))
    print(f"모델이 저장되었습니다: {experiment_path}")
