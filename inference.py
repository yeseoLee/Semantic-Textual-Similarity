import argparse
import yaml
import pandas as pd
import os
from tqdm.auto import tqdm

import torch

# import transformers
# import pandas as pd

import pytorch_lightning as pl

# import wandb
##############################
from utils import data_pipeline


if __name__ == "__main__":

    # baseline_config 설정 불러오기
    with open("./config/config.yaml", encoding="utf-8") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # 저장된 폴더 이름
    exp_name = "0913_2020_minseo"

    # dataloader / model 설정
    dataloader = data_pipeline.Dataloader(CFG)
    model = torch.load(f"./experiments/{exp_name}/model.pt")
    # trainer 인스턴스 생성
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=CFG["train"]["max_epoch"],
        log_every_n_steps=1,
    )

    # Inference part
    predictions = trainer.predict(model=model, datamodule=dataloader)
    ## datamodule에서 predict_dataloader 호출

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("./data/raw/sample_submission.csv")
    output["target"] = predictions
    output.to_csv(f"./output/output_({exp_name}).csv", index=False)
