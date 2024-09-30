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


def get_latest_experiment_folder(base_path="./experiments"):
    # base_path 내의 폴더 리스트 가져오기
    experiment_folders = [
        f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))
    ]

    # 폴더가 없을 경우 None 반환
    if not experiment_folders:
        return None

    # 폴더 생성 시간 기준으로 정렬 (가장 최근 폴더가 마지막에 위치)
    experiment_folders.sort(
        key=lambda x: os.path.getmtime(os.path.join(base_path, x)), reverse=True
    )

    # 가장 최근에 생성된 폴더 반환
    return experiment_folders[0]


if __name__ == "__main__":

    # baseline_config 설정 불러오기
    with open("./config/config.yaml", encoding="utf-8") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # 저장된 폴더 이름 가장 최근걸로 불러오기
    exp_name = get_latest_experiment_folder()

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
