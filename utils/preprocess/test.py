import yaml
import torch
import os

import pytorch_lightning as pl
from utils import data_pipeline, utils
from model.model import Model

if __name__ == "__main__":

    # baseline_config 설정 불러오기
    with open("./config/config.yaml", encoding="utf-8") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # dataloader 설정 (test data만 사용)
    dataloader = data_pipeline.Dataloader(CFG)

    # experiments 폴더 내부 실험 폴더
    exp_name = CFG["inference"]["exp_name"]

    # 저장된 모델 불러오기
    model_path = f"./experiments/{exp_name}/model.pt"

    if os.path.exists(model_path):
        model = torch.load(model_path)
        print(f"모델이 불러와졌습니다: {model_path}")
    else:
        raise FileNotFoundError(f"{model_path} 파일을 찾을 수 없습니다.")

    # trainer 인스턴스 생성
    trainer = pl.Trainer(accelerator="gpu", devices=1)

    # Test part (metrics)
    trainer.test(model=model, datamodule=dataloader)
    ## datamodule에서 test_dataloader 호출
    ## predict_path로 설정된 test.csv가 사용된다
