import os
from datetime import datetime


def get_experiment_folder_name(CFG):
    # 현재 시간 기록
    current_time = datetime.now().strftime("%m%d_%H%M")

    # CFG 값을 가져와서 폴더 이름에 추가
    user_name = CFG["user_name"]
    model_name = CFG["model"]["model_name"]
    lr = CFG["train"]["learning_rate"]
    batch_size = CFG["train"]["batch_size"]

    # 폴더 이름 생성
    experiment_folder_name = (
        f"{current_time}_{model_name}_lr{lr}_batch{batch_size}({user_name})"
    )

    return experiment_folder_name


def create_experiment_folder(CFG, base_path="./experiments"):

    # 월일_시간분_user_name 형식으로 폴더 이름 생성
    experiment_folder_name = get_experiment_folder_name(CFG)

    # experiments 경로에 해당 폴더 생성
    experiment_path = os.path.join(base_path, experiment_folder_name)
    os.makedirs(experiment_path, exist_ok=True)

    return experiment_path
