import random
import numpy as np
import pandas as pd
import torch
from typing import List
from itertools import combinations


##
def seed_everything(SEED=42):
    """
    시드 고정
    """
    deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def postprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    후처리 함수 : 정규화
    """
    # target 컬럼의 최소값과 최대값
    min_val = df["target"].min()
    max_val = df["target"].max()

    # 정규화: 0.0~1.0 범위로 변환
    df["target"] = (df["target"] - min_val) / (max_val - min_val)
    # 다시 0.0~5.0 범위로 변환
    df["target"] = df["target"] * 5.0
    return df

    # 예측값에 단순 -0.1 뺄셈
    # df["target"] = df["target"].apply(lambda x: max(0, x - 0.1))
    # return df


def ensemble(
    result_path_list: List[pd.DataFrame],
    score_list: List[float],
    postprocessing_list: List[bool],
    save_path="./ensemble_results/ensemble.csv",
) -> pd.DataFrame:
    """
    점수 가중 평균 Ensemble
    """
    df_submission, weight_sum = None, 0
    for i, (path, weight, pp) in enumerate(
        zip(result_path_list, score_list, postprocessing_list)
    ):
        df_now = pd.read_csv(path)
        # 후처리를 진행
        if pp:
            df_now = postprocessing(df_now)

        # i == 0에서 제출 파일 생성 / 점수 가중 합
        if i == 0:
            df_submission = pd.read_csv(path)
            df_submission["target"] = weight * df_now["target"]
        else:
            df_submission["target"] += weight * df_now["target"]

        weight_sum += weight

    # 점수 가중 평균
    df_submission["target"] /= weight_sum
    df_submission["target"] = df_submission["target"]
    df_submission.to_csv(save_path, index=False)


def calculate_entropy(data):
    value_counts = data.value_counts(normalize=True)
    entropy = -np.sum(
        value_counts * np.log2(value_counts + 1e-9)
    )  # 작은 값 추가로 로그 계산시 0 방지
    return entropy


def calculate_gini(data):
    sorted_data = np.sort(data)  # 데이터를 정렬
    n = len(data)

    # 지니 계수 계산
    cumulative_values = np.cumsum(sorted_data)  # 누적 합계
    gini = (2 * np.sum(cumulative_values) / cumulative_values[-1] - (n + 1)) / n

    # 지니 계수는 양수여야 하므로 절대값을 취해 0에서 1 사이로 맞춤
    return np.abs(gini)


def ensemble_with_metrics(
    result_path_list: List[str],
    score_lists: List[List[float]],
    postprocessing_list: List[bool],
):
    """
    주어진 여러 점수 조합에 대해 앙상블을 수행하고, 그 결과의 지니계수와 엔트로피를 계산합니다.
    """
    for score_list in score_lists:
        save_path = f"./ensemble_results/ensemble_{'_'.join(map(str, score_list))}.csv"
        print(f"\n=== 앙상블 결과 (score_list: {score_list}) ===")

        # 앙상블 진행
        df_submission = ensemble(
            result_path_list, score_list, postprocessing_list, save_path=save_path
        )

        # 지니 계수 및 엔트로피 계산
        entropy = calculate_entropy(df_submission["target"])
        gini = calculate_gini(df_submission["target"])

        # 결과 출력
        print(f"Entropy: {entropy:.4f}")
        print(f"Gini Coefficient: {gini:.4f}")


def ensemble_with_combinations(
    result_path_list: List[str],
    score_lists: List[List[float]],
    postprocessing_list: List[bool],
    n: int,
):
    """
    주어진 경로 목록에서 n개 조합을 생성하여 앙상블을 진행한 후, 엔트로피 및 지니계수를 출력합니다.
    """
    if len(result_path_list) < n:
        print("Error: 경로 목록의 개수가 n보다 적습니다.")
        return

    # n개의 조합을 생성
    path_combinations = list(combinations(result_path_list, n))
    score_combinations = list(combinations(score_lists, n))

    for idx, (path_combination, score_combination) in enumerate(
        zip(path_combinations, score_combinations)
    ):
        print(f"\n=== 조합 {idx + 1}: 선택된 파일들 ===")
        for path in path_combination:
            print(f"- {path}")

        save_path = f"./ensemble_results/ensemble_combination_{idx + 1}.csv"

        # 선택된 파일 경로와 점수 조합으로 앙상블 수행
        df_submission = ensemble(
            path_combination, score_combination, postprocessing_list, save_path
        )

        # 지니 계수 및 엔트로피 계산
        entropy = calculate_entropy(df_submission["target"])
        gini = calculate_gini(df_submission["target"])

        # 평균 점수 계산
        avg_score = np.mean([np.mean(scores) for scores in score_combination])

        # 결과 출력
        print(f"Average Score: {avg_score:.4f}")
        print(f"Entropy: {entropy:.4f}")
        print(f"Gini Coefficient: {gini:.4f}")
