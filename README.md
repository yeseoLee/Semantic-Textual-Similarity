![alt text](banner.png)

# Lv.1 NLP 기초 프로젝트 : 문장 간 유사도 측정(STS)

</div>

## **프로젝트 개요**
> 진행 기간: 24년 9월 10일 ~ 24년 9월 26일

> 데이터셋: 
> - 학습 데이터셋 9,324개
> - 검증 데이터셋 550개
> - 평가 데이터는 1,100개  
>
> 평가 데이터의 50%는 Public 점수 계산에 활용되어 실시간 리더보드에 표기가 되고, 남은 50%는 Private 결과 계산에 활용되었습니다.

부스트캠프AI Tech 7기의 Level1 과정으로 NLP 기초 대회입니다. 주제는 ‘문장 간 유사도 측정’으로, 두 문장이 얼마나 유사한지를 수치화하는 자연어처리 N21 태스크인 의미 유사도 판별(Semantic Text Similarity, 이하 STS)을 진행했습니다. 학습 데이터에 주어진 문장 두 개와 유사도 점수를 기반으로 평가 데이터의 두 문장 간의 유사도를 0과 5 사이의 값으로 예측하는 모델을 구축하였습니다.


## **프로젝트 구조**
```
📦project1
 ┣ 📂config
 ┃ ┗ 📜config.yaml
 ┣ 📂data
 ┣ 📂model
 ┃ ┗ 📜model.py
 ┣ 📂output
 ┣ 📂tb_logs
 ┣ 📂utils
 ┃ ┣ 📂ensemble
 ┃ ┣ 📂preprocess
 ┣ 📜README.md
 ┣ 📜inference.py
 ┣ 📜requirements.txt
 ┗ 📜train.py
```

## **Contributors**

<table align='center'>
  <tr>
    <td align="center">
      <img src="https://github.com/yeseoLee.png" alt="이예서" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/yeseoLee">
        <img src="https://img.shields.io/badge/%EC%9D%B4%EC%98%88%EC%84%9C-grey?style=for-the-badge&logo=github" alt="badge 이예서"/>
      </a>    
    </td>
    <td align="center">
      <img src="https://github.com/Sujinkim-625.png" alt="김수진" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/Sujinkim-625">
        <img src="https://img.shields.io/badge/%EA%B9%80%EC%88%98%EC%A7%84-grey?style=for-the-badge&logo=github" alt="badge 김수진"/>
      </a>    
    </td>
    <td align="center">
      <img src="https://github.com/luckyvickyricky.png" alt="김민서" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/luckyvickyricky">
        <img src="https://img.shields.io/badge/%EA%B9%80%EB%AF%BC%EC%84%9C-grey?style=for-the-badge&logo=github" alt="badge 김민서"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/koreannn.png" alt="홍성재" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/koreannn">
        <img src="https://img.shields.io/badge/%ED%99%8D%EC%84%B1%EC%9E%AC-grey?style=for-the-badge&logo=github" alt="badge 홍성재"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/gayeon7877.png" alt="양가연" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/gayeon7877">
        <img src="https://img.shields.io/badge/%EC%96%91%EA%B0%80%EC%97%B0-grey?style=for-the-badge&logo=github" alt="badge 양가연"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/hsmin9809.png" alt="홍성민" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/hsmin9809">
        <img src="https://img.shields.io/badge/%ED%99%8D%EC%84%B1%EB%AF%BC-grey?style=for-the-badge&logo=github" alt="badge 홍성민"/>
      </a> 
    </td>
  </tr>
</table>

## 역할분담

| 이름   | 역할                                                                                                                                                                                                                                                                  |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 김민서 | 베이스라인 코드 구현, 텐서보드 기능 구현, 허깅페이스 내 모델 Search, 모델링 및 튜닝(`klue/roberta-large`, `klue/roberta-base`, `team-lucid/deberta-v3-base-korean`, `deliciouscat/kf-deberta-base-cross-sts`, `upskyy/kf-deberta-multitask`, `kakaobank/kf-deberta-base`, `klue/bert-base`), 앙상블(`soft voting`, `weighted voting`) |
| 김수진 | Task에 적합한 모델 Search, 데이터 증강(`swap`), 데이터 분할, 모델링 및 튜닝(`snunlp/KR-ELECTRA-discriminator`), 앙상블(`weighted voting`)                                                                                                                        |
| 양가연 | 데이터 전처리(`hanspell`, `soynlp`), 데이터 증강(`copied_sentence`, `swap`, `synonym replacement`, `undersampling`, `masking`), 모델링 및 튜닝(`kykim/electra-kor-base`, `snunlp/KR-ELECTRA-discriminator`, `klue/roberta-large`, `WandB`), 앙상블(`weighted voting`)         |
| 이예서 | EDA(`Label 분포`, `Source 분포`, `Sentence length 분석`), 데이터 전처리(`특수문자 제거`, `초성 대체`, `띄어쓰기/맞춤법 교정`), 데이터 증강(`sentence swap`, `sentence copy`, `korEDA(SR, RI, RS)`, `K-TACC(BERT_RMR, ADVERB)`), 앙상블(`weighted voting`)                       |
| 홍성민 | 모델링 및 튜닝(`kykim/KR-ELECTRA-Base`), 앙상블(`weighted voting`), 베이스라인 코드 수정과 기능 추가                                                                                                                                                           |
| 홍성재 | 하이퍼 파라미터 튜닝(`BS`, `Epoch`, `LR`), 모델 최적화 및 앙상블(`Koelectra-base-v3-discriminator`, `roberta-small`, `bert-base-multilingual-cased` / `Soft voting`)                                                                                     |

## Dependencies
* torch==2.1.0
* transformers==4.35.2
* pytorch-lightning==2.1.2

## Usage
1. Setting
```
$ pip install -r requirements.txt
```
2. Training
```angular2html
$ python3 train.py
```
3. Inference
```angular2html
$ python3 inference.py
```

## 프로젝트 타임라인

<img width="2715" alt="Gantt chart template (Community) (3)" src="https://github.com/user-attachments/assets/3a300753-f0f4-4d86-81ea-df66ed29ad9a">

## 프로젝트 수행결과

<img width="3456" alt="Gantt chart template (Community) (4)" src="https://github.com/user-attachments/assets/02560fce-076e-4b82-b3a7-c35539615da1">

## 리더보드 결과
![image](https://github.com/user-attachments/assets/e666e639-3bfe-4bed-95b1-4fd3a93ed745)

