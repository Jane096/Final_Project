# Real or Not? NLP를 통한 자연재해 Twitter 분류 데이터 분석 

## 기획배경
업데이트 된 BERT 기반 NLP 분석을 실험하기 위해 진행한 데이터 분석 프로그램 입니다.

데이터 수집에 대한 애로사항 없이 BERT의 성능 측정에 집중하기 위해서 Kaggle을 활용하였으며, 트위터 내용 중 재해 관련된 트윗을 학습하여
예측하는 Competition을 선정해 수행하였습니다

## 프로젝트 개요

2020/03/09 ~ 03/12 - 기초적인 데이터 분석 과제 학습을 위해 Kaggle의 상시 오픈 과제인 Titanic 예제 스터디

2020/03/13 ~ 03/15 - NLP 과제 수행 위한 workflow 및 필요 기술 정립

2020/03/16 ~ 03/20 - 데이터 전처리 및 일반 모델링과 BERT를 통한 workflow 실현

2020/03/23 ~ 03/30 - 모델별 성능 비교 후 BERT 세부 튜닝 집중(각 과정은 history 엑셀 파일에 저장)

2020/03/31 ~       - 제출 및 Hyper-parameter 업데이트 

## 프로젝트 활용 기술
### Python 3.x 기반

1. 데이터 파악/처리

     - 데이터 탐색(EDA) : numpy, pandas, matplot, seaborn
  
     - 데이터 전처리 : nltk, re, string, gensim
  
2. 데이터 분석/모델링

     - 일반 ML, 일반 신경망 : scikit-learn, keras, sequential
  
     - BERT : tensorflow 2.0, tensorflow-hub
     
## Workflow

처리/탐색 

- 전처리 부분이 작업량의 대부분을 차지하고 중요하기 때문에 4명이서 동시에 진행하는 것이 낫다는 의견을 반영해 데이터셋 확인해서 필터링 해야할 
  부분을 각자 찾아내고 공유할 예정(특수기호,  숫자, 이모티콘, slang, 영역을 나눠서 찾기) 

embedding

- GloVe, TF-IDF 등 여러 word embedding 적용해보기

필터링 

- 일괄 필터링이 가능한 부분은 함수를 정의해서 사용
- 약어 처리 가능한 알고리즘 탐색하기

모델 설계

- 이해도가 높은 조원 2명의 주도 하에 기본형태로 모델 설계할 예정 

- 모델 설계 후 일반 ML과 BERT의 Base-line 정확도 확인하기

- RNN 모델의 경우, word-embedding 방식(GloVe와 TF-IDF) 테스트 결과로 적정 embedding 선정 예정

- BERT의 Hyper-parameter 조정해가며 오버피팅 빠지는 지점과 정확도 높이기 작업 진행 

- 각 팀별로 진행한 경우의 수 history 정리
  
- Fine-Tuning history : BERT_FineTuning.csv (또는 xlsx파일) 참고
  

  
  
  
  
  
  
  
  
  
  



