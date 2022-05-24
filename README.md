# NLP_project
과제: 한국어 문장의 유사도 분석 모델 훈련 및 서비스화
의미적 텍스트 유사도(Semantic Textual Similarity)는 두 문장 사이의 의미적 동등성의 정도를 측정하는 것입니다. 이 과제에서는 AIRBNB (colloquial review), policy(formal news), and paraKQC(smart home queries)에서 추출된 문장 데이터를 사용하여, 의미적 텍스트 유사도 모델을 훈련하는 것을 목표로 합니다.   
과제의 데이터는 KLUE-STS를 사용합니다.   
1. Train과 Dev 데이터가 공개되어 있습니다. Train set만을 사용하여 모델을 훈련해주세요.
2. Train set을 다시 training data와 validation data로 나누어서 훈련시키는 것을 권장합니다.
3. 모델은 어떤 NLP 모델을 사용해도 됩니다. BERT 등 주어진 학습 환경에서 훈련 가능한 모델을 활용해주
세요.
4. 공개된 Pretrained 모델을 사용하여도 됩니다. (출처 명시)

---
## data
### train data
#### KLUE
1. KLUE(https://klue-benchmark.com/team)의 KLUE-STS data(https://kluebenchmark.
com/tasks/67/overview/description)를 사용합니다. 이 데이터 셋은 AIRBNB (colloquial review), policy(formal news), and paraKQC(smart home queries)에서 추출되었습니다.
2. 데이터는 Train, Dev 각 11668, 519개의 문장 Pair로 이루어져있습니다.
3. Train set만을 사용하여 훈련해주세요.
4. 각 Pair 데이터 내의 sentence1, sentence2, labels를 사용해 훈련해주세요. 다른 값은 훈련 시에 사용하면 안 됩니다.
### test data

---
## content

---
## author

---
## license
