from flask import Flask, render_template, request
import torch
import os
import sys

_CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _CUR_PATH)

from transformers import ElectraModel, ElectraTokenizer

import numpy as np
# from transformers import BertTokenizer
from pydantic import BaseModel
# from CustomModel.BERT_baseline import Bert_baseline
from test import check_st


app = Flask(__name__)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# load checkpoint file
_CKPT_PATH = os.path.join(_CUR_PATH, "./model") #ㄴㅏ으ㅣ checkpoint 파일 위치 설정
ckpt = torch.load(os.path.join(_CKPT_PATH, "model.ckpt.best_3"), map_location=device)

#model 사전 로드
model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
model.load_state_dict(ckpt["model_state_dict"])

# Tokenzier 사전 로드 (필수)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


class Data(BaseModel):
    sentence : str




@app.route('/') 
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['question1']
    data2 = request.form['question2']

    
    tokens = tokenizer(data1, data2, padding=True, truncation=True, return_tensors="pt")

    #token device 올리기
    tokens.to(device)

    output = model(**tokens)

    pred = float(output[0][0][0].detach().cpu().numpy())

    label = 1 if pred>=3.0 else 0

    message = ""

    if label == 0:
        message = (f'0 : 다른 문장')
    else:
        message = (f'1 : 비슷하거나, 같은 문장')



    return render_template('predict.html', value = message)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0')