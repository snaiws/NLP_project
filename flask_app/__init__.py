from flask import Flask, render_template, request
import torch
import os
import sys

_CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _CUR_PATH)


import numpy as np
from transformers import BertTokenizer
from pydantic import BaseModel
from CustomModel.BERT_baseline import Bert_baseline
from test import check_st


app = Flask(__name__)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# load checkpoint file
_CKPT_PATH = os.path.join(_CUR_PATH, "./model") #ㄴㅏ으ㅣ checkpoint 파일 위치 설정
ckpt = torch.load(os.path.join(_CKPT_PATH, "model.ckpt.best"), map_location=device)

#model 사전 로드
model = Bert_baseline(hidden_size=768, n_label=6)
model.load_state_dict(ckpt["model_state_dict"])

# Tokenzier 사전 로드 (필수)
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")


class Data(BaseModel):
    sentence : str




@app.route('/') 
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['question1']
    data2 = request.form['question2']

    
    s1=check_st(data1)
    s2=check_st(data2)
    text = s1 + ' [SEP] ' + s2
    

    tensorized_input = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True, # max_length를 넘는 문장은 이 후 토큰을 제거함
            max_length=512,
            return_tensors='pt' # 토크나이즈된 결과 값을 텐서 형태로 반환
        )

    with torch.no_grad():
        logits = model(**tensorized_input)

    label = np.argmax(logits, axis=-1)

    message = ""

    if label < 4:
        message = (f'0 : 다른 문장')
    else:
        message = (f'1 : 비슷하거나, 같은 문장')



    return render_template('predict.html', value = message)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0')