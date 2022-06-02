# import os
# import sys
import re
# import pandas as pd
# import numpy as np 
# import copy
# import time
# import random


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader,  RandomSampler, SequentialSampler, random_split
# from torch.nn.utils import clip_grad_norm_

# from transformers import RobertaTokenizer, RobertaModel
# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
# import transformers
# from transformers import AdamW, get_linear_schedule_with_warmup

# from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
# from transformers import BertTokenizer

# from CustomModel.BERT_baseline import Bert_baseline


def check_st(text):
    s = re.sub(r"[^ㄱ-힣0-9\s]","",text)
    return s


