import torch.nn as nn
from transformers import BertForSequenceClassification

class Bert_seq(nn.Module):

    def __init__(self, hidden_size: int, n_label: int, freeze_base: bool = False):
        super(Bert_seq, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained("klue/bert-base")

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad=False


        dropout_rate = 0.1
        linear_layer_hidden_size = 6

        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, linear_layer_hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(linear_layer_hidden_size, n_label)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        last_hidden_states = outputs.hidden_states[-1] # last hidden states (batch_size, sequence_len, hidden_size)
        cls_token_last_hidden_states = last_hidden_states[:,0,:] # (batch_size, first_token, hidden_size)
        logits  = self.classifier(cls_token_last_hidden_states)


        return logits
