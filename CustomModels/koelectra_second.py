import torch.nn as nn
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, ElectraModel

class koelectra_second(nn.Module):

    def __init__(self, hidden_size: int, n_label: int, freeze_base: bool = False):
        super(koelectra_second, self).__init__()

        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")  # KoELECTRA-Base-v3

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

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_states = outputs['last_hidden_state'] # last hidden states (batch_size, sequence_len, hidden_size)
        cls_token_last_hidden_states = last_hidden_states[:,0,:] # (batch_size, first_token, hidden_size)

        logits = self.classifier(cls_token_last_hidden_states)

        return logits
