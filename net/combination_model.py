import os
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from net.cls_pooler.last3pooler import Last3ConcatPooler, Last3MeanPooler, Last3WeightedPooler
from net.cls_pooler.pure_pooler import PurePooler
from net.hidden_pooler.attention_pooler import AttentionPooler
from net.hidden_pooler.lstm_pooler import LSTMPooler
from net.hidden_pooler.lstm_attention_pooler import LSTMAttentionPooler
from net.hidden_pooler.mean_max_pooler import MeanMaxPooler
from net.utils.loss_fct import FocalLoss, CrossEntropyLoss
from torch.nn import BCELoss
from loss_fct import FocalLoss
CLSPooler = {
    'last3concat': Last3ConcatPooler,
    'last3mean': Last3MeanPooler,
    'last3weighted': Last3WeightedPooler,
    'pure': PurePooler,
}
HiddenPooler = {
    'lstm': LSTMPooler,
    'lstm_attention': LSTMAttentionPooler,
    'mean_max': MeanMaxPooler,
    'attention': AttentionPooler,
}

class CombinationModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CombinationModel, self).__init__(config)
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.cls_pooler = self.cls_pooler(config)
        # self.hidden_pooler = self.hidden_pooler(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        all_layer_outputs = outputs[2]
        hidden_output = outputs[0]
        pooled_output = ()

        if self.cls_pooler is not None:
            cls_pooled_output = self.cls_pooler(all_layer_outputs)
            # print(cls_pooled_output.device)
            pooled_output += (cls_pooled_output, )
        if self.hidden_pooler is not None:
            hidden_pooled_output = self.hidden_pooler(hidden_output[:, 1:], attention_mask[:, 1:])
            # print(hidden_pooled_output.device)
            pooled_output += (hidden_pooled_output, )

        if len(pooled_output) > 1:
            pooled_output = torch.cat(pooled_output, 1)
        else:
            pooled_output = pooled_output[0]

        pooled_output = self.pooler(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.num_labels)
        logits = torch.sigmoid(logits)

        loss = self.loss_fct(logits, labels)
        return loss, logits

    @classmethod
    def build_model(cls, args, config, pretrain_path=None):
        cls_pooler = CLSPooler.get(args.cls_pooler, None)
        hidden_pooler = HiddenPooler.get(args.hidden_pooler, None)
        if cls_pooler is None and hidden_pooler is None:
            raise ValueError("You have to initialize one pooler")
        nums = 0
        if cls_pooler:
            cls_pooler = cls_pooler(config)
            nums += 1
        if hidden_pooler:
            hidden_pooler = hidden_pooler(config)
            nums += 1
        hidden_size_changed = config.hidden_size * nums
        pooler = nn.Sequential(
            nn.Linear(hidden_size_changed, config.hidden_size),
            nn.Tanh()
        )
        if args.loss_fct == 'cross_entropy':
            loss_fct = FocalLoss(num_class=config.num_labels, gamma=0)
        elif args.loss_fct == 'focal_loss':
            loss_fct = FocalLoss(num_class=config.num_labels, gamma=1)
        elif args.loss_fct == 'label_smooth':
            loss_fct = FocalLoss(num_class=config.num_labels, smooth=0.05)
        elif args.loss_fct == 'bce_loss':
            loss_fct = BCELoss()
        if pretrain_path:
            model = cls.from_pretrained(pretrain_path, config=config)
        else:
            model = cls(config)
        model.cls_pooler = cls_pooler
        model.hidden_pooler = hidden_pooler
        model.pooler = pooler
        model.loss_fct = loss_fct
        return model
    @classmethod
    def load_model(cls, args, config, pretrain_path):
        model = cls.build_model(args, config)
        model_dir = os.path.join(pretrain_path, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_dir))
        return model






