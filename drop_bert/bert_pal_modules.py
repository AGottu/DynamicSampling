from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import masked_softmax, weighted_sum
from drop_bert.bert_modules import BertLayer, BertSelfAttention, BertLayerNorm, \
                                                            BertIntermediate, BertOutput, BertConfig

class BertSelfOutputPAL(nn.Module):
    def __init__(self, config, num_task_layers=1):
        super(BertSelfOutputPAL, self).__init__()
        self.num_task_layers = num_task_layers
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pal = torch.nn.ModuleList([BERTPAL(config) for _ in range(num_task_layers)])
        self.pal_task_encoder = torch.nn.Linear(config.hidden_size, 1)
        self.pal_task_selector = torch.nn.Linear(config.hidden_size, num_task_layers)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, hidden_states, input_tensor, attention_mask):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.num_task_layers > 1:
            task_specific = []
            for i in range(len(self.pal)):
                task_specific.append((self.pal[i](hidden_states)).unsqueeze(-1))

            task_specific = torch.cat(task_specific, -1)
            task_logits = self.pal_task_encoder(hidden_states).squeeze(-1)
            task_weights = self.softmax(task_logits + attention_mask.squeeze())
            task_vector = weighted_sum(hidden_states, task_weights)
            task_distribution = self.softmax(self.pal_task_selector(task_vector))
            weighted_task_specfic = task_specific * task_distribution.view(task_distribution.size(0), 1, 1, task_distribution.size(1))
            summed_task_specific = weighted_task_specfic.sum(-1)
        else:
            summed_task_specific = self.pal[0](hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor + summed_task_specific)
        return hidden_states

class BertAttentionPAL(nn.Module):
    def __init__(self, config):
        super(BertAttentionPAL, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutputPAL(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor, attention_mask)
        return attention_output

class BERTPAL(nn.Module):
    def __init__(self, config):
        super(BERTPAL, self).__init__()
        self.pals = nn.Sequential()
        self.pals.add_module("H2L", nn.Linear(config.hidden_size, config.pal_dim_size))
        self.pals.add_module("L2H", nn.Linear(config.pal_dim_size, config.hidden_size))

    def forward(self, hidden_states):
        return self.pals(hidden_states)

class BertLayerPAL(nn.Module):
    def __init__(self, config):
        super(BertLayerPAL, self).__init__()
        self.attention = BertAttentionPAL(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

@Seq2SeqEncoder.register("bert_encoder_pal")
class BERTEncoderPAL(Seq2SeqEncoder):
    def __init__(self, config):
        super(BERTEncoderPAL, self).__init__()
        bert_config = BertConfig.from_dict(config)

        self.layer = []
        for i in range(bert_config.num_hidden_layers):
            if i == bert_config.num_hidden_layers-1:
                layer = BertLayerPAL(bert_config)
            else:
                layer = BertLayer(bert_config)
            self.layer.append(layer)

        self.layer = nn.ModuleList(self.layer)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers