"""An implementation of Bert Model with knrm output layers with explanable structure 
v6
- same as v5, but using shared bert for query and document
- change kernel weight activation to relu
"""
import typing

import torch
import torch.nn as nn
from pytorch_transformers import BertModel
import torch.nn.functional as F

from matchzoo import preprocessors
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks
from matchzoo.modules import BertModule
from matchzoo.modules import GaussianKernel




class BertKNRMex(BaseModel):
    """
    Bert Model with knrm output layers and explanable output structure 

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='mode', value='bert-base-uncased',
                         desc="Pretrained Bert model."))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))       
        params.add(Param(
            name='kernel_num',
            value=11,
            hyper_space=hyper_spaces.quniform(low=5, high=20),
            desc="The number of RBF kernels."
        ))
        params.add(Param(
            name='sigma',
            value=0.1,
            hyper_space=hyper_spaces.quniform(
                low=0.01, high=0.2, q=0.01),
            desc="The `sigma` defines the kernel width."
        ))
        params.add(Param(
            name='exact_sigma', value=0.001,
            desc="The `exact_sigma` denotes the `sigma` "
                 "for exact match."
        ))
        
        params.add(Param(
            name='token_dim', value=512,
            desc="The maximum number of tokens for BERT."
        ))
        
        return params
    @classmethod
    def get_default_preprocessor(
        cls,
        mode: str = 'bert-base-uncased'
    ) -> BasePreprocessor:
        """:return: Default preprocessor."""
        return preprocessors.BertPreprocessor(mode=mode)

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre'
    ):
        """:return: Default padding callback."""
        return callbacks.BertPaddingSingle(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
            pad_value=pad_value,
            pad_mode=pad_mode)
    
    def build(self):
        """Build model structure."""
        self.bert = BertModule(mode=self._params['mode'])
        self.dropout = nn.Dropout(p=self._params['dropout_rate'])
        self.q_w = nn.Parameter(torch.tensor(1.1, requires_grad=True))
        self.q_b = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        
        if 'base' in self._params['mode']:
            dim = 768
        elif 'large' in self._params['mode']:
            dim = 1024
        self.kernels = nn.ModuleList()
        for i in range(self._params['kernel_num']):
            mu = 1. / (self._params['kernel_num'] - 1) + (2. * i) / (self._params['kernel_num'] - 1) - 1.0
            sigma = 0.1
            if mu > 1.0:
                sigma = 0.01
                mu = 1.0
            self.kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        
        self.weighted_kernel = self._make_perceptron_layer(self._params['kernel_num'], 1, nn.ReLU())
        self.out = self._make_output_layer(self._params['token_dim'])

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   K = number of kernels

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        
        query, doc = inputs['text_left'], inputs['text_right']
        bert_q = self.bert(query)[0]
        bert_d =  self.bert(doc)[0]
        
        # shape = [B, (L+1), (R+1)]
        matching_matrix = torch.einsum('bld,brd->blr',
            F.normalize(bert_q, p=2, dim=-1),
            F.normalize(bert_d, p=2, dim=-1)
        )
        # sim(query token, query CLS) dim = [B,L]
        query_token_d_weight = nn.ReLU()(self.q_b + self.q_w * torch.squeeze(torch.einsum('bid,bld->bil',
            F.normalize(bert_q[:,0:1], p=2, dim=-1),
            F.normalize(bert_q, p=2, dim=-1)
        ), 1))[:,1:]
        
        
        # sim(doc token, doc CLS) dim = [B,R]
        doc_token_d_weight = nn.ReLU()(self.q_b + self.q_w * torch.squeeze(torch.einsum('bid,brd->bir',
            F.normalize(bert_d[:,0:1], p=2, dim=-1),
            F.normalize(bert_d, p=2, dim=-1)
        ), 1))[:,1:]
        
        KM = []
        
        for kernel in self.kernels:
            # shape = [B, L]
            K_q = torch.log1p(torch.einsum('blr,br -> bl', 
                      kernel(matching_matrix[:,1:,1:]),
                      doc_token_d_weight)) #add weight here in the future
            KM.append(K_q)

        # KM shape K of [B, L]
       
        # shape = [B,L,K]
        phi = torch.stack(KM, dim=2)
        
        # shape = [B,L], activation is tanh
        word_score = torch.squeeze(self.weighted_kernel(phi), -1)
        #query token weight = sim(query_token, doc_CLS), dim = [B, L]
        
        weighted_word_score = query_token_d_weight * word_score
        target = torch.zeros(word_score.shape[0], self._params['token_dim'])
        target[:, :word_score.shape[1]] = word_score
        out = self.out(target)
        return out

'''
Epoch 1/5: 100%
64/64 [55:32<00:00, 52.08s/it, loss=0.694]
[Iter-128 Loss-0.692]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.3204 - normalized_discounted_cumulative_gain@5(0.0): 0.3979 - mean_average_precision(0.0): 0.3669

Epoch 2/5: 100%
64/64 [44:55<00:00, 42.12s/it, loss=0.687]
[Iter-192 Loss-0.692]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.3468 - normalized_discounted_cumulative_gain@5(0.0): 0.4181 - mean_average_precision(0.0): 0.3826

Epoch 3/5: 100%
64/64 [42:53<00:00, 40.22s/it, loss=0.678]
[Iter-256 Loss-0.686]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.509 - normalized_discounted_cumulative_gain@5(0.0): 0.569 - mean_average_precision(0.0): 0.5099

Epoch 4/5: 100%
64/64 [43:12<00:00, 40.51s/it, loss=0.690]
[Iter-320 Loss-0.681]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.5554 - normalized_discounted_cumulative_gain@5(0.0): 0.6279 - mean_average_precision(0.0): 0.5665

Epoch 5/5: 100%
64/64 [42:35<00:00, 39.93s/it, loss=0.683]
[Iter-384 Loss-0.671]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.6346 - normalized_discounted_cumulative_gain@5(0.0): 0.6973 - mean_average_precision(0.0): 0.6465

Cost time: 13750.721867084503s
'''




