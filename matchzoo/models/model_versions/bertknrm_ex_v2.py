"""An implementation of Bert Model with knrm output layers with explanable structure 
Add nonlinear activation (tahn) when combining kernal scores
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
        self.bert_q = BertModule(mode=self._params['mode'])
        self.bert_d = BertModule(mode=self._params['mode'])
        self.dropout = nn.Dropout(p=self._params['dropout_rate'])
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
        
        self.weighted_kernel = self._make_perceptron_layer(self._params['kernel_num'], 1, nn.Tanh())
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
        bert_q = self.bert_q(query)[0]
        bert_d =  self.bert_d(doc)[0]
        
        # shape = [B, L, R]
        matching_matrix = torch.einsum('bld,brd->blr',
            F.normalize(bert_q, p=2, dim=-1),
            F.normalize(bert_d, p=2, dim=-1)
        )

        KM = []
        for kernel in self.kernels:
            # shape = [B, L]
            K_q = torch.log1p(kernel(matching_matrix).sum(dim=-1)) #add weight here in the future
            KM.append(K_q)

        # KM shape K of [B, L]
       
        # shape = [B,L,K]
        phi = torch.stack(KM, dim=2)
        
        # shape = [B,L], activation is tanh
        word_score = torch.squeeze(self.weighted_kernel(phi), -1)
        target = torch.zeros(word_score.shape[0], self._params['token_dim'])
        target[:, :word_score.shape[1]] = word_score
        out = self.out(self.dropout(target))
        return out


'''
[Iter-64 Loss-0.694]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.3215 - normalized_discounted_cumulative_gain@5(0.0): 0.3945 - mean_average_precision(0.0): 0.3773
[Iter-128 Loss-0.692]:
Validation: normalized_discounted_cumulative_gain@3(0.0): 0.3903 - normalized_discounted_cumulative_gain@5(0.0): 0.4714 - mean_average_precision(0.0): 0.4366
[Iter-192 Loss-0.678]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.6536 - normalized_discounted_cumulative_gain@5(0.0): 0.6955 - mean_average_precision(0.0): 0.6601
 [Iter-256 Loss-0.597]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.7029 - normalized_discounted_cumulative_gain@5(0.0): 0.7402 - mean_average_precision(0.0): 0.7028
  [Iter-320 Loss-0.486]:
  Validation: normalized_discounted_cumulative_gain@3(0.0): 0.6452 - normalized_discounted_cumulative_gain@5(0.0): 0.6956 - mean_average_precision(0.0): 0.6496

Cost time: 18268.48766684532s
'''