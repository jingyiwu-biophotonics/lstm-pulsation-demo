"""
lstm_detection_model.py

Models for pulsation tracing using temporal convolutional blocks and 
a bidirectional LSTM. Designed for use in the pulsation LSTM demo repo.

Classes
-------
TemporalConvResidualBlock : nn.Module
    Residual block with non-causal temporal convolutions.
DetectionModel : nn.Module
    Base detection model combining temporal convolution with residual block.
LSTMDetectionModel : DetectionModel
    Full detection model with bidirectional LSTM and dense output layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvResidualBlock(nn.Module):
    """
    Non-causal temporal convolutional residual block.

    Parameters
    ----------
    n_hid : int
        Number of hidden channels (feature dimension).
    kernel_size : int, optional
        Size of the convolution kernel. Default = 3.
    stride : int, optional
        Stride of the convolution. Default = 1.
    dilation : int, optional
        Dilation factor for the convolution. Default = 1.
    dropout : float, optional
        Dropout probability. Default = 0.0.
    """
    def __init__(self, n_hid, kernel_size=3, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        self.tcn1 = nn.Conv1d(
            n_hid, n_hid, kernel_size=kernel_size,
            stride=stride, dilation=dilation,
            padding=((kernel_size - 1) * dilation) // 2
        )
        self.tcn2 = nn.Conv1d(
            n_hid, n_hid, kernel_size=3,
            dilation=dilation, padding=dilation
        )
        self.dropout = dropout

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, channels].

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input.
        """
        out = F.dropout(
            F.relu(self.tcn1(x.transpose(1, 2))),
            training=self.training, p=self.dropout
        )
        out = F.dropout(
            F.relu(self.tcn2(out)),
            training=self.training, p=self.dropout
        ).transpose(1, 2)
        return out + x  # residual connection


class DetectionModel(nn.Module):
    """
    Base detection model with temporal convolution and residual block.

    Parameters
    ----------
    n_hid : int
        Number of hidden channels.
    n_layers : int
        Number of layers (currently used in subclasses).
    stride : int, optional
        Stride for convolution. Default = 1.
    """
    def __init__(self, n_hid, n_layers, stride=1):
        super().__init__()
        self.tcn1 = nn.Conv1d(1, n_hid, kernel_size=7, padding=3)
        self.tcn2 = TemporalConvResidualBlock(
            n_hid, kernel_size=3, stride=stride, dropout=0.0
        )
        self.aux = nn.Linear(n_hid, 1)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, channels], usually channels=1.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, n_hid].
        """
        x = F.relu(self.tcn1(x.transpose(1, 2))).transpose(1, 2)
        return self.tcn2(x)


class LSTMDetectionModel(DetectionModel):
    """
    Bidirectional LSTM-based detection model.

    Extends DetectionModel by applying a bidirectional LSTM and 
    dense fully-connected layers.

    Parameters
    ----------
    n_hid : int
        Number of hidden channels.
    n_layers : int
        Number of LSTM layers.
    """
    def __init__(self, n_hid, n_layers):
        super().__init__(n_hid, n_layers)
        self.lstm = nn.LSTM(
            n_hid, n_hid, num_layers=n_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(n_hid * 2, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, channels].

        Returns
        -------
        tuple of torch.Tensor
            (1) Main output: [batch_size, seq_len, 1]
            (2) Auxiliary output: [batch_size, seq_len, 1]
        """
        x = super().forward(x)
        out, _ = self.lstm(x)
        return self.fc(out), self.aux(x)
