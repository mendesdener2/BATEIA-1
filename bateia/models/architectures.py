import torch
import torch.nn as nn

# --- Modelo 1: Multi-Layer Perceptron (MLP) ---
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.3):
        super(MLPRegressor, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# --- Modelo 2: Long Short-Term Memory (LSTM) ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_dense_layers, dense_neurons, dropout):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        dense_layers = []
        in_features = hidden_size
        for i in range(n_dense_layers):
            dense_layers.append(nn.Linear(in_features, dense_neurons))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            in_features = dense_neurons
        dense_layers.append(nn.Linear(in_features, 1))
        self.regressor = nn.Sequential(*dense_layers)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.regressor(last_time_step_out)
        return prediction

# --- Modelo 3: Gated Recurrent Unit (GRU) ---
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_dense_layers, dense_neurons, dropout):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        dense_layers = []
        in_features = hidden_size
        for i in range(n_dense_layers):
            dense_layers.append(nn.Linear(in_features, dense_neurons))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            in_features = dense_neurons
        dense_layers.append(nn.Linear(in_features, 1))
        self.regressor = nn.Sequential(*dense_layers)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_time_step_out = gru_out[:, -1, :]
        prediction = self.regressor(last_time_step_out)
        return prediction

# --- Modelo 4: 1D Convolutional Neural Network (1D-CNN) ---
class CNN1DRegressor(nn.Module):
    def __init__(self, input_channels, seq_length, n_conv_layers, out_channels, kernel_size, n_dense_layers, dense_neurons, dropout):
        super(CNN1DRegressor, self).__init__()
        
        conv_layers = []
        in_channels = input_channels
        for i in range(n_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv_part = nn.Sequential(*conv_layers)
        
        dense_layers = [nn.Flatten()]
        in_features = out_channels
        for i in range(n_dense_layers):
            dense_layers.append(nn.Linear(in_features, dense_neurons))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            in_features = dense_neurons
        dense_layers.append(nn.Linear(in_features, 1))
        self.dense_part = nn.Sequential(*dense_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv_part(x)
        x = self.dense_part(x)
        return x
