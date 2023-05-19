from names import *
from utils import *
import pandas as pf
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.model_selection import train_test_split


def model_run(data, device):

  target_column = DEFECT_INT_F3

  df = drop_columns_if_exist(data, DATE_AND_TIME_F1)
  pd.set_option('display.max_columns', None)  # Show all columns

  # Hacer mejor seleccon de las columnas

  normal_df = df[df[target_column] == 1]
  defect_df = df[df[target_column] != 1]

  print("Normal pieces data:", normal_df.shape)
  print("Defective pieces data:", defect_df.shape)

  train_df, validate = train_test_split(normal_df, test_size=0.2)

  test_df, val_df = train_test_split(validate, test_size=0.5)

  train_sequences = train_df.astype(np.float32).to_numpy().tolist()
  val_sequences = val_df.astype(np.float32).to_numpy().tolist()
  test_sequences = test_df.astype(np.float32).to_numpy().tolist()
  defect_sequences = defect_df.astype(np.float32).to_numpy().tolist()

  print("###################################################################################")

  train_dataset, seq_len, n_features = create_dataset(train_sequences)
  val_dataset, _, _ = create_dataset(val_sequences)
  test_dataset, _, _ = create_dataset(test_sequences)
  defect_dataset, _, _ = create_dataset(defect_sequences)

  model = RecurrentAutoencoder(seq_len, n_features, device)
  model = model.to(device)

  model, history = train_model(
    model,
    train_dataset,
    val_dataset,
    n_epochs=150,
    device=device
  )

  return model


class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, device, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


def create_dataset(df):

  sequences = np.array(df, dtype=np.float32).tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features


def train_model(model, train_dataset, val_dataset, n_epochs, device):

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)

  return model.eval(), history