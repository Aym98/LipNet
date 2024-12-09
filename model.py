import numpy as np
import cv2 as cv
import imageio
import matplotlib.pyplot as plt
import gdown
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import rgb_to_grayscale
from torchnlp.encoders import LabelEncoder 
from torcheval.metrics.functional import word_error_rate
from data_preprocessing import LipDataset, collate_fn, ctc_decode
from CONFIG import EPOCHS, BATCH_SIZE, device, path
print("packages imported successfully!")



class LipNet(nn.Module):

    def __init__(self):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= 1, out_channels= 32, kernel_size= (3, 5, 5), bias=False)
        self.pool1 = nn.MaxPool3d(kernel_size= (1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels= 32, out_channels= 64, kernel_size= (3, 5, 5), bias=False)
        self.pool2 = nn.MaxPool3d(kernel_size= (1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels= 64, out_channels= 96, kernel_size= (3, 3, 3),  bias=False)
        self.pool3 = nn.MaxPool3d(kernel_size= (1, 2, 2), stride=(1, 2, 2))
        self.gru1 = nn.GRU(input_size = 96 * 3 * 10,hidden_size =256, bidirectional=True, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(512)
        self.gru2 = nn.GRU(input_size = 512,hidden_size =256, bidirectional=True, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(512)
        self.dense = nn.Linear(in_features=512, out_features=41)

    def forward(self, x):
        #Shape of x is B, D, H, W, C
        #if torch.isnan(x).any():
            #print("Input contains nan")
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv1(x)
        #if torch.isnan(x).any():
            #print("After conv1 contains nan")
        x = F.relu(x)
        #if torch.isnan(x).any():
            #print("After relu1 contains nan")
        x = self.pool1(x)
        
        x = F.relu(x)
        
      
        x = F.relu(self.conv2(x))
     
        x = F.relu(self.pool2(x))
        
       
        x = self.conv3(x)
      
        x = F.relu(x)
       
        x = F.relu(self.pool3(x))
        

        b, c, d, h, w = x.size() # Batch, Channels, D, H, W
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(b, d, -1) # X shape is (B, D, Channels * H * W)

        x, hidden = self.gru1(x)
    
        x = self.layer_norm1(x)
        
        x = F.relu(x)
    
        x, hidden = self.gru2(x)
      
        x = self.layer_norm2(x)
   
        x = F.relu(x)
   


        x = self.dense(x) 
        return x
    

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= 1, out_channels= 32, kernel_size= (1, 5, 5), bias=False)
        self.pool1 = nn.MaxPool3d(kernel_size= (1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels= 32, out_channels= 64, kernel_size= (1, 5, 5), bias=False)
        self.pool2 = nn.MaxPool3d(kernel_size= (1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels= 64, out_channels= 96, kernel_size= (1, 3, 3),  bias=False)
        self.pool3 = nn.MaxPool3d(kernel_size= (1, 2, 2), stride=(1, 2, 2))

        self.flatten = nn.Flatten(start_dim=2)

        self.attention_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2880, 
                nhead=8, 
                dim_feedforward=512,
                batch_first=True,
                dropout= 0.25
            ), 
            num_layers=2
        )

        self.dense = nn.Linear(in_features=2880, out_features=41)


    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        
        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)  # [1, 96, 75, 3, 10]
        
        # Flatten and permute for attention layers
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.flatten(x)  # Shape: (batch_size, seq_len, feature_dim)
        # Shape: (batch_size, seq_len, feature_dim)
        

        # Attention layer 1
        x = self.attention_layers(x)
      

        # Final dense layer
        x = self.dense(x)

        return x 


def train_for_one_epoch(model, optimizer, loss_fn, training_loader):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(training_loader):
        frames, labels = data
        frames, labels = frames.to(device), labels.to(device)

        tokens = model(frames)


        B, target_D = labels.size()
        B, D, C = tokens.size()
        tokens = tokens.permute(1, 0, 2) # shape D, B, C
        tokens = F.log_softmax(tokens, dim = 2)

        input_lengths = torch.full((B,), D, dtype=torch.long)
        target_lengths = torch.full((B,), target_D, dtype=torch.long)
        assert(f"input lenghts is greater than the output length {input_lengths > target_lengths}")
        loss = loss_fn(tokens, labels, input_lengths, target_lengths)
        epoch_loss += loss.item()

        optimizer.zero_grad()  # Clear accumulated gradients
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()  # Update model weights
        
    return epoch_loss / (i + 1)

def predict(model, x) : 
     return torch.argmax(model(x), axis = 2)

def validate_model(model, loss_fn, val_loader):
    model.eval()
    running_vloss = 0
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
                vframes, vlabels = vdata
                vframes, vlabels = vframes.to(device), vlabels.to(device)
                vout = model(vframes)

                B, target_D = vlabels.size()
                B, D, C = vout.size()
                vout = vout.permute(1, 0, 2) # shape D, B, C
                vout = F.log_softmax(vout, dim = 2)
                vinput_lengths = torch.full((B,), D, dtype=torch.long)
                vtarget_lengths = torch.full((B,), target_D, dtype=torch.long)

                vloss = loss_fn(vout, vlabels, vinput_lengths, vtarget_lengths).item()
                running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    return avg_vloss, vout 

def train(model, optimizer, loss_fn, n_epochs, scheduler, training_loader, val_loader):
     history = {"epochs" : [], "traing_loss": [], "val_loss": []}
     best_models = {"epoch" : [], "loss" : [], "predictions" : []}
     best_v_loss = 1000
     for epoch in range(n_epochs):
        avg_loss = train_for_one_epoch(model, optimizer, loss_fn, training_loader)
        avg_vloss, last_pred = validate_model(model, loss_fn, val_loader)
        scheduler.step(avg_vloss)

        history["epochs"].append(epoch +1)
        history["traing_loss"].append(avg_loss)
        history["val_loss"].append(avg_vloss)

        print(f"EPOCH {epoch+1} : training loss : {avg_loss}, validation loss : {avg_vloss}")

        if avg_vloss < best_v_loss :
            best_v_loss = avg_vloss
            best_models["epoch"].append(epoch + 1)
            best_models["loss"].append(avg_vloss)
            best_models["predictions"].append(ctc_decode(torch.argmax(last_pred, axis = 2)))

        if (epoch + 1) % 25 ==0:
             torch.save(model, f"LipAttention_EPOCH_{epoch+1}.pt")
               

     return history, best_models


def plot_loss(history):
    fig, axs = plt.subplots(1, 2, sharex = True)
    x_axis = history["epochs"]
    axs[0].plot(x_axis, history["train"])
    axs[0].set_title(f'train loss per epoch')
    axs[1].plot(x_axis, history["val"])
    axs[1].set_title(f'validation loss per epoch')
    plt.show()


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

"""if __name__ == "__main__":

    training_data = LipDataset(path, "train")
    training_loader = DataLoader(training_data, batch_size=BATCH_SIZE, collate_fn = collate_fn)
    val_data = LipDataset(path, split = "val")
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn = collate_fn)
    model = LipNet()
    model.apply(initialize_weights)
    loss_fn = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2)
    history, best_models = train(model, optimizer, loss_fn, EPOCHS, scheduler)
    plot_loss(history)"""
