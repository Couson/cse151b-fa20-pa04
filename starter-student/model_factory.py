################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

# Build and return the model here based on the configuration.

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Embedding, LSTM
import torch.nn.functional as F

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    
    if model_type == 'LSTM':
        return cnnLSTM(hidden_size, embedding_size, vocab)
    elif model_type == 'RNN':
        return cnnRNN(hidden_size, embedding_size, vocab)
    else:
        raise ValueError('Invalid Model Name')


class cnnLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(cnnLSTM, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.embed = Embedding(len(vocab), embedding_size)
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers = 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.fc(features.view(features.size(0), -1))
        embeddings = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)

    def sample(self, images, max_len, deter):

        sampled_ids = []
        with torch.no_grad():
            features = self.resnet(images)
        features = self.fc(features.view(features.size(0), -1)).unsqueeze(1)
        hiddens, states = self.lstm(features)
        outputs = self.linear(hiddens.squeeze(1))

        if deter:
            for _ in range(max_len):
                outputs = self.embed(predicted).unsqueeze(1)
                hiddens, states = self.lstm(outputs, states)
                outputs = self.linear(hiddens.squeeze(1))

                predicted = torch.max(outputs, 1)[1]
                sampled_ids.append(predicted.item())

            return torch.stack(sampled_ids, 1)

        else:
            for _ in range(max_len):
                outputs = self.embed(predicted).squeeze(1)
                hiddens, states = self.lstm(outputs, states)
                outputs = self.linear(hiddens.squeeze(1))

                probabilities = F.softmax(outputs.div(self.__generation_config['temperature']).squeeze(0).squeeze(0), dim=1) 
                predicted_id = torch.multinomial(probabilities.data, 1)
                sampled_ids.append(predicted_id)

            return torch.cat(sampled_ids, 1)

        



class cnnRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(cnnRNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.embed = Embedding(len(vocab), embedding_size)
        self.rnn = nn.rnn(input_size=embedding_size, hidden_size=hidden_size, num_layers = 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.fc(features.view(features.size(0), -1))
        embeddings = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.rnn(inputs)
        return self.linear(hiddens)
        