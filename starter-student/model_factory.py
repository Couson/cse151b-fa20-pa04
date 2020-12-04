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
    
    if model_type == 'LSTM1':
        return cnnLSTM1(hidden_size, embedding_size, vocab)
    elif model_type == 'LSTM2':
        return cnnLSTM2(hidden_size, embedding_size, vocab)
    elif model_type == 'RNN':
        return cnnRNN(hidden_size, embedding_size, vocab)
    else:
        raise ValueError('Invalid Model Name')
    
class cnnLSTM1(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(cnnLSTM1, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.embed = Embedding(len(vocab), embedding_size)
        self.decoder = LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers = 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.fc(features.view(features.size(0), -1))
        embeddings = self.embed(captions[:,:-1])
        
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.decoder(inputs)
        out = self.linear(hiddens)
        return out
    
    def sample(self, images, max_len, deter, temp = None):
        sampled_ids = []
        if deter:
            for i in range(max_len):
                if i == 0:
                    with torch.no_grad():
                        features = self.resnet(images)
                    inputs = self.fc(features.view(features.size(0), -1)).unsqueeze(1)
                    hiddens, states = self.decoder(inputs)
                    outputs = self.linear(hiddens.squeeze(1))
                else:
                    inputs = self.embed(predicted).unsqueeze(1)
                    hiddens, states = self.decoder(inputs, states)
                    outputs = self.linear(hiddens.squeeze(1))

                predicted = outputs.argmax(1)
                sampled_ids.append(predicted)

            return sampled_ids

        else:
            for i in range(max_len):
                if i == 0:
                    with torch.no_grad():
                        features = self.resnet(images)
                    inputs = self.fc(features.view(features.size(0), -1)).unsqueeze(1)
                    hiddens, states = self.decoder(inputs)
                    outputs = self.linear(hiddens.squeeze(1))
                else:
                    inputs = self.embed(predicted)
                    hiddens, states = self.decoder(inputs, states)
                    outputs = self.linear(hiddens.squeeze(1))

                probabilities = F.softmax(outputs.div(temp).squeeze(0).squeeze(0), dim=1) 
                predicted = torch.multinomial(probabilities.data, 1)
                sampled_ids.append(predicted)

            return sampled_ids

        
class cnnLSTM2(cnnLSTM1):
    def __init__(self, hidden_size, embedding_size, vocab):
        super().__init__(hidden_size, embedding_size, vocab)
        
#         resnet = models.resnet50(pretrained=True)
#         modules = list(resnet.children())[:-1] 
#         self.resnet = nn.Sequential(*modules)
#         self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
#         self.embed = Embedding(len(vocab), embedding_size)
#         self.linear = nn.Linear(hidden_size, len(vocab))
        
        
        self.decoder = LSTM(input_size=embedding_size * 2, hidden_size=hidden_size, num_layers = 1, batch_first=True)
    
    def forward(self, images, captions):
        
        with torch.no_grad():
            features = self.resnet(images)
        features = self.fc(features.view(features.size(0), -1))        
        
        zero_padding = torch.zeros([captions.size()[0], 1], dtype = torch.long).to('cuda')
        padded_captions = torch.cat((zero_padding, captions), 1)
        embeddings = self.embed(padded_captions[:, :-1])
        
        embed_dim = embeddings.size()[1]        
        inputs = torch.cat((features.unsqueeze(1).repeat(1, embed_dim, 1), embeddings), 2)
        
        hiddens, _ = self.decoder(inputs)
        out = self.linear(hiddens)
        return out
    
    def sample(self, images, captions, max_len, deter, temp = None):
        sampled_ids = []
        for i in range(max_len):
            if i == 0:                
                with torch.no_grad():
                    features = self.resnet(images)                
                features = self.fc(features.view(features.size(0), -1)).unsqueeze(1)
                print(features.size())
                
                zero_padding = torch.zeros([captions.size()[0], 1], dtype = torch.long).to('cuda')
                print(zero_padding.size())
                
                embeddings = self.embed(zero_padding)
                print(embeddings.size())

                inputs = torch.cat((features, embeddings), 2)
                hiddens, states = self.decoder(inputs)
                outputs = self.linear(hiddens.squeeze(1))
                
                
            else:
                ### predicted cat img
                predicted = torch.cat((features, predicted), 2)
                inputs = self.embed(predicted)
                hiddens, states = self.decoder(inputs, states)
                outputs = self.linear(hiddens.squeeze(1))

            probabilities = F.softmax(outputs.div(temp).squeeze(0).squeeze(0), dim=1) 
            predicted = torch.multinomial(probabilities.data, 1)
            sampled_ids.append(predicted)

        return sampled_ids

class cnnRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(cnnRNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.embed = Embedding(len(vocab), embedding_size)
        self.decoder = nn.RNN(input_size = embedding_size, hidden_size=hidden_size, num_layers = 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.fc(features.view(features.size(0), -1))
        embeddings = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.decoder(inputs)
        return self.linear(hiddens)
    
    def sample(self, images, max_len, deter, temp = None):
        sampled_ids = []
        if deter:
            for i in range(max_len):
                if i == 0:
                    with torch.no_grad():
                        features = self.resnet(images)
                    inputs = self.fc(features.view(features.size(0), -1)).unsqueeze(1)
                    hiddens, states = self.decoder(inputs)
                    outputs = self.linear(hiddens.squeeze(1))
                else:
                    inputs = self.embed(predicted).unsqueeze(1)
                    hiddens, states = self.decoder(inputs, states)
                    outputs = self.linear(hiddens.squeeze(1))

                predicted = outputs.argmax(1)
                sampled_ids.append(predicted)

            return sampled_ids

        else:
            for i in range(max_len):
                if i == 0:
                    with torch.no_grad():
                        features = self.resnet(images)
                    inputs = self.fc(features.view(features.size(0), -1)).unsqueeze(1)
                    hiddens, states = self.decoder(inputs)
                    outputs = self.linear(hiddens.squeeze(1))
                else:
                    inputs = self.embed(predicted)
                    hiddens, states = self.decoder(inputs, states)
                    outputs = self.linear(hiddens.squeeze(1))

                probabilities = F.softmax(outputs.div(temp).squeeze(0).squeeze(0), dim=1) 
                predicted = torch.multinomial(probabilities.data, 1)
                sampled_ids.append(predicted)

            return sampled_ids