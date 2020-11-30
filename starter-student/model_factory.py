################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

# Build and return the model here based on the configuration.

from torch.nn import Embedding, LSTM
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    resnet50 = models.resnet50(pretrained = True)
    resnet50.fc=nn.Linear(2048, embedding_size)
    
    
    embed = Embedding(len(vocab), embedding_size)
    lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers = 1, batch_first=True)
    
    raise NotImplementedError("Model Factory Not Implemented")
