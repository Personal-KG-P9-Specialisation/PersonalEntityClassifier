from svm_train import get_train_data
from torch import nn
from torch.utils.data import DataLoader, Dataset
import json
from transformers import RobertaConfig, RobertaModel,RobertaTokenizer
import torch
from sklearn import svm, metrics
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score,recall_score,f1_score

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta = RobertaModel.from_pretrained('roberta-base')
class NNDataset(Dataset):
    def __init__(self, data,gts, transform=None, target_transform=None):
        
        self.train_data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(gts,dtype=torch.float32)        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.train_data[idx], self.labels[idx]


model = nn.Sequential(
    nn.Linear(24576, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

if __name__ == "__main__":
    train,train_gt,seq_len = get_train_data('/code2/data/input1.jsonl',tokenizer,roberta)
    train_data = NNDataset(train,train_gt)
    val,val_gt,_ = get_train_data('/code2/data/input2.jsonl',tokenizer,roberta,seq_len=seq_len)
    val_data = NNDataset(val,val_gt)
    test,test_gt,_ = get_train_data('/code2/data/input3.jsonl',tokenizer,roberta,seq_len=seq_len)
    test_data = NNDataset(test,test_gt)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 16

    with torch.no_grad():
        t = torch.autograd.Variable(torch.Tensor([0.5])).to(device)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    model.to(device)


    import warnings
    warnings.simplefilter("ignore")

    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
    epochs = 100
    best_val_f1 = 0
    best_epoch = {}
    for epoch in range(epochs):
        running_loss = 0.0
        precisions = 0
        recalls = 0
        f1s = 0
    
        for i, data in enumerate(train_dataloader):    
            inputs, labels = data
        #one = inputs[1,1,:,:]
            inputs,labels = inputs.to(device), labels.to(device)
        
        
            optimizer.zero_grad()
            outputs = model(inputs)
            outs = torch.reshape((outputs > t).float(),(-1,))

            precisions += precision_score(torch.reshape(labels.cpu(),(-1,)),outs.cpu())
            recalls += recall_score(torch.reshape(labels.cpu(),(-1,)),outs.cpu())
            f1s += f1_score(torch.reshape(labels.cpu(),(-1,)),outs.cpu())

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 ==19:
                train_result = {'train_precision':precisions/i, 'train_recall':recalls/i, 'train_f1': f1s/i}
                print(f"[{epoch+1}, {i+1:5d}] loss: {running_loss/20:.3f} precision: {train_result['train_precision']:.3f} recall {train_result['train_recall']:.3f} f1 score {train_result['train_f1']:.3f}")
                running_loss = 0.0
        precisions_v = 0
        recalls_v = 0
        f1s_v = 0
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                input_valid,labels_valid = data
                input_valid,labels_valid = input_valid.to(device), labels_valid.to(device)
                out_v = torch.reshape((model(input_valid) > t).float(),(-1,))
                labels_valid = torch.reshape(labels_valid.float(),(-1,))
            
            

                precisions_v += precision_score(labels_valid.cpu(),out_v.cpu())
                recalls_v += recall_score(labels_valid.cpu(),out_v.cpu())
                f1s_v += f1_score(labels_valid.cpu(),out_v.cpu())
                if i == len(valid_dataloader)-1:
                    print(f'[{epoch+1}] val precision: {precisions_v/i:.3f} val recall {recalls_v/i:.3f} val f1 score {f1s_v/i:.3f}')
                    if (f1s_v/i) > best_val_f1:
                        print('Achieved a better model')
                        torch.save(model.state_dict(), f"{epoch+1}_'nn'_weights.pth")
                        best_val_f1 = (f1s_v/i)
                        best_epoch = {
                            'epoch': epoch+1,
                            'val_precision': (precisions_v/i), 
                            'val_recall': recalls_v/i, 
                            'val_f1':f1s_v/i,
                            'train_precision': train_result['train_precision'],
                            'train_recall': train_result['train_recall'],
                            'train_f1': train_result['train_f1']
                    }


    print('Finished Training')
    print(f"Best Training in Epoch {best_epoch['epoch']} with train: precision {best_epoch['train_precision']:.3f}, recall {best_epoch['train_recall']:.3f}, f1 score {best_epoch['train_f1']:.3f}; valid: precision {best_epoch['val_precision']:.3f} recall {best_epoch['val_recall']:.3f} f1 {best_epoch['val_f1']:.3f}")
    with torch.no_grad():
        t = torch.autograd.Variable(torch.Tensor([0.5]))
        precisions_t = 0
        recalls_t = 0
        f1s_t = 0
        for data in test_dataloader:
            inputs,labels = data
            model = model.cpu()
            outputs = model(inputs)
            outs = torch.reshape((outputs > t).float(),(-1,))
            labels = torch.reshape(labels,(-1,))
            precisions_t += precision_score(labels,outs)
            recalls_t += recall_score(labels,outs)
            f1s_t += f1_score(labels,outs)
        print(f'Test results: precision {precisions_t/len(test_dataloader)}, recall {recalls_t/len(test_dataloader)}, f1 score {f1s_t/len(test_dataloader)}')