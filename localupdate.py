import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset


class DatasetSplit(Dataset):
    def __init__(self,dataset,index_of_samples):
        self.dataset = dataset
        self.index_of_samples = [int(index) for index in index_of_samples]

    def __len__(self):
        return len(self.index_of_samples)

    def __getitem__(self,item):
        image,label = self.dataset[self.index_of_samples[item]]
        return torch.tensor(image),torch.tensor(label)


class LocalUpdate(object):
    def __init__(self,args,dataset,index_of_samples):
        self.args = args
        self.trainloader,self.validloader,self.testloader = self.train_valid_test(
            dataset,list(index_of_samples))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss()

    def train_valid_test(self, dataset, index_of_samples):
        index_of_train = index_of_samples[:int(0.8 * len(index_of_samples))]
        index_of_valid = index_of_samples[int(0.8 * len(index_of_samples)) : int(0.9 * len(index_of_samples))]
        index_of_test = index_of_samples[int(0.9 * len(index_of_samples)) :]

        train_loader = DataLoader(DatasetSplit(dataset,index_of_train),
                                  batch_size=self.args.local_bs,shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, index_of_valid),
                                  batch_size=max(int(len(index_of_valid) / 10), 1), shuffle=False)
        test_loader = DataLoader(DatasetSplit(dataset, index_of_test),
                                  batch_size=max(int(len(index_of_test) / 10), 1), shuffle=False)

        return train_loader,valid_loader,test_loader

    def local_train(self,model,global_round):
        model.train()
        epoch_loss = []

        optimizer = None
        if self.args.optimizer =='sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for index_of_batch,(images,labels) in enumerate(self.trainloader):
                images,labels = images.to(self.device),labels.to(self.device)
                model.zero_grad()
                labels_hat = model(images)
                loss = self.criterion(labels_hat,labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (index_of_batch % 10 == 0):
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     global_round + 1, iter + 1, index_of_batch * len(images),
                    #     len(self.trainloader.dataset),
                    #     100. * index_of_batch / len(self.trainloader),loss.item()
                    # ))
                    batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0 , 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            total += len(labels)

        accuracy = correct / total
        return accuracy, loss
