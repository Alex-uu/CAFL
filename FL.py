import time
from tqdm import tqdm
import copy
import math

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
from scipy.fft import dct, idct
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from options import args_parser,exp_details
from dataset import get_dataset
from models import MLP,CNNMnist,CNNFashion_Mnist,CNNCifar
from localupdate import LocalUpdate
from globalupdate import average_weights, test_inference


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'

    train_dataset, test_dataset, user_samples = get_dataset(args)

    global_model = None
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in,dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()
    global_weights_ = copy.deepcopy(global_weights)

    train_loss,train_accuracy = [],[]
    val_acc_list,net_list = [],[]
    cv_loss,cv_acc = [],[]
    val_loss_pre,counter = 0,0

    weights_numbers = copy.deepcopy(global_weights)

    for epoch in tqdm(list(range(args.epochs))):

        local_weights, local_losses = [], []
        local_weights_ = []

        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users),1)
        index_of_users = np.random.choice(list(range(args.num_users)), m, replace=False)
        index = 0

        for k in index_of_users:

            local_model = LocalUpdate(args=args,dataset=train_dataset,
                                      index_of_samples=user_samples[k])

            w,loss = local_model.local_train(
                model=copy.deepcopy(global_model),global_round=epoch)

            local_weights_.append(copy.deepcopy(w))

            if args.model == 'cnn':
                for key,_ in list(w.items()):
                    
                    N = w[key].numel()
                    weights_numbers[key] = torch.tensor(N)
                    M = max(int(args.compression_ratio * N), 1)

                    w_dct = dct(w[key].numpy().reshape((-1, 1)))
                    e = epoch
                    if e >= int(N / M):
                        e = e - int(N / M) * int(epoch / int(N / M))
                    y = w_dct[e * M:min((e + 1) * M, N), :]

                    epsilon_user = args.epsilon + np.zeros_like(y)

                    min_weight = min(y)
                    max_weight = max(y)
                    center = (max_weight + min_weight) / 2  # + torch.zeros_like(y)
                    radius = (max_weight - center) if (max_weight - center) != 0. else 1  # + torch.zeros_like(y)
                    miu = y - center
                    Pr = (np.exp(epsilon_user) - 1) / (2 * np.exp(epsilon_user))
                    u = np.zeros_like(y)
                    for i in range(len(y)):
                        u[i, 0] = np.random.binomial(1, Pr[i, :])

                    for i in range(len(y)):
                        if u[i, 0] > 0:
                            y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) + 1) / (np.exp(epsilon_user[i, :]) - 1))
                        else:
                            y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) - 1) / (np.exp(epsilon_user[i, :]) + 1))

                    #     if u[i, 0] > 0:
                    #         y[i, :] = center + radius * ((np.exp(epsilon_user) + 1) / (np.exp(epsilon_user) - 1))
                    #     else:
                    #         y[i, :] = center - radius * ((np.exp(epsilon_user) + 1) / (np.exp(epsilon_user) - 1))

                    w[key] = torch.from_numpy(y)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            print(f'| client-{index + 1} : {k} finished!!! |')
            index += 1
            
        print(f'\n | Client Training End!!! | \n')


        shapes_global = copy.deepcopy(global_weights)
        for key,_ in list(shapes_global.items()):
            shapes_global[key] = shapes_global[key].shape

        partial_global_weights = average_weights(local_weights)
        
        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)

        for key,_ in partial_global_weights.items():
            N = weights_numbers[key].item()
            M = max(int(args.compression_ratio * N), 1)
            rec_matrix = np.zeros((N, 1))
            e = epoch
            if e >= int(N / M):
                e = e - int(N / M) * int(epoch / int(N / M))
            rec_matrix[e * M:min((e + 1) * M, N), :] = partial_global_weights[key]
            x_rec = idct(rec_matrix)
            global_weights_1D = global_weights[key].numpy().reshape((-1, 1))
            global_weights_1D[e * M:min((e + 1) * M, N), :] = (global_weights_1D[e * M:min((e + 1) * M, N), :] + x_rec[e * M:min((e + 1) * M, N), :]) / 2
            global_weights[key] = torch.from_numpy(global_weights_1D.reshape(shapes_global[key]))

            # global_weights[key] = partial_global_weights[key].reshape(shapes_global[key])

            print('key: ', key, '\t global_weights: ', global_weights[key].shape)

        global_model.load_state_dict(global_weights)
        # global_model.load_state_dict(partial_global_weights)
        print(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')

        list_acc, list_loss = [], []
        global_model.eval()
        for k in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      index_of_samples=user_samples[k])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        print(f'\nAvg Training States after {epoch + 1} global rounds:')
        print(f'Avg Training Loss : {train_loss[-1]}')
        print('Avg Training Accuracy : {:.2f}% \n'.format(100 * train_accuracy[-1]))

        if math.isnan(train_loss[-1]):
            train_loss.pop()
            train_accuracy.pop()
            break

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    runtime = time.time() - start_time
    print(('\n Total Run Time: {0:0.4f}'.format(runtime)))

    
    data_log= {'Train Loss' : train_loss, 'Train Accuracy' : train_accuracy,
               'Test Loss' : test_loss, 'Test Accuracy' : test_acc}
    record = pd.DataFrame(data_log)
    record.to_csv('../log/CIFAR/p_fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].csv'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))




    matplotlib.use('Agg')
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_E[{}]_iid[{}]_CR[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.iid, args.compression_ratio))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(list(range(len(train_accuracy))), train_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/CIFAR/p_fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].png'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))
