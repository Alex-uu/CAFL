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

                    ''''''
                    # y = w[key].numpy().reshape((-1, 1))[:M, :]
                    ''''''
                    w_dct = dct(w[key].numpy().reshape((-1, 1)))
                    e = epoch
                    if e >= int(N / M):
                        e = e - int(N / M) * int(epoch / int(N / M))
                    y = w_dct[e * M:min((e + 1) * M, N), :]

                    # print('key: ', key, '\t y: ', y[-1, :])

                    '''差分隐私'''
                    # epsilon 张量化
                    epsilon_user = args.epsilon + np.zeros_like(y)

                    min_weight = min(y)
                    max_weight = max(y)
                    # 每一层权重或偏置的变化范围 [c-r,c+r]
                    center = (max_weight + min_weight) / 2  # + torch.zeros_like(y)
                    radius = (max_weight - center) if (max_weight - center) != 0. else 1  # + torch.zeros_like(y)
                    # print(center, radius)
                    # 参数与 center 之间的距离 μ
                    miu = y - center
                    # print('miu: ', miu)

                    # 伯努利采样概率 Pr[u=1]
                    # Pr = ((y - center) * (np.exp(epsilon_user) - 1) + radius * (np.exp(epsilon_user) + 1)) / (2 * radius * (np.exp(epsilon_user) + 1))
                    Pr = (np.exp(epsilon_user) - 1) / (2 * np.exp(epsilon_user))
                    # print('Pr: ', Pr)
                    # 伯努利变量
                    u = np.zeros_like(y)
                    for i in range(len(y)):
                        u[i, 0] = np.random.binomial(1, Pr[i, :])
                    # print('u: ', u)

                    # 自适应扰动
                    # for i in range(len(y)):
                    #     if u[i, 0] > 0:
                    #         y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) + 1) / (np.exp(epsilon_user[i, :]) - 1))
                    #     else:
                    #         y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) - 1) / (np.exp(epsilon_user[i, :]) + 1))

                    #     if u[i, 0] > 0:
                    #         y[i, :] = center + radius * ((np.exp(epsilon_user) + 1) / (np.exp(epsilon_user) - 1))
                    #     else:
                    #         y[i, :] = center - radius * ((np.exp(epsilon_user) + 1) / (np.exp(epsilon_user) - 1))
                    # y = center + miu * (np.exp(epsilon_user) + 1) / (np.exp(epsilon_user) - 1)
                    # y = center + miu * (np.exp(epsilon_user) - 1) / (np.exp(epsilon_user) + 1)

                    w[key] = torch.from_numpy(y)
                ######################################################


            # break
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            print(f'| client-{index + 1} : {k} finished!!! |')
            index += 1
        # break
        print(f'\n | Client Training End!!! | \n')


        '''Step-3 : 服务器端更新全局模型'''
        '''Step-3.1 : 平均聚合本地模型'''
        # 获取聚合前的全局模型每一层的形状
        shapes_global = copy.deepcopy(global_weights)
        for key,_ in list(shapes_global.items()):
            shapes_global[key] = shapes_global[key].shape
            # print(shapes_global[key])
        # break

        # print(local_weights)


        ''''''
        # global_weights_ = average_weights(local_weights_)
        # print('未压缩的全局模型：', global_weights_)
        partial_global_weights = average_weights(local_weights)
        # for key,_ in partial_global_weights.items():
        #     print(partial_global_weights[key].shape)
        # print('压缩后、重构前的全局模型：', partial_global_weights)
        # break

        # 计算本轮通信的平均损失
        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)


        '''Step-3.2 : 计算动量和误差反馈'''
        # ########################
        # if Monmentum == None and error == None:
        #     Monmentum = copy.deepcopy(global_weights)
        #     error = copy.deepcopy(global_weights)
        #
        #     for key,_ in list(Monmentum.items()):
        #         zeros_tensor = torch.zeros_like(Monmentum[key])
        #         Monmentum[key] = zeros_tensor
        #         zeros_tensor = torch.zeros_like(error[key])
        #         error[key] = zeros_tensor
        # ########################
        # for key,_ in list(global_weights.items()):
        #     Monmentum[key] = rho * Monmentum[key] + global_weights[key]
        #     error[key] = eta * Monmentum[key] + error[key]
        #     # print('key: ', key,'error: ', error[key])
        #     # print(error[key].shape)
        #     # break
        #
        #
        #     '''Step-3.3 : 重构全局模型（解压缩）'''
        #     N = weights_numbers[key].item()
        #     M = max(int(args.compression_ratio * N), 1)
        #
        #     ''''''
        #     # if args.model == 'mlp':
        #     #     # 重构（同样一行一行地处理）
        #     #     # 获取当前层的参数张量的维度
        #     #     dim = error[key].ndim
        #     #     # print(dim)
        #     #
        #     #     # 获取当前层的行数 rows
        #     #     if dim > 1:
        #     #         rows = error[key].shape[0]
        #     #     elif dim == 1:
        #     #         rows = 1
        #     #
        #     #     if 1 < rows:
        #     #         # 堆叠重构后的每一行 y_hat
        #     #         y_hat_matrix = None
        #     #         for row in range(rows):
        #     #             s_start = np.dot(theta.T, error[key][row].reshape((-1, 1)).numpy())
        #     #             s_finish = l1eq_pd(np.squeeze(s_start), np.squeeze(theta), [], np.squeeze(error[key][row].reshape((-1, 1)).numpy().T), 1e-3)
        #     #             # 重构后
        #     #             y_hat = np.dot(Psi, s_finish)
        #     #             # print(y_hat)
        #     #             # print(y_hat.shape)    # (1, N)
        #     #
        #     #             if y_hat_matrix is None:
        #     #                 y_hat_matrix = copy.deepcopy(torch.from_numpy(y_hat))
        #     #             else:
        #     #                 y_hat_matrix = torch.vstack((y_hat_matrix, torch.from_numpy(y_hat)))
        #     #
        #     #         # 更新全局模型参数
        #     #         global_weights[key] = y_hat_matrix
        #     #
        #     #     elif 1 == rows:
        #     #         s_start = np.dot(theta.T, error[key].reshape((-1, 1)).numpy())
        #     #         s_finish = l1eq_pd(np.squeeze(s_start), np.squeeze(theta), [], np.squeeze(error[key].reshape((-1, 1)).numpy().T), 1e-3)
        #     #         # 重构后
        #     #         y_hat = np.dot(Psi, s_finish)
        #     #         # print(y_hat.shape)
        #     #
        #     #         global_weights[key] = torch.from_numpy(y_hat.reshape(shapes_global[key]))
        #     #
        #     # elif args.model == 'cnn':
        #     #     s_start = np.dot(theta_log[key].T, error[key].numpy())
        #     #     s_finish = l1eq_pd(np.squeeze(s_start), np.squeeze(theta), [],
        #     #                        np.squeeze(error[key].numpy()), 1e-3)
        #     #     # 重构后
        #     #     x_rec = np.dot(Psi_log[key], s_finish)
        #     #
        #     #     global_weights[key] = torch.from_numpy(x_rec.reshape(shapes_global[key]))
        #
        #     ''''''
        #     ####### OMP
        #     ''''''
        #     # A = theta
        #     # K = N
        #     # y_rec = cs_omp(error[key].numpy(), A, K)
        #     # global_weights[key] = torch.from_numpy(y_rec.reshape(shapes_global[key]))
        #     ''''''
        #     # s_rec = cs_omp_1(error[key].numpy(), copy.deepcopy(theta_log[key]), N)
        #     # print('s_rec: ', s_rec)
        #     # x_rec = np.dot(Psi_log[key], s_rec)
        #     # print('x_rec: ', x_rec)
        #     ''''''
        #     # 重建矩阵
        #     rec_matrix = np.zeros((N, 1))
        #     rec_matrix[:M, :] = error[key]
        #     x_rec = idct(rec_matrix)
        #     global_weights[key] = torch.from_numpy(x_rec.reshape(shapes_global[key]))
        #
        #     print('key: ', key, '\t global_weights: ', global_weights[key].shape)

        # 重构


        for key,_ in partial_global_weights.items():
            # break
            N = weights_numbers[key].item()
            M = max(int(args.compression_ratio * N), 1)
            # 重构矩阵
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


        # break
        # print('压缩后、重构后的全局模型：', global_weights)
        global_model.load_state_dict(global_weights)
        # global_model.load_state_dict(partial_global_weights)
        print(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')


        '''Step-3.4 : 误差累积'''
        # 对重构后的全局模型再压缩
        # if args.model == 'mlp':
        #     for key,_ in list(global_weights.items()):
        #         ###################################################### 一行一行地处理
        #         # 获取当前层的参数张量的维度
        #         dim = global_weights[key].ndim
        #
        #         # 获取当前层的行数 rows 、每一行所拥有的参数个数 N
        #         if dim > 1:
        #             rows = global_weights[key].shape[0]
        #             N = global_weights[key].shape[-1]
        #         elif dim == 1:
        #             rows = 1
        #             N = len(global_weights[key])
        #
        #         # M << N
        #         M = int(N * args.compression_ratio)
        #         # 测量矩阵
        #         Phi = math.sqrt(1 / M) * torch.randn(M, N)
        #
        #         if 1 < rows:
        #             for row in range(rows):
        #                 # 压缩
        #                 y = torch.mm(Phi, global_weights[key][row].reshape((-1, 1)))    # (M, 1)
        #
        #                 # 误差累积
        #                 error[key][row] = error[key][row] - y.reshape((1, -1))
        #         elif 1 == rows:
        #             y = torch.mm(Phi, global_weights[key].reshape((-1, 1)))
        #             error[key] = error[key] - y.reshape((1, -1))
        #
        # elif args.model == 'cnn':
        #     for key, _ in list(global_weights.items()):
        #         # inv_Psi = np.linalg.inv(Psi_log[key])
        #         # s = np.dot(inv_Psi, global_weights[key].reshape((-1, 1)).numpy())
        #         # y = np.dot(Phi_log[key], global_weights[key].reshape((-1, 1)).numpy())
        #
        #         N = weights_numbers[key].item()
        #         M = max(int(args.compression_ratio * N), 1)
        #         y_dct = dct(global_weights[key].numpy().reshape((-1, 1)))
        #         y = torch.from_numpy(y_dct[:M, :])
        #
        #         # 误差累积
        #         error[key] = error[key] - y


        '''评估一轮通信后的全局模型'''
        list_acc, list_loss = [], []
        global_model.eval()
        # 在所有客户端上计算每一轮通信的平均训练精确率
        for k in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      index_of_samples=user_samples[k])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        # 计算所有客户端上评估后的平均精确率
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # 打印每一轮通信后的平均全局训练损失和精确率
        print(f'\nAvg Training States after {epoch + 1} global rounds:')
        print(f'Avg Training Loss : {train_loss[-1]}')
        print('Avg Training Accuracy : {:.2f}% \n'.format(100 * train_accuracy[-1]))

        if math.isnan(train_loss[-1]):
            train_loss.pop()
            train_accuracy.pop()
            break



    '''通信结束'''
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    runtime = time.time() - start_time
    print(('\n Total Run Time: {0:0.4f}'.format(runtime)))


    # 保存数据到 log
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
