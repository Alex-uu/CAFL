import numpy as np
from torchvision import datasets,transforms


def mnist_iid(dataset,num_users):
    num_samples = int(len(dataset) / num_users)
    index_of_samples = [i for i in range(len(dataset))]
    user_samples = {}
    for k in range(num_users):
        user_samples[k] = set(np.random.choice(index_of_samples,num_samples,replace=False))
        index_of_samples = list(set(index_of_samples) - user_samples[k])
    return user_samples


def mnist_noniid(dataset,num_users):
    num_shards,num_samples = 300,200
    index_of_shards = [i for i in range(num_shards)]
    user_samples = {k : np.array([]) for k in range(num_users)}
    index_of_samples = np.arange(num_shards * num_samples)
    labels = dataset.train_labels.numpy()
    samples_labels = np.vstack((index_of_samples,labels))
    samples_labels = samples_labels[:,samples_labels[1,:].argsort()]
    index_of_samples = samples_labels[0,:]

    for k in range(num_users):
        random_2_shards = set(np.random.choice(index_of_shards, int(num_shards / num_users), replace=False))
        index_of_shards = list(set(index_of_shards) - random_2_shards)
        for shard in random_2_shards:
            user_samples[k] = np.concatenate(
                (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    return user_samples


def mnist_noniid_unequal(dataset,num_users):
    num_shards,num_samples = 1200,50
    index_of_shards = [i for i in range(num_shards)]
    user_samples = {k : np.array([]) for k in range(num_users)}
    index_of_samples = np.arange(num_shards * num_samples)
    labels = dataset.train_labels.numpy()
    samples_labels = np.vstack((index_of_samples, labels))
    samples_labels = samples_labels[:, samples_labels[1, :].argsort()]
    index_of_samples = samples_labels[0, :]
    min_shards = 1
    max_shards = 30
    random_shards_size = np.random.randint(min_shards,max_shards + 1,size=num_users)
    random_shards_size = np.around(random_shards_size / sum(random_shards_size)
                                   * num_shards)
    random_shards_size = random_shards_size.astype(int)

    if sum(random_shards_size) > num_shards:
        for k in range(num_users):
            random_1_shard = set(np.random.choice(index_of_shards,1,replace=False))
            index_of_shards = list(set(index_of_shards) - random_1_shard)
            for shard in random_1_shard:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

        random_shards_size = random_shards_size - 1

        for k in range(num_users):
            if len(index_of_shards) == 0:
                continue
            shards_size = random_shards_size[k]
            if shards_size > len(index_of_shards):
                shards_size = len(index_of_shards)
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    else:
        for k in range(num_users):
            shards_size = random_shards_size[k]
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

        if len(index_of_shards) > 0:
            shards_size = len(index_of_shards)
            k = min(user_samples,key=lambda x: len(user_samples.get(x)))
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    return user_samples


def cifar_iid(dataset,num_users):
    num_samples = int(len(dataset) / num_users)
    index_of_samples = [i for i in range(len(dataset))]
    user_samples = {}

    for k in range(num_users):
        user_samples[k] = set(np.random.choice(index_of_samples, num_samples, replace=False))
        index_of_samples = list(set(index_of_samples) - user_samples[k])

    return user_samples


def cifar_noniid(dataset,num_users):
    num_shards, num_samples = 200, 250
    index_of_shards = [i for i in range(num_shards)]
    user_samples = {k: np.array([]) for k in range(num_users)}
    index_of_samples = np.arange(num_shards * num_samples)
    labels = np.array(dataset.targets)
    samples_labels = np.vstack((index_of_samples, labels))
    samples_labels = samples_labels[:, samples_labels[1, :].argsort()]
    index_of_samples = samples_labels[0, :]

    for k in range(num_users):
        random_2_shards = set(np.random.choice(index_of_shards, 2, replace=False))
        index_of_shards = list(set(index_of_shards) - random_2_shards)
        for shard in random_2_shards:
            user_samples[k] = np.concatenate(
                (user_samples[k], index_of_samples[shard * num_samples: (shard + 1) * num_samples]), axis=0)

    return user_samples


def get_dataset(args):
  
    train_dataset,test_dataset,user_samples = None,None,None

    if args.dataset == 'cifar':
        data_dir = '../data/cifar'

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        train_dataset = datasets.CIFAR10(data_dir,train=True,download=True,
                                         transform=transform)
        test_dataset = datasets.CIFAR10(data_dir,train=False,download=True,
                                        transform=transform)

        if args.iid:
            user_samples = cifar_iid(train_dataset,args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_samples = cifar_noniid(train_dataset,args.num_users)


    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist'
        else:
            data_dir = '../data/fmnist'

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))])

        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir,train=True,download=True,
                                           transform=transform)
            test_dataset = datasets.MNIST(data_dir,train=False,download=True,
                                          transform=transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=transform)


        if args.iid:
            user_samples = mnist_iid(train_dataset,args.num_users)
        else:
            if args.unequal:
                user_samples = mnist_noniid_unequal(train_dataset,args.num_users)
            else:
                user_samples = mnist_noniid(train_dataset,args.num_users)

    return train_dataset,test_dataset,user_samples
