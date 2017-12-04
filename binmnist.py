from __future__ import print_function

import torch
import torch.utils.data as data_utils

import numpy as np
import os
import itertools as it
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def get_list():
    tuples = it.product([0, 1], repeat=7)
    arrlist = []
    for tup in tuples:
        # arrtemp = np.asarray(tup, dtype=np.int32)
        # arrlist.append(arrtemp)
        arrlist.append(tup)
    # print(arrlist[0])
    # print(arrlist[0].dtype)
    indexed = {row:idx for idx, row in enumerate(arrlist)}
    return indexed

def load_binary(bs, workers, pin):
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data/mnist', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=bs, shuffle=True)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data/mnist', train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=bs, shuffle=False)

    x_train = train_loader.dataset.train_data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )
    y_train = np.array( train_loader.dataset.train_labels.float().numpy(), dtype=int)

    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )
    y_test = np.array( test_loader.dataset.test_labels.float().numpy(), dtype=int)

    np.random.seed(4648)
    x_train[x_train > 0.67] = 1
    x_test[x_test > 0.67] = 1
    x_train = np.random.binomial(1, x_train)
    x_test = np.random.binomial(1, x_test)
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))
    # print(x_test[128], x_test[128].T)
    # xtv = np.empty((x_test.shape[0], 112, 7), dtype=np.int32)
    xtv = np.empty((x_test.shape[0], 112, 1), dtype=np.int32)
    # xth = np.empty((x_test.shape[0], 112, 7), dtype=np.int32)

    # for i in range(x_test.shape[0]):
    #     temp = np.hsplit(x_test[i], 4)
    #     xtv[i] = np.concatenate(temp)
    #     # temp = np.hsplit(x_test[i].T, 4)
    #     # xth[i] = np.concatenate(temp)

    # both = np.concatenate((xtv[128], xth[128]))
    # print(both[60:80])
    # print(both[172:192])

    indexed = get_list()
    nptemp = np.empty((112, 1), dtype=np.int32)
    for i in range(x_test.shape[0]):
        temp = np.hsplit(x_test[i], 4)
        temp = np.concatenate(temp)
        for idx in range(len(temp)):
            tempa = temp[idx].tolist()
            # print(tempa)
            tempb = tuple(tempa)
            # print(tempb)
            # print(indexed[tempb])
            nptemp[idx] = indexed[tempb]
        xtv[i] = nptemp
    print(xtv[128])

    # validation set
    # x_val = x_train[55000:60000]
    # y_val = np.array(y_train[55000:60000], dtype=int)
    # x_train = x_train[0:55000]
    # y_train = np.array(y_train[0:55000], dtype=int)
    #
    # train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    # train_loader = data_utils.DataLoader(train, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)
    #
    # validation = data_utils.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    # val_loader = data_utils.DataLoader(validation, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
    #
    # test = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    # test_loader = data_utils.DataLoader(test, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
    #
    # return train_loader, val_loader, test_loader

def load_static_mnist(bs, workers, pin):
    # convert .amat files to numpy arrays
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join('../data', 'bin_mnist', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('../data', 'bin_mnist', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('../data', 'bin_mnist', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    # np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )
    print(y_val.shape)
    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)

    return train_loader, val_loader, test_loader

# ======================================================================================================================
def load_dynamic_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.train_data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )

    y_train = np.array( train_loader.dataset.train_labels.float().numpy(), dtype=int)

    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array( test_loader.dataset.test_labels.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_train = np.random.binomial(1, x_train)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

load_binary(64, 1, True)
# get_list()
