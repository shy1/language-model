# pytorch implementation of:
# "Reservoir Computing on the Hypersphere" - M. Andrecut arXiv:1706.07896
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sru.cuda_functional import SRU, SRUCell
import re
import pickle
import time
from random import shuffle
import numpy as np
import math
import os
import argparse
import loaders
import sys

global dtype
global i_size
global h_size
global trainchunks


dtype = torch.cuda.FloatTensor
loadsaved = 0
n = 2
stride = 1
# .03444, .05573, .09 .146, .236 .382 .618
leak = 0.382
step = 0
lrate = 0.0001
chunklen = 112
bs = 64
trainsize = 0
i_size = 128
mult = 4
# h_size = 1024 * mult
h_size = 128
inp_size = int(h_size / 3)
o_size = 2000
out_size = 10
s_size = 256
l_size = h_size + s_size
layers = 3
lr_period = 10
dropout = 0.2
rnndrop = 0.2
lindrop = 0.382

torch.cuda.manual_seed(481639)
torch.manual_seed(481639)
rnnstr = str(layers) + "x" + str(s_size) + "."
weightfile = '/home/user01/dev/language-model/saved/mnistR.mq4.Lin.' + rnnstr + str(h_size) + '.bs1.NA.382.a0001.p'

# aestr = "1024" + "x" + "2048" + "x"
# aefile = '/home/user01/dev/language-model/saved/scoderMSE.' + aestr + str(h_size) + '.TNH.4405.a0001.p'
# aefile = '/home/user01/dev/language-model/saved/scoderCOS.1920x4096.PRLU.4405.a0001.p'

def trainSO():

    weightfile = '/home/user01/dev/language-model/saved/mnistNoESN.V2D2O382.' + rnnstr + str(i_size) + '.BD.bs64.a0001.p'
    # aesaved = torch.load(aefile)
    # ae = AutoEncoder(h_size, o_size)
    # ae.load_state_dict(aesaved["AE"])
    # ae.cuda()

    if loadsaved:
        saved = torch.load(weightfile)
        U1 = saved["U1"]
    else:
        U1 = makeU(h_size, i_size)

    # weightfile = '/home/user01/dev/language-model/saved/weights5x768.' + str(h_size) + '.p31.s0.4405.b1a002.p'
    allweights = dict()
    allweights["U1"] = U1

    model = Model(i_size, s_size, out_size=out_size, layers=layers, dropout=dropout, rnndrop=rnndrop, lindrop=lindrop)

    if loadsaved:
        model.load_state_dict(saved["SRU"])

    model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Model Parameters:", params)

    trainset = loaders.MooreMnistNoESN(U1, h_size, tvt='train', leakrate=leak)
    valset = loaders.MooreMnistNoESN(U1, h_size, tvt='val', leakrate=leak)
    # targets = trainset.getTargets()
    # targets = torch.from_numpy(targets)
    # targets.cuda()
    # targetv = Variable(targets, requires_grad=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=3, drop_last=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=True, num_workers=3, drop_last=True, pin_memory=True)

    print("loaded: {} file: {}".format(loadsaved, weightfile))
    print("lr: {} leak: {} hidden: {} VDr: {} Dr: {} SRU: {}x{}".format(lrate, leak, h_size, model.rnndrop, model.lindrop, layers, s_size))
    model.print_pnorm()

    criterion = nn.CrossEntropyLoss().cuda()
    needs_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(needs_grad, model.parameters()),
        lr=lrate
    )
    # optimizer = optim.Adam(model.parameters(), lr=lrate)
    if loadsaved:
        optimizer.load_state_dict(saved["opt"])
    # optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.99, weight_decay=0, nesterov=True)
    interval = 78.0
    lastidx = chunklen - 1
    batchloss = torch.Tensor(1).type(dtype)
    eploss = torch.Tensor(1).type(dtype)
    batcherr = torch.cuda.LongTensor(1)
    # out = Variable(torch.Tensor(1, 10).type(dtype))
    startp = time.perf_counter()
    for ep in range(64):
        batcherr[0] = 0
        batchloss[0] = 0
        eploss[0] = 0
        model.train()
        count = 0
        for i, (states, posed, targets) in enumerate(trainloader):
            states = states.permute(1, 0, 2)
            posed = posed.permute(1, 0, 2).contiguous()
            print(states.size(), posed.size(), targets.size())

            targets = targets.contiguous()
            targets = targets.cuda(async=True)
            states = states.contiguous()
            states = states.cuda(async=True)
            posed = posed.cuda(async=True)
            vtargets = Variable(targets, requires_grad=False)
            vstates = Variable(states, requires_grad=False)
            vposed = Variable(posed, requires_grad=False)
            # if i == 0:
            #     print(rnnstates.size())
            optimizer.zero_grad()
            # print(vstates)
            output = model(vstates, vposed)
            # output = output.permute(1, 0, 2)
            # if i == 0:
            #     print("o/t:", output, vtargets.data.size())
            loss = criterion(output, vtargets.squeeze())
            loss.backward()
            # norm_grads(model.parameters())
            optimizer.step()

            batchloss[0] += loss.data[0]
            top_n, top_i = output.topk(1)
            # print("o/t:", top_i.data.squeeze().size(), vtargets.data.size())
            # batcherr[0] += int(top_i[0][0].data != vtargets.squeeze().data)
            batcherr[0] += top_i.data.squeeze().ne(vtargets.squeeze().data).sum()


        elapsedp = time.perf_counter() - startp
        tm, ts = divmod(elapsedp, 60)
        # eploss[0] += batchloss[0] / interval
        bloss = batchloss[0]
        avgerr = batcherr[0]
        # print(len(trainset), "v:", len(valset), len(valloader))
        print("{:3d} {:.5f} {:.5f} {:4d}m {:02d}s".format(ep, bloss / len(trainloader), avgerr / (len(trainloader) * bs), int(tm), int(ts)))
        # eloss = eploss[0]
        # print("{:3d} {:.5f}".format(ep, eloss / 11))
        allweights["SRU"] = model.state_dict()
        allweights["opt"] = optimizer.state_dict()
        torch.save(allweights, weightfile)
        model.print_pnorm()
        batcherr[0] = 0
        batchloss[0] = 0
        model.eval()
        for i, (states, posed, targets) in enumerate(valloader):
                states = states.permute(1, 0, 2)
                posed = posed.permute(1, 0, 2).contiguous()

                targets = targets.contiguous()
                targets = targets.cuda(async=True)
                states = states.contiguous()
                states = states.cuda(async=True)
                posed = posed.cuda(async=True)
                vtargets = Variable(targets, requires_grad=False)
                vstates = Variable(states, requires_grad=False)
                vposed = Variable(posed, requires_grad=False)
            # if i == 0:
            #     print(rnnstates.size())
            optimizer.zero_grad()
            output = model(vstates, vposed)
            loss = criterion(output, vtargets.squeeze())

            batchloss[0] += loss.data[0]
            top_n, top_i = output.topk(1)
            # batcherr[0] += int(top_i[0][0].data != vtargets.squeeze().data)
            batcherr[0] += top_i.data.squeeze().ne(vtargets.squeeze().data).sum()

        bloss = batchloss[0]
        avgerr = batcherr[0]
        print("\nValidation Set")
        print("{:3d} {:6d} {:.5f} {:.5f}\n".format(ep, len(valset), bloss / len(valloader), avgerr / (len(valloader) * bs)))
        batchloss[0] = 0
        batcherr[0] = 0

def trainlin():

    # weightfile = '/home/user01/dev/language-model/saved/weights5x768.' + str(h_size) + '.p31.s0.4405.b1a002.p'
    # aesaved = torch.load(aefile)
    # ae = AutoEncoder(h_size, o_size)
    # ae.load_state_dict(aesaved["AE"])
    # ae.cuda()

    if loadsaved:
        saved = torch.load(weightfile)
        U1 = saved["U1"]
    else:
        U1 = makeU(h_size, i_size)

    # weightfile = '/home/user01/dev/language-model/saved/weights5x768.' + str(h_size) + '.p31.s0.4405.b1a002.p'
    allweights = dict()
    allweights["U1"] = U1

    # model = Model(h_size, s_size, out_size=out_size, layers=layers, dropout=dropout, rnndrop=rnndrop, lindrop=lindrop)
    model = Net()
    # lin = nn.Linear(chunklen * h_size, out_size).cuda()
    # drop = nn.Dropout(lindrop)
    if loadsaved:
        model.load_state_dict(saved["LW"])

    model.cuda()
    # W1 = Variable(torch.Tensor(chunklen * h_size).type(dtype))

    val_range = (3.0/(h_size * chunklen))**0.5
    print(val_range)
    for p in model.parameters():
        print(p.size(), p.dim())
        # p.data.uniform_(-val_range, val_range)
        if p.dim() > 1:  # matrix
            p.data.uniform_(-val_range, val_range)
        else:
            p.data.zero_()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Model Parameters:", params)

    trainset = loaders.MooreMnistSRU(U1, h_size, tvt='train', leakrate=leak)
    valset = loaders.MooreMnistSRU(U1, h_size, tvt='val', leakrate=leak)
    # targets = trainset.getTargets()
    # targets = torch.from_numpy(targets)
    # targets.cuda()
    # targetv = Variable(targets, requires_grad=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=7, drop_last=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=True, num_workers=7, drop_last=True, pin_memory=True)

    print("loaded: {} file: {}".format(loadsaved, weightfile))
    print("lr: {} leak: {} hidden: {} VDr: {} Dr: {} SRU: {}x{}".format(lrate, leak, h_size, rnndrop, lindrop, layers, s_size))
    # model.print_pnorm()

    criterion = nn.NLLLoss().cuda()
    # needs_grad = lambda x: x.requires_grad
    # optimizer = optim.Adam(
    #     filter(needs_grad, model.parameters()),
    #     lr=lrate
    # )
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    if loadsaved:
        optimizer.load_state_dict(saved["opt"])
    # optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.99, weight_decay=0, nesterov=True)
    interval = 5000.0
    lastidx = chunklen - 1
    batchloss = torch.Tensor(1).type(dtype)
    eploss = torch.Tensor(1).type(dtype)
    batcherr = torch.cuda.LongTensor(1)
    first10 = torch.cuda.LongTensor(2, 10)
    # out = Variable(torch.Tensor(1, 10).type(dtype))
    startp = time.perf_counter()
    for ep in range(24):
        batcherr[0] = 0
        batchloss[0] = 0
        eploss[0] = 0
        model.train()
        count = 0
        for i, (states, targets) in enumerate(trainloader):
            states = states.permute(1, 0, 2)
            # print(states.size(), targets.size())
            # targets = targets.permute(1, 0, 2)

            ## TODO: test if calling cuda() outside of variable definition
            ## has any effect on performance
            targets = targets.contiguous()
            targets = targets.cuda(async=True)
            states = states.contiguous()
            states = states.cuda(async=True)
            vtargets = Variable(targets, requires_grad=False)
            vstates = Variable(states, requires_grad=False)
            # if i == 0:
            #     print(rnnstates.size())
            optimizer.zero_grad()
            output = model(vstates.view(-1, chunklen * h_size))
            loss = criterion(output, vtargets.squeeze())
            # print("o/t:", output, vtargets.data.size())
            # loss = criterion(output, vtargets.squeeze())

            loss.backward()
            # norm_grads(model.parameters())
            optimizer.step()

            batchloss[0] += loss.data[0]
            top_n, top_i = output.topk(1)
            # print("o/t:", top_i.data.squeeze().size(), vtargets.data.size())
            # batcherr[0] += int(top_i[0][0].data != vtargets.squeeze().data)
            batcherr[0] += top_i.data.squeeze().ne(vtargets.squeeze().data).sum()
            # count += bs
            # if i == 0:
            #     print("be/topi/targets:", batcherr, top_i.data.squeeze(), vtargets.squeeze().data)
            if i < 10:
                # print(top_i.data, vtargets.data)
                first10[0][i] = top_i.squeeze().data[0]
                first10[1][i] = vtargets.data[0]
                if i == 9:
                    print("be/topi/targets:", batcherr / bs, first10)
            if (i + 1) % interval == 0:
                elapsedp = time.perf_counter() - startp
                tm, ts = divmod(elapsedp, 60)
                eploss[0] += batchloss[0] / (interval * bs)
                bloss = batchloss[0]
                avgerr = batcherr[0]
                # print("c:ixb", count, interval * bs)
                print("{:3d} {:6d} {:.5f} {:.5f} {:4d}m {:02d}s".format(ep, i + 1, bloss / (interval * bs), avgerr / (interval * bs), int(tm), int(ts)))
                batchloss[0] = 0
                batcherr[0] = 0
                count = 0

        eloss = eploss[0]
        print("{:3d} {:.5f}".format(ep, eloss / 11))
        allweights["LW"] = model.state_dict()
        allweights["opt"] = optimizer.state_dict()
        torch.save(allweights, weightfile)
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in model.parameters() ]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))
        # model.print_pnorm()
        batcherr[0] = 0
        batchloss[0] = 0
        model.eval()
        for i, (states, targets) in enumerate(valloader):
            states = states.permute(1, 0, 2)
            # targets = targets.permute(1, 0, 2)

            targets = targets.contiguous()
            targets = targets.cuda(async=True)
            states = states.contiguous()
            states = states.cuda(async=True)
            vtargets = Variable(targets, requires_grad=False)
            vstates = Variable(states, requires_grad=False)
            # if i == 0:
            #     print(rnnstates.size())
            optimizer.zero_grad()
            # output = lin(vstates.view(-1))
            # output = F.log_softmax(output, dim=0)
            # loss = F.nll_loss(output.unsqueeze(0), vtargets.squeeze())
            output = model(vstates.view(-1, chunklen * h_size))
            loss = criterion(output, vtargets.squeeze())


            batchloss[0] += loss.data[0]
            top_n, top_i = output.topk(1)
            # batcherr[0] += int(top_i[0][0].data != vtargets.squeeze().data)
            batcherr[0] += top_i.data.squeeze().ne(vtargets.squeeze().data).sum()

        bloss = batchloss[0]
        avgerr = batcherr[0]
        print("\nValidation Set")
        print("{:3d} {:6d} {:.5f} {:.5f}\n".format(ep, len(valloader), bloss / (len(valloader) * bs), avgerr / (len(valloader) * bs)))
        batchloss[0] = 0
        batcherr[0] = 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop = nn.Dropout(lindrop)
        self.lin = nn.Linear(chunklen * h_size, out_size)

    def forward(self, x):
        output = self.drop(x)
        output = self.lin(output)
        return F.log_softmax(output, dim=1)

def trainset():

    # weightfile = '/home/user01/dev/language-model/saved/weights5x768.' + str(h_size) + '.p31.s0.4405.b1a002.p'
    # aesaved = torch.load(aefile)
    # ae = AutoEncoder(h_size, o_size)
    # ae.load_state_dict(aesaved["AE"])
    # ae.cuda()

    if loadsaved:
        saved = torch.load(weightfile)
        U1 = saved["U1"]
    else:
        U1 = makeU(h_size, i_size)

    # weightfile = '/home/user01/dev/language-model/saved/weights5x768.' + str(h_size) + '.p31.s0.4405.b1a002.p'
    allweights = dict()
    allweights["U1"] = U1

    model = Model(h_size, s_size, out_size=out_size, layers=layers, dropout=dropout, rnndrop=rnndrop, lindrop=lindrop)

    if loadsaved:
        model.load_state_dict(saved["SRU"])

    model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Model Parameters:", params)

    trainset = loaders.MooreMnistSRU(U1, h_size, tvt='train', leakrate=leak)
    valset = loaders.MooreMnistSRU(U1, h_size, tvt='val', leakrate=leak)
    # targets = trainset.getTargets()
    # targets = torch.from_numpy(targets)
    # targets.cuda()
    # targetv = Variable(targets, requires_grad=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=7, drop_last=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=True, num_workers=7, drop_last=True, pin_memory=True)

    print("loaded: {} file: {}".format(loadsaved, weightfile))
    print("lr: {} leak: {} hidden: {} VDr: {} Dr: {} SRU: {}x{}".format(lrate, leak, h_size, model.rnndrop, model.lindrop, layers, s_size))
    model.print_pnorm()

    criterion = nn.CrossEntropyLoss().cuda()
    needs_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(needs_grad, model.parameters()),
        lr=lrate
    )
    # optimizer = optim.Adam(model.parameters(), lr=lrate)
    if loadsaved:
        optimizer.load_state_dict(saved["opt"])
    # optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.99, weight_decay=0, nesterov=True)
    interval = 5000.0
    lastidx = chunklen - 1
    batchloss = torch.Tensor(1).type(dtype)
    eploss = torch.Tensor(1).type(dtype)
    batcherr = torch.cuda.LongTensor(1)
    # out = Variable(torch.Tensor(1, 10).type(dtype))
    startp = time.perf_counter()
    for ep in range(24):
        batcherr[0] = 0
        batchloss[0] = 0
        eploss[0] = 0
        model.train()
        count = 0
        for i, (states, targets) in enumerate(trainloader):
            states = states.permute(1, 0, 2)
            # print(states.size(), targets.size())
            # targets = targets.permute(1, 0, 2)

            ## TODO: test if calling cuda() outside of variable definition
            ## has any effect on performance
            targets = targets.contiguous()
            targets = targets.cuda(async=True)
            states = states.contiguous()
            states = states.cuda(async=True)
            vtargets = Variable(targets, requires_grad=False)
            vstates = Variable(states, requires_grad=False)
            # if i == 0:
            #     print(rnnstates.size())
            optimizer.zero_grad()
            output = model(vstates)

            # print("o/t:", output, vtargets.data.size())
            loss = criterion(output, vtargets.squeeze())
            loss.backward()
            # norm_grads(model.parameters())
            optimizer.step()

            batchloss[0] += loss.data[0]
            top_n, top_i = output.topk(1)
            # print("o/t:", top_i.data.squeeze().size(), vtargets.data.size())
            # batcherr[0] += int(top_i[0][0].data != vtargets.squeeze().data)
            batcherr[0] += top_i.data.squeeze().ne(vtargets.squeeze().data).sum()
            # count += bs
            if i == 0:
                print("be/topi/targets:", batcherr, top_i.data.squeeze(), vtargets.squeeze().data)
            if (i + 1) % interval == 0:
                elapsedp = time.perf_counter() - startp
                tm, ts = divmod(elapsedp, 60)
                eploss[0] += batchloss[0] / (interval * bs)
                bloss = batchloss[0]
                avgerr = batcherr[0]
                # print("c:ixb", count, interval * bs)
                print("{:3d} {:6d} {:.5f} {:.5f} {:4d}m {:02d}s".format(ep, i + 1, bloss / (interval * bs), avgerr / (interval * bs), int(tm), int(ts)))
                batchloss[0] = 0
                batcherr[0] = 0
                count = 0

        eloss = eploss[0]
        print("{:3d} {:.5f}".format(ep, eloss / 11))
        allweights["SRU"] = model.state_dict()
        allweights["opt"] = optimizer.state_dict()
        torch.save(allweights, weightfile)
        model.print_pnorm()
        batcherr[0] = 0
        batchloss[0] = 0
        model.eval()
        for i, (states, targets) in enumerate(valloader):
            states = states.permute(1, 0, 2)
            # targets = targets.permute(1, 0, 2)

            targets = targets.contiguous()
            targets = targets.cuda(async=True)
            states = states.contiguous()
            states = states.cuda(async=True)
            vtargets = Variable(targets, requires_grad=False)
            vstates = Variable(states, requires_grad=False)
            # if i == 0:
            #     print(rnnstates.size())
            optimizer.zero_grad()
            output = model(vstates)
            loss = criterion(output, vtargets.squeeze())

            batchloss[0] += loss.data[0]
            top_n, top_i = output.topk(1)
            # batcherr[0] += int(top_i[0][0].data != vtargets.squeeze().data)
            batcherr[0] += top_i.data.squeeze().ne(vtargets.squeeze().data).sum()

        bloss = batchloss[0]
        avgerr = batcherr[0]
        print("\nValidation Set")
        print("{:3d} {:6d} {:.5f} {:.5f}\n".format(ep, len(valloader), bloss / (len(valloader) * bs), avgerr / (len(valloader) * bs)))
        batchloss[0] = 0
        batcherr[0] = 0

def norm_grads(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    for p in parameters:
        gnorm = p.grad.data.norm(norm_type)
        if gnorm > 1:
            p.grad.data.div_(gnorm + 1e-8)


class Model(nn.Module):
    def __init__(self, in_size, hid_size, out_size=10, layers=2, dropout=0.0, rnndrop=0.2, lindrop=0.0):
        super(Model, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.rnndrop = rnndrop
        self.lindrop = lindrop

        self.drop = nn.Dropout(self.lindrop)
        self.rnn = SRU(self.in_size, self.hid_size,
            num_layers = self.layers,          # number of stacking RNN layers
            dropout = self.dropout,           # dropout applied between RNN layers
            rnn_dropout = self.rnndrop,       # variational dropout applied on linear transformation
            use_tanh = 0,            # use tanh?
            use_relu = 1,            # use ReLU?
            bidirectional = True    # bidirectional RNN ?
        )
        self.output_layer = nn.Linear(self.hid_size * chunklen * 2, self.out_size)

        self.init_weights()
        # self.rnn.set_bias(args.bias)

    def init_weights(self):
        val_range = (3.0/(self.hid_size*chunklen))**0.5
        print(val_range)
        for p in self.parameters():
            print(p.size(), p.dim())
            # p.data.uniform_(-val_range, val_range)
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()


    def forward(self, x):
        # print("x:", x.size())
        # if self.lindrop > 0:
        #     x = self.drop(x)
        output, hidden = self.rnn(x)
        # print("out1:", output.size())
        output = output.permute(1, 0, 2)
        output = output.contiguous()
        output = output.view(-1, self.hid_size * chunklen * 2)
        # print("out2:", output.size())
        if self.lindrop > 0:
            output = self.drop(output)
        output = self.output_layer(output)
        return output

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.layers, batch_size, self.hid_size).zero_())
        return zeros

    def print_pnorm(self):
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in self.parameters() ]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))

class Hypervoir(nn.Module):
    def __init__(self, i_size, h_size, leak, U):
        super(Hypervoir, self).__init__()
        self.leak = leak
        self.h_size = h_size
        self.lastidx = h_size - 1
        self.U = U
        self.hidden = Variable(torch.zeros(self.h_size).type(dtype), requires_grad=False)
        self.rollend = torch.FloatTensor(1, 1).type(dtype)
        self.rollfront = torch.FloatTensor(1, self.lastidx).type(dtype)

        # self.hidden2out = nn.Linear(h_size, i_size, bias=False)
        # self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, bs):
        return Variable(torch.zeros(bs, self.h_size).type(dtype), requires_grad=False)

    def roll(self, hidden):
        # self.rollend[0] = hidden[0][self.lastidx]
        self.rollend = hidden.narrow(1, self.lastidx, 1)
        self.rollfront = hidden.narrow(1, 0, self.lastidx)
        hidden = torch.cat((self.rollend, self.rollfront), 1)
        return hidden
    # input (u_col) is the column from random input weight matrix U
    # with the same index as the input bigram
    def forward(self, chunk):
        self.hidden = self.primeStates(chunk, self.U, self.leak)
        self.hidden = self.getLastState(chunk, self.hidden, self.U, self.leak)
        # output = self.hidden2out(hidden)
        # output = self.softmax(output)
        return hidden

    def nextHidden(self, hidd, ingot, leak):
        hidd = (1.0 - leak) * hidd + leak * (ingot + self.roll(hidd))
        hidd = hidd / hidd.norm()
        return hidd

    def primeStates(self, chunk, w_in, leak):
        ## run through the sequence in the reverse direction prior to the forward
        #  pass in order prime the hidden states and avoid the negative effects
        # of initial reservoir transience
        ## set first hidden state directly as the input weight matrix's column
        # rather than using leaky combinations with an inital all-zeros state
        last = len(chunk) - 1
        hidd = w_in[:, chunk[last]].unsqueeze(0)
        hidd = hidd / hidd.norm()
        for inp in reversed(chunk[1:last]):
            ingot = w_in[:, inp]
            hidd = self.nextHidden(hidd, ingot, leak)
            # del ingot
        return hidd

    def getLastState(self, chunk, hidden, w_in, leak):
        length = len(chunk)
        for inp in chunk:
            ingot = w_in[:, inp.unsqueeze(0)]
            hidden = self.nextHidden(hidden, ingot, leak)
        return hidden

    def getStates(self, chunk, w_in, leak):
        processed = dict()
        primechunk = chunk[0:8]
        chunk = chunk[8:len(chunk)]
        hidd = self.primeStates(primechunk, w_in, leak)
        length = len(chunk) - 1
        targets = torch.LongTensor(length)
        states = torch.FloatTensor(length, h_size)
        for i in range(length):
            targets[i] = chunk[i+1]
            ingot = w_in[:, chunk[i]]
            hidd = self.nextHidden(hidd, ingot, leak)
            states[i] = hidd
            del ingot
        del chunk
        processed["states"] = states
        processed["targets"] = targets
        return processed

class AutoEncoder(nn.Module):
    def __init__(self, h_size, o_size):
        super(AutoEncoder, self).__init__()
        self.h_size = h_size
        self.o_size = o_size
        self.encoder1 = nn.Sequential(
            nn.Linear(h_size, o_size, bias=False),
            # nn.Tanh()
            nn.PReLU()
        )
        # self.encoder2 = nn.Sequential(
        #     nn.Linear(o_size, 1024, bias=False),
        #     nn.Tanh()
        # )
        # self.encoder3 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Tanh()
        # )
        # self.decoder3 = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.Tanh()
        # )
        # self.decoder2 = nn.Sequential(
        #     nn.Linear(1024, o_size, bias=False),
        #     nn.Tanh()
        # )
        self.decoder1 = nn.Sequential(
            nn.Linear(o_size, h_size, bias=False),
        )

        for param in self.parameters():
            param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        val_range = (16.0/o_size)**0.5
        print(val_range)
        for p in self.parameters():
            # print(p, p.dim())
            p.data.uniform_(-val_range, val_range)

    def forward(self, x):
        encoded = self.encoder1(x)
        encoded = encoded / encoded.norm()
        # encoded = self.encoder2(encoded)
        # encoded = encoded / encoded.norm()
        # encoded = self.encoder3(encoded)
        # encoded = encoded / encoded.norm()
        # decoded = self.decoder3(encoded)
        # decoded = decoded / decoded.norm()
        # decoded = self.decoder2(decoded)
        # decoded = decoded / decoded.norm()
        # decoded = self.decoder1(decoded)
        # decoded = decoded / decoded.norm()
        return encoded

class GutenbergDataset(Dataset):

    def __init__(self, U1, test=False, bs=1, chunkfile='/home/user01/dev/language-model/saved/chunks192.9-271641.p', chunklen=192, primelen=9, n=2, stride=1, leakrate=0.4405):
        """
        Args:
            chunkfile (string): path to the pickled text sequences
            chunklen (int): length of each text sequence
        """

        self.test = test
        self.bs = bs
        self.chunkfile = chunkfile
        self.chunklen = chunklen
        self.n = n
        self.stride = stride
        self.leak = leakrate
        self.gramindex, self.glist = self.gramIndexes()
        self.U1 = U1
        # self.U1 = self.makeU(h_size, i_size)
        # self.chunklist = pickle.load(open(chunkfile, "rb"))
        # self.s1 = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
        # self.s2 = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
        # self.s3 = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
        self.trainchunks = pickle.load(open(chunkfile, "rb"))
        if test == False:
            self.trainchunks = self.trainchunks[0:244480]
        else:
            self.trainchunks = self.trainchunks[244480:271632]

        self.start = 0
        #
        # s1 = getBatch()
        # print(self.start)
        # s1.cuda()
        # s2 = getBatch()
        # print(self.start)
        # s2.cuda()
        # s3 = getBatch()
        # print(self.start)

    def __len__(self):
        return len(self.trainchunks)

    def __getitem__(self, idx):
        minibatch = self.getStates(self.trainchunks[idx], self.U1, self.leak)
        ministates = minibatch["states"]
        minitargets = minibatch["targets"]
        return ministates, minitargets

    def getU(self):
        return self.U1

    def getBatch(self):
        states = torch.Tensor(self.bs, self.chunklen, h_size).type(dtype)
        targets = torch.Tensor(self.bs, self.chunklen).type(torch.LongTensor)
        for i in range(self.start, self.bs):
            microbatch = self.getStates(self.trainchunks[i], self.U1, self.leak)
            states[i] = microbatch["states"]
            targets[i] = microbatch["targets"]
        self.start += bs
        return states, targets

    def makeU(self, h_size, i_size):
        U1 = torch.randn(h_size, i_size).type(torch.FloatTensor)
        U1 = self.normalize_u(U1)
        return U1

    # normalize input matrix U to have zero mean and unit length
    def normalize_u(self, u):
        n_columns = u.size(1)
        for col in range(n_columns):
            u[:, col] = u[:, col] - u[:, col].mean()
            u[:, col] = u[:, col] / u[:, col].norm()
        return u

    def rollCpu(self, hidd):
        lastidx = h_size - 1
        rollend = hidd.narrow(1, lastidx, 1)
        rollfront = hidd.narrow(1, 0, lastidx)
        hidd = torch.cat((rollend, rollfront), 1)
        return hidd

    def nextHidden(self, hidd, ingot, leak):
        hidd = (1.0 - leak) * hidd + leak * (ingot + self.rollCpu(hidd))
        hidd = hidd / hidd.norm()
        return hidd

    def primeStates(self, chunk, w_in, leak):
        ## set first hidden state directly as the input weight matrix's column
        # rather than using leaky combinations with an inital all-zeros state
        ## in the iniitial zeros case the resulting values for the new state
        # would be scaled down by the leak rate without a corresponding increase
        # from the previous states contribution since (1 - leak) * zeros and
        # roll(zeros) both = 0
        hidd = w_in[:, chunk[0]].unsqueeze(0)
        hidd = hidd / hidd.norm()
        for i in range(1, len(chunk)):
            ingot = w_in[:, chunk[i]]
            hidd = self.nextHidden(hidd, ingot, leak)
            del ingot
        return hidd

    def getStates(self, chunk, w_in, leak):
        processed = dict()
        primechunk = chunk[0:8]
        chunk = chunk[8:len(chunk)]
        hidd = self.primeStates(primechunk, w_in, leak)
        length = len(chunk) - 1
        targets = torch.LongTensor(length)
        states = torch.FloatTensor(length, h_size)
        for i in range(length):
            targets[i] = chunk[i+1]
            ingot = w_in[:, chunk[i]].unsqueeze(0)
            hidd = self.nextHidden(hidd, ingot, leak)
            states[i] = hidd
            del ingot
        del chunk
        processed["states"] = states
        processed["targets"] = targets
        return processed

    def gramIndexes(self):
        wv = KeyedVectors.load_word2vec_format('/home/user01/dev/wang2vec/embeddings-i3e4-ssg-neg15-s1024w6.txt', binary=False)
        temp = wv.index2word
        glist = np.array(temp[1:len(temp)])
        glist = [re.sub(r'_', ' ', j) for j in glist]
        gramindex = {gram:idx for idx, gram in enumerate(glist)}
        del wv
        return gramindex, glist

def trainm():
    if loadsaved:
        saved = torch.load(weightfile)
        U1 = saved["U1"]
        W1 = saved["W1"]
    weightfile = '/home/user01/dev/language-model/saved/weightsManual.' + str(h_size) + '.s0.4405.a0001.p'
    allweights = dict()

    traindata = GutenbergDataset()
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bs, shuffle=True, num_workers=3, drop_last=True, pin_memory=True)
    allweights["U1"] = traindata.getU()
    print("loaded: {} file: {}".format(loadsaved, weightfile))
    print("lr: {} leak: {} step: {} hidden: {} inp: {} SRU: {}x{}".format(lrate, leak, step, h_size, inp_size, layers, s_size))
    allweights = dict()
    W1 = Variable(torch.zeros(h_size, i_size).type(dtype), requires_grad=True)
    W1.cuda()
    optimizer = optim.Adam([W1], lr=lrate, betas=(0.9, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.99, weight_decay=0, nesterov=True)
    startp = time.perf_counter()
    batchloss = 0
    interval = 4096
    for ep in range(10):
        # global optimizer
        # start_batch_idx = (len(traindata) / 256) * ep
        for i, (states, targets) in enumerate(trainloader):
            states = states.permute(1, 0, 2)
            targets = targets.permute(1, 0)
            targets = targets.cuda(async=True)
            # print(states.size(), targets.size())
            statev = Variable(states.cuda(), requires_grad=False)
            targetv = Variable(targets, requires_grad=False)

            # print(statev.data.size())
            # if (i + 1) % 256 == 0:
            #     global_step = ((i + 1) / 256) + start_batch_idx
            #     batch_lr = lrate * sgdr(lr_period, global_step)
            #     optimizer = set_optimizer_lr(optimizer, batch_lr)
            optimizer.zero_grad()
            logits = statev.squeeze(1).mm(W1)
            outputs = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(outputs, targetv.squeeze(1))
            loss.backward()
            optimizer.step()
            batchloss += loss.data[0]

            if i > 0 and i % interval == 0:
                elapsedp = time.perf_counter() - startp
                tm, ts = divmod(elapsedp, 60)
                print(ep, i, batchloss / interval, "\t| {}m {}s \t| {:.5f}".format(int(tm), int(ts), lrate))
                batchloss = 0

        allweights["U1"] = traindata.getU()
        allweights["W1"] = W1
        allweights["opt"] = optimizer.state_dict()
        torch.save(allweights, weightfile)

def getBatch(start):
    states = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
    targets = torch.Tensor(bs, chunklen).type(torch.LongTensor)
    for i in range(start, bs):
        microbatch = getStates(trainchunks[i], U1, leak)
        states[i] = microbatch["states"]
        targets[i] = microbatch["targets"]
    start += bs
    return states, targets, start

def makeU(h_size, i_size):
    U1 = torch.randn(h_size, i_size).type(torch.FloatTensor)
    U1 = normalize_u(U1)
    return U1

# normalize input matrix U to have zero mean and unit length
def normalize_u(u):
    n_columns = u.size(1)
    for col in range(n_columns):
        u[:, col] = u[:, col] - u[:, col].mean()
        u[:, col] = u[:, col] / u[:, col].norm()
    return u

def rollCpu(hidd):
    lastidx = h_size - 1
    rollend = hidd.narrow(1, lastidx, 1)
    rollfront = hidd.narrow(1, 0, lastidx)
    hidd = torch.cat((rollend, rollfront), 1)
    return hidd

def nextHidden(hidd, ingot, leak):
    hidd = (1.0 - leak) * hidd + leak * (ingot + rollCpu(hidd))
    hidd = hidd / hidd.norm()
    return hidd

def primeStates(chunk, w_in, leak):
    ## set first hidden state directly as the input weight matrix's column
    # rather than using leaky combinations with an inital all-zeros state
    ## in the iniitial zeros case the resulting values for the new state
    # would be scaled down by the leak rate without a corresponding increase
    # from the previous states contribution since (1 - leak) * zeros and
    # roll(zeros) both = 0
    hidd = w_in[:, chunk[0]].unsqueeze(0)
    hidd = hidd / hidd.norm()
    for i in range(1, len(chunk)):
        ingot = w_in[:, chunk[i]]
        hidd = nextHidden(hidd, ingot, leak)
    return hidd

def getStates(chunk, w_in, leak):
    processed = dict()
    primechunk = chunk[0:8]
    chunk = chunk[8:len(chunk)]
    hidd = primeStates(primechunk, w_in, leak)
    length = len(chunk) - 1
    targets = torch.LongTensor(length)
    states = torch.FloatTensor(length, h_size)
    for i in range(length):
        targets[i] = chunk[i+1]
        ingot = w_in[:, chunk[i]].unsqueeze(0)
        hidd = nextHidden(hidd, ingot, leak)
        states[i] = hidd
    processed["states"] = states
    processed["targets"] = targets
    return processed

def gramIndexes():
    wv = KeyedVectors.load_word2vec_format('/home/user01/dev/wang2vec/embeddings-i3e4-ssg-neg15-s1024w6.txt', binary=False)
    temp = wv.index2word
    glist = np.array(temp[1:len(temp)])
    glist = [re.sub(r'_', ' ', j) for j in glist]
    gramindex = {gram:idx for idx, gram in enumerate(glist)}
    del wv
    return gramindex, glist

def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))

def getSomeStates(chunk, start, bsize, hidd, w_in, leak):
    # inputs = np.empty(length, dtype=np.int16)
    # ingots = torch.FloatTensor(b_size, i_size)
    processed = dict()
    targets = []
    states = torch.FloatTensor(b_size, h_size)
    for i in range(start, start + length):
        targets.append(chunk[i+1])
        ingot = w_in[:, chunk[i]]
        hidd = nextHidden(hidd, ingot, leak)
        states[i] = hidd
    processed["states"] = states
    processed["targets"] = targets
    return processed

# V = torch.eye(i_size).type(torch.LongTensor)
# U1 = torch.randn(h_size, i_size).type(dtype)
# U1 = normalize_u(U1)
# # W1 = Variable(torch.zeros(h_size, i_size).type(dtype), requires_grad=True)
#
# chunkfile = '/home/user01/dev/language-model/chunks256.p'
# chunklist = pickle.load(open(chunkfile, "rb"))
#
# trainchunks = []
# unknown = []
# for j in range(trainsize):
#     chunk = chunklist[j]
#     sgi = []
#     for idx in range(0, bs + 9, stride):
#         try:
#             sgi.append(gramindex[chunk[idx:idx + n]])
#         except:
#             unknown.append(chunk[idx:idx + n])
#     intchunk = torch.LongTensor(sgi)
#     if len(intchunk) >= 201:
#         trainchunks.append(intchunk)

# model = Hypervoir(i_size, h_size, leak)
# model = model.cuda()
# criterion = nn.NLLLoss()


# for param in model.parameters():
#     print(param.size())



# if trainsize == 768:
#     statefile = '/home/user01/dev/language-model/saved/states768.p'
#     targetfile = '/home/user01/dev/language-model/saved/targets768.p'
#     states = torch.from_numpy(pickle.load(open(statefile, "rb")))
#     targets = torch.from_numpy(pickle.load(open(targetfile, "rb")))
# else:
#
# npstates = states.numpy()
# npstates = targets.numpy()

# pickle.dump(npstates, open(targetfile, "wb"))
# traindata = GutenbergDataset()
# testdata = GutenbergDataset(test=True)
# trainloader = torch.utils.data.DataLoader(traindata, batch_size=bs, shuffle=True, num_workers=4)
# testloader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
def getBatch(trainchunks, U1, states, targets, start):
    for i in range(start, start + bs):
        microbatch = getStates(trainchunks[i], U1, leak)
        states[i - start] = microbatch["states"]
        targets[i - start] = microbatch["targets"]
    start += bs
    return states, targets, start


def train():

    gramindex, glist = gramIndexes()
    # chunklist = pickle.load(open(chunkfile, "rb"))
    U1 = makeU(h_size, i_size)
    # s1 = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
    # s2 = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
    # s3 = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
    chunkfile='/home/user01/dev/language-model/saved/chunks192.9-271641.p'
    trainchunks = pickle.load(open(chunkfile, "rb"))
    testchunks = trainchunks[244480:271641]
    trainchunks = trainchunks[0:244480]
    start = 0


    if loadsaved:
        saved = torch.load(weightfile)
        U1 = saved["U1"]
        W1 = saved["W1"]
    # weightfile = '/home/user01/dev/language-model/saved/weightsSRU.2x512.' + str(h_size) + '.p10.s0.4405.a001.p'
    allweights = dict()

    # optimizer = optim.SGD([{'params': [W1]}], lr=lrate, momentum=0.995, weight_decay=0, nesterov=True)
    # optimizer = optim.SGD([W1], lr=lrate, momentum=0.998, weight_decay=0, nesterov=True)
    model = Model(h_size, s_size, layers=layers)
    model.cuda()

    states = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
    targets = torch.Tensor(bs, chunklen).type(torch.LongTensor)
    newstates = torch.Tensor(bs, chunklen, h_size).type(torch.FloatTensor)
    newtargets = torch.Tensor(bs, chunklen).type(torch.LongTensor)

    states, targets, start = getBatch(trainchunks, U1, states, targets, start)
    states.contiguous().pin_memory()
    targets.contiguous().pin_memory()
    print(start)

    print("loaded: {} file: {}".format(loadsaved, weightfile))
    print("lr: {} leak: {} step: {} hidden: {}".format(lrate, leak, step, h_size))

    weightfile = '/home/user01/dev/language-model/saved/weightsSRU.2x512.' + str(h_size) + '.p10.s0.4405.a001.p'
    allweights = dict()




    flag = 0
    for ep in range(10):
        global optimizer
        shuffle(trainchunks)
        start_batch_idx = bs * (ep)
        train_loss = 0
        eploss = 0
        for i in range(1, 382):
            batchloss = 0
            if flag == 0:
                cstates = Variable(states, requires_grad=False).cuda()
                ctargets = Variable(targets, requires_grad=False).cuda(async=True)
                newstates, newtargets, start = getBatch(trainchunks, U1, newstates, newtargets, start)
                newstates.contiguous().pin_memory()
                newtargets.contiguous().pin_memory()
                flag = 1
            else:
                cstates = Variable(newstates, requires_grad=False).cuda()
                ctargets = Variable(newtargets, requires_grad=False).cuda(async=True)
                states, targets, start = getBatch(trainchunks, U1, states, targets, start)
                flag = 0

            global_step = i + start_batch_idx
            batch_lr = lrate * sgdr(lr_period, global_step)
            optimizer = set_optimizer_lr(optimizer, batch_lr)
            for j in range(chunklen):
                optimizer.zero_grad()
                output = model(cstates[j])
                loss = criterion(output, ctargets[j])
                loss.backward()
                optimizer.step()
                batchloss += loss.data[0]

            if i % 64 == 0:
                elapsedp = time.perf_counter() - startp
                tm, ts = divmod(elapsedp, 60)
                print(ep, i, batchloss / bs, "| {}m {}s | {:.5f}".format(int(tm), int(ts), batch_lr))

            # progress_bar(batch_idx, len(trainchunks), 'Loss: %.3f | LR: %.5f'
            #     % (eploss/(batch_idx+1), batch_lr))
        # eperr = (eperr / count) * 100
        # eploss = (eploss / len(trainchunks))
            # if (batch_idx+1) % 128 == 0:
            #     elapsedp = time.perf_counter() - startp
            #     tm, ts = divmod(elapsedp, 60)
            #     print(ep, eploss / (batch_idx + 1), "| {}m {}s | {:.5f}".format(int(tm), int(ts), batch_lr))

        allweights["U1"] = U1
        allweights["SRU"] = model.state_dict()
        allweights["opt"] = optimizer.state_dict()
        torch.save(allweights, weightfile)


# train()
# trainset()
# trainlin()
trainSO()
