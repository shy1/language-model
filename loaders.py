import torch
import numpy as np
from torch.utils.data import Dataset
import pickle

class MooreMnistNoESN(Dataset):
    def __init__(self, U1, h_size, tvt='train', bs=1, lastonly=False, chunkfile='../data/mooreQ4_mnist_tvt.p', chunklen=112, leakrate=0.4405):

        torch.manual_seed(481639)
        self.bs = bs
        self.tvt = tvt
        self.lastonly = lastonly
        self.h_size = h_size
        self.chunkfile = chunkfile
        self.chunklen = chunklen
        self.leak = leakrate
        self.U1 = U1.numpy()
        # print("U1:", self.U1.shape)
        # chunkfile='../data/bin_mnist/vscan_binmist_tvt.p'
        posefile = '../data/moore_posed.p'
        splits = pickle.load(open(chunkfile, "rb"))
        posed = pickle.load(open(chunkfile, "rb"))
        if self.tvt == 'train':
            self.pixelchunks = splits['x_train'].reshape(splits['x_train'].shape[0], splits['x_train'].shape[1], 1).astype(np.int64)
            self.posedchunks = posed['x_train'].reshape(posed['x_train'].shape[0], posed['x_train'].shape[1], 1).astype(np.int64)
            self.targets = splits['y_train']
        elif self.tvt == 'val':
            self.pixelchunks = splits['x_val'].reshape(splits['x_val'].shape[0], splits['x_val'].shape[1], 1).astype(np.int64)
            self.posedchunks = posed['x_val'].reshape(posed['x_val'].shape[0], posed['x_val'].shape[1], 1).astype(np.int64)
            self.targets = splits['y_val']
        elif self.tvt == 'test':
            self.pixelchunks = splits['x_test'].reshape(splits['x_test'].shape[0], splits['x_test'].shape[1], 1).astype(np.int64)
            self.posedchunks = posed['x_test'].reshape(posed['x_test'].shape[0], posed['x_test'].shape[1], 1).astype(np.int64)
            self.targets = splits['y_test']
        else:
            self.pixelchunks = splits['x_train'].reshape(splits['x_train'].shape[0], splits['x_train'].shape[1], 1).astype(np.int64)
            self.posedchunks = posed['x_train'].reshape(posed['x_train'].shape[0], posed['x_train'].shape[1], 1).astype(np.int64)
            self.targets = splits['y_train']

        self.pixelchunks = torch.from_numpy(self.pixelchunks)
        self.posedchunks = torch.from_numpy(self.posedchunks)
        self.targets = torch.from_numpy(self.targets)
        # self.targetsc = torch.cuda.LongTensor(self.targets)
        # print(type(self.pixelchunks), self.pixelchunks.dtype)
        # print(type(self.targets), self.targets.dtype)

    def __len__(self):
        return len(self.pixelchunks)

    def __getitem__(self, idx):
        allsteps = torch.zeros(self.chunklen, 256).type(torch.FloatTensor)
        indexes = self.pixelchunks[idx]
        for i in range(self.chunklen):
            tensor = torch.zeros(1, 256).type(torch.FloatTensor)
            tensor[0][indexes[i]] = 1
            allsteps[i] = tensor

        posedsteps = torch.zeros(self.chunklen, 256).type(torch.FloatTensor)
        indexes = self.posedchunks[idx]
        for i in range(self.chunklen):
            tensor = torch.zeros(1, 128).type(torch.FloatTensor)
            tensor[0][indexes[i]] = 1
            posedsteps[i] = tensor
        return allsteps, posedsteps, self.targets[idx]

    def getTargets(self):
        return self.targets

class MooreMnistSRU(Dataset):
    def __init__(self, U1, h_size, tvt='train', bs=1, lastonly=False, chunkfile='../data/mooreQ4_mnist_tvt.p', chunklen=256, leakrate=0.4405):

        torch.manual_seed(481639)
        self.bs = bs
        self.tvt = tvt
        self.lastonly = lastonly
        self.h_size = h_size
        self.chunkfile = chunkfile
        self.chunklen = chunklen
        self.leak = leakrate
        self.U1 = U1.numpy()
        # print("U1:", self.U1.shape)

        splits = pickle.load(open(chunkfile, "rb"))
        if self.tvt == 'train':
            self.pixelchunks = splits['x_train'].reshape(splits['x_train'].shape[0], splits['x_train'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_train']
        elif self.tvt == 'val':
            self.pixelchunks = splits['x_val'].reshape(splits['x_val'].shape[0], splits['x_val'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_val']
        elif self.tvt == 'test':
            self.pixelchunks = splits['x_test'].reshape(splits['x_test'].shape[0], splits['x_test'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_test']
        else:
            self.pixelchunks = splits['x_train'].reshape(splits['x_train'].shape[0], splits['x_train'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_train']

        self.targets = torch.from_numpy(self.targets)
        # self.targetsc = torch.cuda.LongTensor(self.targets)
        # print(type(self.pixelchunks), self.pixelchunks.dtype)
        # print(type(self.targets), self.targets.dtype)

    def __len__(self):
        return len(self.pixelchunks)

    def __getitem__(self, idx):
        # minibatch = self.getStates(self.pixelchunks[idx], self.U1, self.leak)
        minibatch = self.getStates(self.pixelchunks[idx], self.targets[idx], self.U1, self.leak)
        ministates = minibatch["states"]
        minitargets = minibatch["targets"]
        if self.lastonly == True:
            lastidx = len(ministates) - 1
            ministates = ministates.narrow(0, lastidx, 1)
            minitargets = minitargets.narrow(0, lastidx, 1)
        return ministates, minitargets

    def nextHidden(self, hidd, ingot, leak):
        hidd = (1.0 - leak) * hidd + leak * (ingot + np.roll(hidd, 1))
        hidd = hidd / np.linalg.norm(hidd)
        return hidd

    def primeStates(self, chunk, w_in, leak):
        ## run through the sequence in the reverse direction prior to the forward
        #  pass in order prime the hidden states and avoid the negative effects
        # of initial reservoir transience
        ## set first hidden state directly as the input weight matrix's column
        # rather than using leaky combinations with an inital all-zeros state
        last = len(chunk) - 1
        # print(type(chunk[last]))
        # hidd = np.zeros(self.h_size, dtype=np.float32)
        hidd = w_in[:, chunk[last]].reshape(w_in[:, chunk[last]].shape[0])
        hidd = hidd / np.linalg.norm(hidd)
        for inp in reversed(chunk[1:last]):
            ingot = w_in[:, inp].reshape(w_in[:, inp].shape[0])
            hidd = self.nextHidden(hidd, ingot, leak)
        # for inp in range(len(chunk)):
        #     ingot = w_in[:, inp]
        #     hidd = self.nextHidden(hidd, ingot, leak)
        return hidd

    def getStates(self, chunk, target, w_in, leak):
        processed = dict()
        # print(type(chunk[0]))
        length = len(chunk)
        last = length - 1
        states = torch.FloatTensor(length, self.h_size)
        targets = torch.LongTensor(length, 1)
        hidd = self.primeStates(chunk, w_in, leak)

        for i in range(length):
            ingot = w_in[:, chunk[i]].reshape(w_in[:, chunk[i]].shape[0])
            hidd = self.nextHidden(hidd, ingot, leak)
            states[i] = torch.from_numpy(hidd)
            # targets[i] = target
            # targets[i] = chunk[i+1]

        # ingot = w_in[:, chunk[last]]
        # hidd = self.nextHidden(hidd, ingot, leak)
        # states[last] = torch.from_numpy(hidd)
        # targets[last] = target
        processed["states"] = states
        processed["targets"] = target
        return processed

    def getTargets(self):
        return self.targets

class MooreMnist(Dataset):
    def __init__(self, U1, h_size, tvt='train', bs=1, lastonly=False, chunkfile='../data/mooreQ4_mnist_tvt.p', chunklen=256, leakrate=0.4405):

        torch.manual_seed(481639)
        self.bs = bs
        self.tvt = tvt
        self.lastonly = lastonly
        self.h_size = h_size
        self.chunkfile = chunkfile
        self.chunklen = chunklen
        self.leak = leakrate
        self.U1 = U1.numpy()
        # print("U1:", self.U1.shape)

        splits = pickle.load(open(chunkfile, "rb"))
        if self.tvt == 'train':
            self.pixelchunks = splits['x_train'].reshape(splits['x_train'].shape[0], splits['x_train'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_train']
        elif self.tvt == 'val':
            self.pixelchunks = splits['x_val'].reshape(splits['x_val'].shape[0], splits['x_val'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_val']
        elif self.tvt == 'test':
            self.pixelchunks = splits['x_test'].reshape(splits['x_test'].shape[0], splits['x_test'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_test']
        else:
            self.pixelchunks = splits['x_train'].reshape(splits['x_train'].shape[0], splits['x_train'].shape[1], 1).astype(np.int32)
            self.targets = splits['y_train']

        self.targets = torch.from_numpy(self.targets)
        # self.targetsc = torch.cuda.LongTensor(self.targets)
        # print(type(self.pixelchunks), self.pixelchunks.dtype)
        # print(type(self.targets), self.targets.dtype)

    def __len__(self):
        return len(self.pixelchunks)

    def __getitem__(self, idx):
        # minibatch = self.getStates(self.pixelchunks[idx], self.U1, self.leak)
        minibatch = self.getStates(self.pixelchunks[idx], self.targets[idx], self.U1, self.leak)
        ministates = minibatch["states"]
        minitargets = minibatch["targets"]
        if self.lastonly == True:
            lastidx = len(ministates) - 1
            ministates = ministates.narrow(0, lastidx, 1)
            minitargets = minitargets.narrow(0, lastidx, 1)
        return ministates, minitargets

    def nextHidden(self, hidd, ingot, leak):
        # print("n1:", hidd.shape)
        hidd = (1.0 - leak) * hidd + leak * (ingot + np.roll(hidd, 1))
        # print("n2:", hidd.shape)
        hidd = hidd / np.linalg.norm(hidd)
        # print("n3:", hidd.shape)
        return hidd

    def primeStates(self, chunk, w_in, leak):
        ## run through the sequence in the reverse direction prior to the forward
        #  pass in order prime the hidden states and avoid the negative effects
        # of initial reservoir transience
        ## set first hidden state directly as the input weight matrix's column
        # rather than using leaky combinations with an inital all-zeros state
        last = len(chunk) - 1
        # print(type(chunk[last]))
        hidd = np.zeros(self.h_size, dtype=np.float32)
        # hidd = w_in[:, chunk[last]]
        # hidd = hidd.reshape(hidd.shape[0])
        # print("a:", hidd.shape)
        # hidd = hidd / np.linalg.norm(hidd)
        # print("b:", hidd.shape)
        for j in range(2):
            for inp in range(len(chunk)):
                ingot = w_in[:, inp]
                hidd = self.nextHidden(hidd, ingot, leak)
        # print("c:", hidd.shape)
        return hidd

    def getStates(self, chunk, target, w_in, leak):
        processed = dict()
        # print(type(chunk[0]))
        length = len(chunk)
        last = length - 1
        states = torch.FloatTensor(length, self.h_size)
        targets = torch.LongTensor(length, 1)
        hidd = self.primeStates(chunk, w_in, leak)


        for i in range(last):
            ingot = w_in[:, chunk[i]].reshape(w_in[:, chunk[i]].shape[0])
            # print("p:", hidd.shape, ingot.shape)
            # hidd = hidd.reshape(hidd.shape[0])
            hidd = self.nextHidden(hidd, ingot, leak)
            states[i] = torch.from_numpy(hidd)
            targets[i] = torch.from_numpy(chunk[i+1])
            # targets[i] = chunk[i+1]
        # print("a:", hidd.shape)
        ingot = w_in[:, chunk[last]].reshape(w_in[:, chunk[last]].shape[0])
        hidd = self.nextHidden(hidd, ingot, leak)
        # print("b:", hidd.shape)
        states[last] = torch.from_numpy(hidd)
        targets[last] = target
        processed["states"] = states
        processed["targets"] = targets
        return processed

    def getTargets(self):
        return self.targets

class BinaryMnist(Dataset):
    def __init__(self, U1, h_size, tvt='train', bs=1, lastonly=False, chunkfile='../data/bin_mnist/vscan_binmist_tvt.p', chunklen=112, leakrate=0.4405):

        torch.manual_seed(481639)
        self.bs = bs
        self.tvt = tvt
        self.lastonly = lastonly
        self.h_size = h_size
        self.chunkfile = chunkfile
        self.chunklen = chunklen
        self.leak = leakrate
        self.U1 = U1.numpy()
        # print("U1:", self.U1.shape)

        splits = pickle.load(open(chunkfile, "rb"))
        if self.tvt == 'train':
            self.pixelchunks = splits['x_train']
            self.targets = splits['y_train']
        elif self.tvt == 'val':
            self.pixelchunks = splits['x_val']
            self.targets = splits['y_val']
        elif self.tvt == 'test':
            self.pixelchunks = splits['x_test']
            self.targets = splits['y_test']
        else:
            self.pixelchunks = splits['x_train']
            self.targets = splits['y_train']

        self.targets = torch.from_numpy(self.targets)
        # self.targetsc = torch.cuda.LongTensor(self.targets)
        # print(type(self.pixelchunks), self.pixelchunks.dtype)
        # print(type(self.targets), self.targets.dtype)

    def __len__(self):
        return len(self.pixelchunks)

    def __getitem__(self, idx):
        # minibatch = self.getStates(self.pixelchunks[idx], self.U1, self.leak)
        minibatch = self.getStates(self.pixelchunks[idx], self.targets[idx], self.U1, self.leak)
        ministates = minibatch["states"]
        minitargets = minibatch["targets"]
        if self.lastonly == True:
            lastidx = len(ministates) - 1
            ministates = ministates.narrow(0, lastidx, 1)
            minitargets = minitargets.narrow(0, lastidx, 1)
        return ministates, minitargets

    def nextHidden(self, hidd, ingot, leak):
        hidd = (1.0 - leak) * hidd + leak * (ingot + np.roll(hidd, 1))
        hidd = hidd / np.linalg.norm(hidd)
        return hidd

    def primeStates(self, chunk, w_in, leak):
        ## run through the sequence in the reverse direction prior to the forward
        #  pass in order prime the hidden states and avoid the negative effects
        # of initial reservoir transience
        ## set first hidden state directly as the input weight matrix's column
        # rather than using leaky combinations with an inital all-zeros state
        last = len(chunk) - 1
        # print(type(chunk[last][0]))
        hidd = w_in[:, chunk[last]]
        hidd = hidd / np.linalg.norm(hidd)
        for inp in reversed(chunk[1:last]):
            ingot = w_in[:, inp]
            hidd = self.nextHidden(hidd, ingot, leak)
        return hidd

    def getStates(self, chunk, target, w_in, leak):
        processed = dict()
        length = len(chunk)
        last = length - 1
        states = torch.FloatTensor(length, self.h_size)
        targets = torch.LongTensor(length, 1)
        hidd = self.primeStates(chunk, w_in, leak)

        for i in range(last):
            ingot = w_in[:, chunk[i]]
            hidd = self.nextHidden(hidd, ingot, leak)
            states[i] = torch.from_numpy(hidd)
            targets[i] = torch.from_numpy(chunk[i+1])

        ingot = w_in[:, chunk[last]]
        hidd = self.nextHidden(hidd, ingot, leak)
        states[last] = torch.from_numpy(hidd)
        targets[last] = target
        processed["states"] = states
        processed["targets"] = targets
        return processed

    def getTargets(self):
        return self.targets

class GutenbergDataset(Dataset):
    ## TODO: see if using standard numpy arrays and functions like np.roll()
    #  and only converting to pytorch tensors at the very end will improve performance

    def __init__(self, U1, test=False, bs=1, chunkfile='/home/user01/dev/language-model/saved/chunks192.9-271641.p', chunklen=192, primelen=9, n=2, stride=1, leakrate=0.4405):
        """
        Args:
            chunkfile (string): path to the pickled text sequences
            chunklen (int): length of each text sequence
        """
        torch.manual_seed(481639)
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

# def makeU(h_size, i_size):
#     torch.manual_seed(481639)
#     U1 = torch.randn(h_size, i_size).type(torch.FloatTensor)
#     U1 = normalize_u(U1)
#     return U1
#
# # normalize input matrix U to have zero mean and unit length
# def normalize_u(u):
#     n_columns = u.size(1)
#     for col in range(n_columns):
#         u[:, col] = u[:, col] - u[:, col].mean()
#         u[:, col] = u[:, col] / u[:, col].norm()
#     return u
#
# U1 = makeU(512, 128)
# dl = BinaryMnist(U1)
#
# s = dl[0]
#
# print("s:", s)
# print("t:", t)
