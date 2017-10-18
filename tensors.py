import torch

def shiftcuda(matrix):
    length = len(matrix)
    end = torch.cuda.FloatTensor([matrix[length - 1]])
    matrix = matrix.narrow(0, 0, length - 1)
    print(end)
    print(matrix)
    matrix = torch.cat((end, matrix), 0)
    return matrix

tx = torch.cuda.FloatTensor([0,1,2,3,4,5])
print(tx)
tx = shiftcuda(tx)
print(tx)
