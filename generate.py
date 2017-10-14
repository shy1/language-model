# https://github.com/spro/practical-pytorch

import torch
import string
import torch.nn.functional as F

from helpers import *
from model import *
from tensorboardX import SummaryWriter


def generate(decoder, prime_str='a', predict_len=100, temperature=0.8):
    #writer = SummaryWriter('runs/gen1')
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)

    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        #print(output.data)
        #softo = F.softmax(output)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        #output_dist2 = softo.data.view(-1).div(temperature).exp()
        #output_dist3 = softo.data.view(-1)
        #print(softo.data)
        #print(output_dist2)
        #print(torch.log(output_dist3))

        top_3 = torch.multinomial(output_dist, 3)
        top_i = top_3[0]
        temp1 = str(all_characters[top_3[0]]) + " - " + str(output_dist[top_3[0]]) + "\n"
        temp2 = str(all_characters[top_3[1]]) + " - " + str(output_dist[top_3[1]]) + "\n"
        temp3 = str(all_characters[top_3[2]]) + " - " + str(output_dist[top_3[2]]) + "\n"
        # for i in range(0,3):
        #     tempo = output_dist[top_3[i]]
        #     tempc = all_characters[top_3[i]]
        #     temps = str(tempc) + " - " + str(tempo)
        #     print(temps)
        print(temp1 + temp2 + temp3 + "\n")
        #print(top_3)
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted

if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='a')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    args = argparser.parse_args()

    all_characters = string.ascii_lowercase + string.digits + string.punctuation + " "
    n_characters = len(all_characters)

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))
