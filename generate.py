# https://github.com/spro/practical-pytorch

import torch

from helpers import *
from model import *

def generate(decoder, prime_str='a', predict_len=100, temperature=0.667):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)

    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

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
    argparser.add_argument('-t', '--temperature', type=float, default=0.667)
    args = argparser.parse_args()

    all_characters = string.ascii_lowercase + string.digits + string.punctuation + " "
    n_characters = len(all_characters)

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))
