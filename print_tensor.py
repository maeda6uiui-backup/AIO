import argparse
import torch
import numpy as np

def print_tensor(filepath):
    tensor=torch.load(filepath)
    print(tensor.detach().cpu().numpy()[0])

if __name__=="__main__":
    np.set_printoptions(threshold=np.inf)

    parser=argparse.ArgumentParser(description="AIO")
    parser.add_argument("--filepath",type=str)
    args=parser.parse_args()

    print_tensor(args.filepath)
