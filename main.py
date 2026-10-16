import numpy as np
from algorithm.selection import conformal_labeling
from algorithm.preprocess import get_data
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--model", type=str, default="resnet34")
parser.add_argument("--n_calib", type=int, default=1000, help="calibration set size")
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="Target FLR level")
args = parser.parse_args()

dataset = args.dataset

Y, Yhat, confidence = get_data(args)

alpha = args.alpha
num_trials = args.num_trials
flr_list = []
power_list = []
selection_size_list = []

for j in tqdm(range(num_trials)):
    n_samples = len(Y)
    n_calib = args.n_calib
    n_test = n_samples - n_calib
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

    flp, power, selection_size = conformal_labeling(Y, Yhat, confidence, cal_indices, alpha, random=args.random)
    flr_list.append(flp)
    power_list.append(power)
    selection_size_list.append(selection_size)


print(f"Realized FLR: {np.mean(flr_list) * 100}%")
print(f"Power: {np.mean(power_list) * 100}%")
print(f"Selection Ratio: {np.mean(selection_size_list) / n_test * 100}%")
print(f"AI only Error: {100 - np.sum(Yhat==Y) / len(Y) * 100}%")
