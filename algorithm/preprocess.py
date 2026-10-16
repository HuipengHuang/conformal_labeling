import pandas as pd
import numpy as np
import json


def get_data(args):
    model, dataset = args.model, args.dataset
    if dataset == "imagenet" or dataset == "mmlu":
        data = pd.read_csv(f"./datasets/{model}_{dataset}.csv")
        Y = data["Y"].to_numpy()
        confidence = data["confidence"].to_numpy()
        Yhat = data["Yhat"].to_numpy()
    elif dataset == "math-l5" or "zebra_logic" or "humaneval+":
        Y, Yhat, confidence = get_llm_data(model, dataset)
    else:
        raise NotImplementedError
    return Y, Yhat, confidence



def get_llm_data(model, dataset):
    file_path = f"./datasets/{model}_{dataset}.jsonl"
    # file_path = f"./data/{args.model}-with-group-confidence.jsonl"
    # file_path = f"./data/thinking_gpqa_diamond_more_confidence.jsonl"
    with open(file_path, "r") as f:
        results = [json.loads(line) for line in f]

    Yhat = []
    Y = []
    bottom10_confidence_list = []

    for i in range(len(results)):
        dit = results[i]
        Yhat.append(dit["is_correct"])
        Y.append(True)
        bottom10_confidence_list.append(dit["bottom10_confidence"])

    Yhat = np.array(Yhat)
    Y = np.array(Y)
    confidence = np.array(bottom10_confidence_list)
    return Y, Yhat, confidence