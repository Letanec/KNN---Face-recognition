import torch
import numpy as np
import matplotlib.pyplot as plt

total_positives = total_negatives = 3000
FAR_limit = 10e-6
max_dist = 10**16
step = 10**14

def print_ROC(model, lfw_pairs, lfw_labels, device):
    pairs_dist = calculatePairDistances(model, lfw_pairs, device)
    FARs = TARs = []
    for threshold in range(0, max_dist, step):   #TODO
        true_positives = false_positives =  0
        for pair_dist, label in zip(pairs_dist, lfw_labels):  
            #print(pair_dist)
            same_person = pair_dist < threshold
            if same_person:
                if label:
                    true_positives += 1
                else:
                    false_positives += 1
        FAR = false_positives / total_negatives
        TAR = true_positives / total_positives
        
        FARs.append(FAR)
        TARs.append(TAR)
    print("fars", FARs)
    print("tars", TARs)
    plt.plot(FARs, TARs)
    plt.show()

def validate(model, lfw_pairs, lfw_labels, device, time_stamp):
    TAR = 0
    pairs_dist = calculatePairDistances(model, lfw_pairs, device)

    for threshold in range(0, max_dist, step):   #TODO
        true_positives = false_positives =  0
        for pair_dist, label in zip(pairs_dist, lfw_labels):  
            #print(pair_dist)
            same_person = pair_dist < threshold
            if same_person:
                if label:
                    true_positives += 1
                else:
                    false_positives += 1
        FAR = false_positives / total_negatives
        if FAR > FAR_limit:
            return TAR
        TAR = true_positives / total_positives
        with open(f"./outputs/all_TARs_and_FARs_{time_stamp}.out", 'a') as f:
            f.write(f'{TAR}\n')
        with open(f"./outputs/test_accs{time_stamp}.out", 'a') as f:
            f.write(f'{TAR}, {FAR}\n')

    return TAR;

def calculatePairDistances(model, lfw_pairs, device):
    model.eval()
    with torch.no_grad():
        pairs_dist = np.zeros(len(lfw_pairs))
        for i, pair in enumerate(lfw_pairs):
            emb_1, emb_2 = model.encode(pair.to(device)).cpu()
            #pairs_dist[i] = distance.euclidean(emb_1, emb_2)
            #pairs_dist[i] = distance.cosine(emb_1, emb_2)
            pairs_dist[i] = np.linalg.norm(emb_1 - emb_2)
    return pairs_dist
