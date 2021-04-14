import torch
import numpy as np
import matplotlib.pyplot as plt

def validate(model, lfw_pairs, lfw_labels, device):
    model.eval()
    with torch.no_grad():
        FARs = TARs = []
        pairs_dist = np.zeros(len(lfw_pairs))
        for i, pair in enumerate(lfw_pairs):
            emb_1, emb_2 = model.encode(pair.to(device)).cpu()
            #pairs_dist[i] = distance.euclidean(emb_1, emb_2)
            #pairs_dist[i] = distance.cosine(emb_1, emb_2)
            pairs_dist[i] = np.linalg.norm(emb_1 - emb_2)

      #print(pairs_dist.shape)

        for threshold in range(0, 10**16, 10**14):   #TODO
            true_positives = false_positives =  0
            for pair_dist, label in zip(pairs_dist, lfw_labels):  
                #print(pair_dist)
                same_person = pair_dist < threshold
                if same_person:
                    if label:
                        true_positives += 1
                    else:
                        false_positives += 1
            FAR = false_positives / 3000
            TAR = true_positives / 3000
            FARs.append(FAR)
            TARs.append(TAR)
        print("fars", FARs)
        print("tars", TARs)
        # plt.plot(FARs, TARs)
        # plt.show()