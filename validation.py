import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

def tars_fars(model, dataloader, device):
    pairs_dist = calculate_pair_distances(model, dataloader, device)
    max_dist = max(pairs_dist)
    labels = extract_labels(dataloader)
    total_positives = sum(labels)
    total_negatives = len(labels) - total_positives 
    FARs = TARs = []
    for threshold in np.arange(0, max_dist, max_dist/100):  
        true_positives = false_positives =  0
        for pair_dist, label in zip(pairs_dist, labels):  
            same_person = pair_dist < threshold
            if same_person:
                if label:
                    true_positives += 1
                else:
                    false_positives += 1
        FAR = false_positives / total_negatives
        TAR = true_positives / total_positives
        FARs.append(FAR.item())
        TARs.append(TAR.item())
    return (TARs, FARs)
    
def print_ROC(TARs, FARs):
    plt.plot(FARs, TARs)
    plt.show()

def verify(model, dataloader, device, FAR_limit = 10e-6):
    TAR = 0
    pairs_dist = calculate_pair_distances(model, dataloader, device)
    max_dist = max(pairs_dist)
    labels = extract_labels(dataloader)
    total_positives = sum(labels)
    total_negatives = len(labels) - total_positives
    for threshold in np.arange(0, max_dist, max_dist/100):  
        true_positives = false_positives =  0
        for pair_dist, label in zip(pairs_dist, labels):  
            same_person = pair_dist < threshold
            if same_person:
                if label:
                    true_positives += 1
                else:
                    false_positives += 1
        FAR = false_positives / total_negatives
        if FAR > FAR_limit:
            return TAR.item()
        TAR = true_positives / total_positives
    return TAR.item()

def validate(model, dataloader, device):   
    correct = total = 0 
    model.eval()
    with torch.no_grad():    
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            correct += predicted.eq(labels.data).cpu().sum()
            total += labels.size(0)
    return (correct / total).item()

def calculate_pair_distances(model, dataloader, device):
    model.eval()
    pairs_dist = []
    with torch.no_grad():    
        for img_1, img_2, _ in dataloader:
            emb_1 = model.encode(img_1.to(device)).cpu()
            emb_2 = model.encode(img_2.to(device)).cpu()
            #pairs_dist.append(np.linalg.norm(emb_1 - emb_2)) - euklidovská vzdálenost
            pairs_dist.append(distance.cosine(emb_1, emb_2))
    return pairs_dist

def extract_labels(dataloader):
    labels = []
    for _, _, label in dataloader:
        labels.append(label)
    return labels
