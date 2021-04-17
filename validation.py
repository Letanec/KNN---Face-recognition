import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def tars_fars(model, dataloader, device):
    tot_TAR = 0
    tot_FAR = 0
    total_positives = 0
    total_negatives = 0
    FARs = TARs = []
    for img1, img2, labels in dataloader:
        pairs_dist = calculate_pair_distances(model, img1, img2, device)
        max_dist = max(pairs_dist)
        
        positives = sum(labels)
        negatives = len(labels) - positives
        total_positives += positives
        total_negatives += negatives

        for threshold in np.arange(0, max_dist, max_dist/100):  
            true_positives = false_positives =  0
            for pair_dist, label in zip(pairs_dist, labels):  
                same_person = pair_dist < threshold
                if same_person:
                    if label:
                        true_positives += 1
                    else:
                        false_positives += 1
            FAR = false_positives / negatives
            TAR = true_positives / positives
            FARs.append(FAR.item())
            TARs.append(TAR.item())
    return (TARs, FARs)

def verify(model, dataloader, device, FAR_limit = 10e-6):
    tot_TAR = 0
    tot_FAR = 0
    total_positives = 0
    total_negatives = 0
    for img1, img2, labels in dataloader:
        pairs_dist = calculate_pair_distances(model, img1, img2, device)
        max_dist = max(pairs_dist)
        positives = sum(labels)
        negatives = len(labels) - positives
        total_positives += positives
        total_negatives += negatives
        for threshold in np.arange(0, max_dist, max_dist/100):  
            true_positives = false_positives =  0
            for pair_dist, label in zip(pairs_dist, labels):  
                same_person = pair_dist < threshold
                if same_person:
                    if label:
                        true_positives += 1
                    else:
                        false_positives += 1
            FAR = false_positives / negatives
            tot_FAR += false_positives
            if FAR > FAR_limit:
                break
            tot_TAR += true_positives
            TAR = true_positives / positives
    tot_TAR = tot_TAR / total_positives
    return tot_TAR.item()
    
def print_ROC(TARs, FARs):
    plt.plot(FARs, TARs)
    plt.show()

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

def calculate_pair_distances(model, img_1, img_2, device):
    model.eval()
    pairs_dist = []
    with torch.no_grad():    
        emb_1 = model.encode(img_1.to(device)).cpu()
        emb_2 = model.encode(img_2.to(device)).cpu()
        pairs_dist = np.linalg.norm(emb_1 - emb_2, axis=1)
    return pairs_dist

def extract_labels(dataloader):
    labels = []
    for _, _, label in dataloader:
        labels.append(label)
    return labels
