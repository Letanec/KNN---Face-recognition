import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import fixed_image_standardization
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.model_selection import KFold
import math
import os
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# LFW evaluace -- prevzato z https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb
class Lfw_evaluation:
    def __init__(self, lfw_root, lfw_pairs, device):
        self.lfw_root = lfw_root
        self.lfw_pairs = lfw_pairs
        self.device = device 
        trans = transforms.Compose([ 
            transforms.Resize(160),   
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization 
        ])
        ds = datasets.ImageFolder(lfw_root, transform=trans)
        ds.samples = [(p, p) for p, _ in ds.samples]
        self.loader = DataLoader(
            ds,
            batch_size=32,
            sampler=SequentialSampler(ds)
        )
        pairs = self._read_pairs(lfw_pairs)
        self.path_list, self.issame_list = self._get_paths(lfw_root, pairs)

    def eval(self, model, far_target=1e-3, distance_metric=0, cached=False):
        if cached:
            embeddings = self.cached_embeddings
        else:
            embeddings_dict = self._get_embeddings_dict(model)
            embeddings = np.array([embeddings_dict[path] for path in self.path_list])
            self.cached_embeddings = embeddings
        _, _, accuracy, val_mean, val_std, _, _, _ = self._eval(embeddings, np.asarray(self.issame_list), distance_metric=distance_metric, far_target=far_target)
        acc_mean = np.mean(accuracy)
        acc_std = np.std(accuracy)
        return acc_mean, acc_std, val_mean, val_std
  
    def plot_roc(self, model, distance_metric=0, cached=False):
        if cached:
            embeddings = self.cached_embeddings
        else:
            embeddings_dict = self._get_embeddings_dict(model)
            embeddings = np.array([embeddings_dict[path] for path in self.path_list])
            self.cached_embeddings = embeddings
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]     
        thresholds = np.arange(0, 4, 0.001)
        tpr, fpr, _, _, _  = self._calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(self.issame_list), distance_metric=distance_metric)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr)
        plt.xlabel("FAR")
        plt.ylabel("TAR")
        plt.title("ROC")
        plt.show()
 
    def _get_embeddings_dict(self, model):
        embeddings = []
        paths = []
        total_iters = len(self.loader)
        model.eval()
        with torch.no_grad():
            for i, (xb, b_paths) in enumerate(self.loader):
                xb = xb.to(self.device)
                b_embeddings = model.encode(xb)
                b_embeddings = b_embeddings.to('cpu').numpy()
                paths.extend(b_paths)
                embeddings.extend(b_embeddings)
                if i%100==0: print('\rlfw verification: ' + "{:.2f}".format((i+1)/total_iters*100) + "%", end='')
        print("\rlfw verification: finished") 
        embeddings_dict = dict(zip(paths,embeddings))
        return embeddings_dict

    def _distance(self, embeddings1, embeddings2, distance_metric=0):
        if distance_metric==0:
            # Euclidian distance
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff),1)
        elif distance_metric==1:
            # distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot / norm
            dist = np.arccos(similarity) / math.pi
        else:
            raise 'Undefined distance metric %d' % distance_metric
        return dist

    def _calculate_roc(self, thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
        assert(embeddings1.shape[0] == embeddings2.shape[0])
        assert(embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds,nrof_thresholds))
        fprs = np.zeros((nrof_folds,nrof_thresholds))
        accuracy = np.zeros((nrof_folds))

        is_false_positive = []
        is_false_negative = []

        indices = np.arange(nrof_pairs)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if subtract_mean:
                mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
            else:
                mean = 0.0
            dist = self._distance(embeddings1-mean, embeddings2-mean, distance_metric)
            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx], _ ,_ = self._calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = self._calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
            _, _, accuracy[fold_idx], is_fp, is_fn = self._calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

            tpr = np.mean(tprs,0)
            fpr = np.mean(fprs,0)
            is_false_positive.extend(is_fp)
            is_false_negative.extend(is_fn)

        return tpr, fpr, accuracy, is_false_positive, is_false_negative

    def _calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
        is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

        tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
        acc = float(tp+tn)/dist.size
        return tpr, fpr, acc, is_fp, is_fn

    def _calculate_val(self, thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
        assert(embeddings1.shape[0] == embeddings2.shape[0])
        assert(embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        val = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)

        indices = np.arange(nrof_pairs)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if subtract_mean:
                mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
            else:
                mean = 0.0
            dist = self._distance(embeddings1-mean, embeddings2-mean, distance_metric)
            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = self._calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
            if np.max(far_train)>=far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0

            val[fold_idx], far[fold_idx] = self._calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

        val_mean = np.mean(val)
        far_mean = np.mean(far)
        val_std = np.std(val)
        return val_mean, val_std, far_mean

    def _calculate_val_far(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))
        val = float(true_accept) / float(n_same)
        far = float(false_accept) / float(n_diff)
        return val, far

    def _eval(self, embeddings, actual_issame, far_target=1e-3, nrof_folds=10, distance_metric=0, subtract_mean=False):
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        tpr, fpr, accuracy, fp, fn  = self._calculate_roc(thresholds, embeddings1, embeddings2,
            np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far = self._calculate_val(thresholds, embeddings1, embeddings2,
            np.asarray(actual_issame), far_target, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
        return tpr, fpr, accuracy, val, val_std, far, fp, fn

    def _add_extension(self, path):
        if os.path.exists(path+'.jpg'):
            return path+'.jpg'
        elif os.path.exists(path+'.png'):
            return path+'.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def _get_paths(self, lfw_dir, pairs):
        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = self._add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self._add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = self._add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self._add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list += (path0,path1)
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list, issame_list

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs, dtype=object)


#evaluace na testovasi sade 
def test(model, dataloader, device):
    correct = total = 0
    total_iters = len(dataloader)
    model.eval()
    with torch.no_grad():    
        for i, (inputs, labels) in enumerate(dataloader):
            labels = labels.to(device)
            outputs = model(inputs.to(device))
            predicted = torch.argmax(outputs.data, dim=1)
            correct += predicted.eq(labels.data).cpu().sum()
            total += labels.size(0)
            if i%100==0: print('\rtest: ' + "{:.2f}".format((i+1)/total_iters*100) + "%", end='')
    print("\rtest: finished") 
    return (correct / total).item()

#vizualizace prostoru embeding??
def visualize_embeding(model, dataloader, device):  
    model.eval()
    total_iters = len(dataloader)
    all_outputs,all_labels = np.zeros((0,512)),[]
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(dataloader):
          outputs = model.encode(inputs.to(device))
          all_outputs = np.vstack((all_outputs, outputs.cpu().detach().numpy()))
          all_labels.extend(list(labels))
          if i%10==0: print('\rvizualization: eval - ' + "{:.2f}".format((i+1)/total_iters*100) + "%", end='')
    print("\rvizualization: TSNE fitting", end='')
    outputs_embedded = TSNE(n_components=2).fit_transform(all_outputs)
    print("\rvizualization: plot", end='') 
    x,y = np.hsplit(outputs_embedded, 2)
    x,y = x.flatten(), y.flatten()
    colors = [list(np.random.rand(3)) for i in range(unique_cnt(all_labels))]
    plt.figure(figsize=(5,5))
    for i in range(len(all_labels)):
      plt.scatter(x[i], y[i], color=colors[all_labels[i]], s=1)
      if i%20==0: print("\rvizualization: plotting - " + "{:.2f}".format((i+1)/len(all_labels)*100) + "%", end='') 
    print("\rvizualization: finished") 
    plt.show()

def unique_cnt(numbers):
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return len(list_of_unique_numbers)