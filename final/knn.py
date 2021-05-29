
#readme
#pip install facenet_pytorch
#unzip /content/drive/MyDrive/KNN/lfw_with_masks.zip
#7za x /content/drive/MyDrive/KNN/casia_with_masks.zip

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy import interpolate
import calendar
import math
from numpy import save
from numpy import load
from sklearn.manifold import TSNE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

dataset_dir = 'casia_with_masks'              
batch_size = 32                  
transformation = transforms.Compose([
    transforms.Resize(160), 
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(dataset_dir, transform=transformation)

indexes = np.arange(len(dataset))
np.random.shuffle(indexes)
train_idxs = indexes[:int(0.9 * len(indexes))]        
test_idxs = indexes[int(0.9 * len(indexes)):]

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_idxs),
    collate_fn=lambda x: default_collate(x).to(device)
)
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(test_idxs),
    collate_fn=lambda x: default_collate(x).to(device)
)

vizualization_with_masks = True
visualization_cnt = 20
indexes_visualization = np.array([i for i in range(len(dataset)) if dataset.imgs[i][1] < visualization_cnt and (vizualization_with_masks or "m." not in dataset.imgs[i][0])])  
visualization_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(indexes_visualization),
    collate_fn=lambda x: default_collate(x).to(device)
)

class Pretrained(nn.Module):
    def __init__(self, arcface=False):
        super(Pretrained, self).__init__()    
        self.arcface = arcface
        self.pretrained_model = InceptionResnetV1(classify=True, pretrained='casia-webface') 
        if arcface:
            self.last_fc = list(pretrained_model.children())[-1]
            self.last_fc.bias = None

    def save_model(self, train_acc):
        model_path = "./models/model_acc_" + '{:.3f}'.format(train_acc) + "_time_" + str(calendar.timegm(time.gmtime()))
        torch.save(self.state_dict(), model_path)
        
    def encode(self, x):
        self.pretrained_model.classify = False
        x = self.pretrained_model(x)
        return x

    def forward(self, x):
        if self.arcface: 
            x = self.encode(x) 
            weight = F.normalize(self.last_fc.weight, p=2, dim=1) 
            logits = x.matmul(weight.T)
        else:
            self.pretrained_model.classify = True
            logits = self.pretrained_model(x)
        return logits


#outputs [B,10] targets [B,]
class ArcFace(nn.Module):
    def __init__(self, margin = 0.5, scale = 64):
        super(ArcFace, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets): #0.5 64
        criterion = nn.CrossEntropyLoss()
        original_target_logits = outputs.gather(1, torch.unsqueeze(targets, 1))
        # arccos grad divergence error https://github.com/pytorch/pytorch/issues/8069
        #eps = 1e-7 
        #original_target_logits = torch.clamp(original_target_logits, -1+eps, 1-eps)
        original_target_logits = original_target_logits * 0.999999
        thetas = torch.acos(original_target_logits) 
        marginal_target_logits = torch.cos(thetas + self.margin)

        #f=original_target_logits-marginal_target_logits
        #print(min(f),max(f))
        #print('orig',original_target_logits)
        #print('mafg',marginal_target_logits)

        one_hot_mask = F.one_hot(targets, num_classes=outputs.shape[1])
        
        diff = marginal_target_logits - original_target_logits
        #diff = torch.clip(diff,-1,0)
        expanded = diff.expand(-1, outputs.shape[1])
        outputs = self.scale * (outputs + (expanded * one_hot_mask))
        return criterion(outputs, targets)

def test(model, dataloader)
    correct = total = 0
    total_iters = len(dataloader)
    model.eval()
    with torch.no_grad():    
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            correct += predicted.eq(labels.data).cpu().sum()
            total += labels.size(0)
            if i%100==0: print('\rtest: ' + "{:.2f}".format((i+1)/total_iters*100) + "%", end='')
    print("\rtest: finished") 
    return (correct / total).item()

def visualize_embeding(model, dataloader):  
    model.eval()
    total_iters = len(dataloader)
    all_outputs,all_labels = np.zeros((0,512)),[]
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(dataloader):
          outputs = model.encode(inputs)
          all_outputs = np.vstack((all_outputs, outputs.cpu().detach().numpy()))
          all_labels.extend(list(labels))
          if i%10==0: print('\rvizualization: eval - ' + "{:.2f}".format((i+1)/total_iters*100) + "%", end='')
    print("\rvizualization: TSNE fitting", end='')
    outputs_embedded = TSNE(n_components=2).fit_transform(all_outputs)
    print("\rvizualization: plot", end='') 
    x,y = np.hsplit(outputs_embedded, 2)
    x,y = x.flatten(), y.flatten()
    colors = [list(np.random.rand(3)) for i in range(unique_cnt(all_labels))]
    for i in range(len(all_labels)):
      plt.scatter(x[i], y[i], color=colors[all_labels[i]])
    print("\rvizualization: finished") 
    plt.show()

def unique_cnt(numbers):
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return len(list_of_unique_numbers)

class Lfw_verification:
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
    pairs = self.read_pairs(lfw_pairs)
    self.path_list, self.issame_list = self.get_paths(lfw_root, pairs)

  def get_embeddings_dict(self, model):
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

  def eval(self, model, distance_metric=0):
    embeddings_dict = self.get_embeddings_dict(model)
    embeddings = np.array([embeddings_dict[path] for path in self.path_list])
    tpr, fpr, accuracy, val, val_std, far, fp, fn = self.evaluate(embeddings, np.asarray(self.issame_list), distance_metric=distance_metric)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    print('acc_mean',acc_mean,'acc_std',acc_std,'val', val,'val_std', val_std,'far', far)

  def verify(self, model, far_target=1e-3, distance_metric=0):
    embeddings_dict = self.get_embeddings_dict(model)
    embeddings = np.array([embeddings_dict[path] for path in self.path_list])
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]     
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = self.calculate_val(thresholds, embeddings1, embeddings2, np.asarray(self.issame_list), far_target, distance_metric=distance_metric)
    return val
  
  def print_roc(self, model, distance_metric=0):
    embeddings_dict = self.get_embeddings_dict(model)
    embeddings = np.array([embeddings_dict[path] for path in self.path_list])
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]     
    thresholds = np.arange(0, 0.9, 0.0001)
    tpr, fpr, accuracy, fp, fn  = self.calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(self.issame_list), distance_metric=distance_metric)
    plt.plot(fpr, tpr)
    plt.show()

  # LFW functions taken from David Sandberg's FaceNet implementation
  def distance(self, embeddings1, embeddings2, distance_metric=0):
      if distance_metric==0:
          # Euclidian distance
          diff = np.subtract(embeddings1, embeddings2)
          dist = np.sum(np.square(diff),1)
      elif distance_metric==1:
          # Distance based on cosine similarity
          dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
          norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
          similarity = dot / norm
          dist = np.arccos(similarity) / math.pi
      else:
          raise 'Undefined distance metric %d' % distance_metric

      return dist

  def calculate_roc(self, thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
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
          dist = self.distance(embeddings1-mean, embeddings2-mean, distance_metric)

          # Find the best threshold for the fold
          acc_train = np.zeros((nrof_thresholds))
          for threshold_idx, threshold in enumerate(thresholds):
              _, _, acc_train[threshold_idx], _ ,_ = self.calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
          best_threshold_index = np.argmax(acc_train)
          for threshold_idx, threshold in enumerate(thresholds):
              tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = self.calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
          _, _, accuracy[fold_idx], is_fp, is_fn = self.calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

          tpr = np.mean(tprs,0)
          fpr = np.mean(fprs,0)
          is_false_positive.extend(is_fp)
          is_false_negative.extend(is_fn)

      return tpr, fpr, accuracy, is_false_positive, is_false_negative

  def calculate_accuracy(self, threshold, dist, actual_issame):
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

  def calculate_val(self, thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
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
          dist = self.distance(embeddings1-mean, embeddings2-mean, distance_metric)
          # Find the threshold that gives FAR = far_target
          far_train = np.zeros(nrof_thresholds)
          for threshold_idx, threshold in enumerate(thresholds):
              _, far_train[threshold_idx] = self.calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
          if np.max(far_train)>=far_target:
              f = interpolate.interp1d(far_train, thresholds, kind='slinear')
              threshold = f(far_target)
          else:
              threshold = 0.0

          val[fold_idx], far[fold_idx] = self.calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

      val_mean = np.mean(val)
      far_mean = np.mean(far)
      val_std = np.std(val)
      return val_mean, val_std, far_mean

  def calculate_val_far(self, threshold, dist, actual_issame):
      predict_issame = np.less(dist, threshold)
      true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
      false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
      n_same = np.sum(actual_issame)
      n_diff = np.sum(np.logical_not(actual_issame))
      val = float(true_accept) / float(n_same)
      far = float(false_accept) / float(n_diff)
      return val, far

  def evaluate(self, embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
      # Calculate evaluation metrics
      thresholds = np.arange(0, 4, 0.01)
      embeddings1 = embeddings[0::2]
      embeddings2 = embeddings[1::2]
      tpr, fpr, accuracy, fp, fn  = self.calculate_roc(thresholds, embeddings1, embeddings2,
          np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
      thresholds = np.arange(0, 4, 0.001)
      val, val_std, far = self.calculate_val(thresholds, embeddings1, embeddings2,
          np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
      return tpr, fpr, accuracy, val, val_std, far, fp, fn

  def add_extension(self, path):
      if os.path.exists(path+'.jpg'):
          return path+'.jpg'
      elif os.path.exists(path+'.png'):
          return path+'.png'
      else:
          raise RuntimeError('No file "%s" with extension png or jpg.' % path)

  def get_paths(self, lfw_dir, pairs):
      nrof_skipped_pairs = 0
      path_list = []
      issame_list = []
      for pair in pairs:
          if len(pair) == 3:
              path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
              path1 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
              issame = True
          elif len(pair) == 4:
              path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
              path1 = self.add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
              issame = False
          if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
              path_list += (path0,path1)
              issame_list.append(issame)
          else:
              nrof_skipped_pairs += 1
      if nrof_skipped_pairs>0:
          print('Skipped %d image pairs' % nrof_skipped_pairs)

      return path_list, issame_list

  def read_pairs(self, pairs_filename):
      pairs = []
      with open(pairs_filename, 'r') as f:
          for line in f.readlines()[1:]:
              pair = line.strip().split()
              pairs.append(pair)
      return np.array(pairs, dtype=object)

  def print_ROC(self, TARs, FARs):
    plt.plot(FARs, TARs)
    plt.show()




lfw_ver_masks = Lfw_verification('lfw_with_masks','drive/MyDrive/KNN/pairs_with_masks.txt', device)
casia_test = Casia_test(test_loader, device)

arcface = False

model = Pretrained(arcface).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4) 
criterion = ArcFace() if arcface else nn.CrossEntropyLoss()

lfw_ver_interval = 1000
train_acc_interval = 100
test_interval = 1000

# Training
iter = 0
train_acc = 0
for epoch in range(6): 
    loss_sum = correct = total = 0
    for batch, (inputs, labels) in enumerate(train_loader_no_masks):    
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        avg_loss = loss_sum / (batch+1)   

        predicted = torch.argmax(outputs.data, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        train_acc = (correct / total).item()

        if iter%train_acc_interval == 0:
            print('iter',iter,'epoch',epoch,'train acc',train_acc,'loss',avg_loss)

        if iter%test_interval == 0:
            test_acc = test(model, test_loader)
            print('iter',iter,'epoch',epoch,'test acc',test_acc)

        if iter != 0 and iter%lfw_ver_interval == 0: 
            print('ver =', "{:.4f}".format(lfw_ver_masks.verify(model)))

        iter += 1