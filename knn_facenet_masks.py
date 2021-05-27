#ver 1e-3 0.939 mask 0.718
#ver 1e-4 0.299 mask 0.203
#s mcnn orezem
#ver 1e-3 0.936 mask 0.737
#ver 1e-4 0.323 mask 0.230

#!pip install facenet_pytorch
#!tar -xvzf  /content/drive/MyDrive/KNN/lfw.tgz
#!unzip /content/drive/MyDrive/KNN/CASIA.zip
#!unzip /content/drive/MyDrive/KNN/lfw_with_masks.zip
#!7za x /content/drive/MyDrive/KNN/casia_with_masks.zip

#https://github.com/timesler/facenet-pytorch/tree/master/examples
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#dataset_dir = 'casia_with_masks'              
dataset_dir = 'CASIA'
batch_size = 32                  
transformation = transforms.Compose([
    transforms.Resize(96), #160),
    np.float32,
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(dataset_dir, transform=transformation)

indexes = np.arange(len(dataset))
np.random.shuffle(indexes)
train_idxs = indexes[:int(0.9 * len(indexes))]        
test_idxs = indexes[int(0.9 * len(indexes)):]

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_idxs)
)

test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(test_idxs)
)

class Pretrained(nn.Module):
    def __init__(self, num_classes):
        super(Pretrained, self).__init__()    
        self.pretrained_model = InceptionResnetV1(classify=False, pretrained='vggface2')  
        emb_size=512 #TODO
        #self.dropout = nn.Dropout(p=0.5, inplace=False)
        #self.batchNorm1d = nn.BatchNorm1d(emb_size)
        #self.relu = nn.ReLU()
        self.last_fc = nn.Linear(emb_size, num_classes, bias=False)

    def save_model(self, train_acc):
        model_path = "./models/resnet50_acc_" + '{:.3f}'.format(train_acc) + "_time_" + str(calendar.timegm(time.gmtime()))
        torch.save(self.state_dict(), model_path)
        
    def encode(self, x):
        x = self.pretrained_model(x)
        #batch, drop...
        return x

    def forward(self, x):
        x = self.encode(x) 
        #x_norm = F.normalize(x, p=2, dim=1) * 0.877
        #w_norm = F.normalize(self.last_fc.weight, p=2, dim=1) * 0.877
        #logits = x_norm.matmul(w_norm.T)
        logits = self.last_fc(x)
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

class Lfw_verification:
  def __init__(self, lfw_root, lfw_pairs, device):
    self.lfw_root = lfw_root
    self.lfw_pairs = lfw_pairs
    self.batch_size = 16
    self.device = device 
    #loader
    trans = transforms.Compose([ 
      transforms.Resize(96),   #160
      np.float32,
      transforms.ToTensor(),
      fixed_image_standardization
    ])
    ds = datasets.ImageFolder(lfw_root, transform=trans)
    ds.samples = [(p, p) for p, _ in ds.samples]
    self.loader = DataLoader(
      ds,
      num_workers = 2, # 0 if os.name == 'nt' else 8,
      batch_size=self.batch_size,
      sampler=SequentialSampler(ds)
    )
    pairs = self.read_pairs(lfw_pairs)
    self.path_list, self.issame_list = self.get_paths(lfw_root, pairs)

    #print(self.path_list)
    #print(self.issame_list)

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
            #classes.extend(yb.numpy())
            paths.extend(b_paths)
            embeddings.extend(b_embeddings)
            if i%100==0: print('\rlfw verification: ' + "{:.2f}".format((i+1)/total_iters*100) + "%", end='')
    print("\rlfw verification") #odradkovani
    embeddings_dict = dict(zip(paths,embeddings))
    return embeddings_dict

  def verify(self, model, far_target=1e-4):
    embeddings_dict = self.get_embeddings_dict(model)
    embeddings = np.array([embeddings_dict[path] for path in self.path_list])
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]     
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = self.calculate_val(thresholds, embeddings1, embeddings2, np.asarray(self.issame_list), far_target)
    return val
  
  def print_roc(self, model):
    embeddings_dict = self.get_embeddings_dict(model)
    embeddings = np.array([embeddings_dict[path] for path in self.path_list])
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]     
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, fp, fn  = self.calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(self.issame_list))
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



#MAIN

lfw_ver = Lfw_verification('lfw_with_masks','drive/MyDrive/KNN/pairs.txt',device)
lfw_ver_masks = Lfw_verification('lfw_with_masks','drive/MyDrive/KNN/pairs_with_masks.txt',device)

arcface = False
model = Pretrained(10575).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4) #0.0001
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

#scheduler = MultiStepLR(optimizer, [5, 10])
criterion = ArcFace() if arcface else nn.CrossEntropyLoss()

lfw_ver_interval = 100
train_acc_interval = 10

# Training
iter = 0
train_acc = 0
for epoch in range(6): 
    loss_sum = correct = total = 0
    for batch, (inputs, labels) in enumerate(train_loader):    

        model.train()
        #model.pretrained_model.eval()
        inputs, labels = inputs.to(device), labels.to(device)
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
          print('iter',iter,'epoch',epoch,'t acc',train_acc,'loss',avg_loss)

        if iter != 0 and iter%lfw_ver_interval == 0: 
          #print('ver = ', "{:.4f}".format(lfw_ver.verify(model)))
          #print('ver masks =', "{:.4f}".format(lfw_ver_masks.verify(model)))
          lfw_ver.print_roc(model)
        
        iter += 1