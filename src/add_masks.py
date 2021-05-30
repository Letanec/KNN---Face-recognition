import numpy as np
import sys
import dlib
import cv2
import matplotlib.pyplot as plt
import urllib.request as urlreq
import os
from pylab import rcParams
import random
import torch
from facenet_pytorch import MTCNN

lfw_dir = '../datasets/lfw'
lfw_dir_new = '../datasets/lfw_with_masks'
pairs_filename = '../datasets/lfw/pairs.txt'
new_pairs_filename = '../datasets/lfw_with_masks/pairs_with_masks.txt'
casia_dir = '../datasets/casia'
casia_dir_new = '../datasets/casia_with_masks'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mtcnn = MTCNN(keep_all=True, device=device)
colors = [(251,223,3), (251,253,3),(34,35,21),(34,185,252)]

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def load(path):
  image = cv2.imread(path)
  return image

def is_img(path):
  return path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

def add_mask(image):
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  try: rects = detector(image_gray,1)
  except: return None
  if (len(rects)==0): return None
  try: landmarks = shape_to_np(predictor(image_gray, rects[0]))
  except: return None
  if random.random() > 0.2: color = (251, 253, 191)
  else: color = colors[random.randint(0,len(colors)-1)]
  mask_shape = np.vstack((landmarks[1:16], landmarks[30]))
  cv2.fillPoly(image, pts=np.int32([mask_shape]), color=color)
  return image

def crop(image, box=None, margin=14):
  image = cv2.resize(image, (512,512))
  if isinstance(box, type(None)): 
    boxes, _, _ = mtcnn.detect(image, landmarks=True)
    if isinstance(boxes, type(None)) or len(boxes)==0: return None, None
    box = boxes[0]
  x,y,w,h = int(box[0]-margin), int(box[1]-margin), int(box[2]-box[0]+2*margin), int(box[3]-box[1]+2*margin)
  x,y = max(0,x), max(0,y)
  image = image[y:(y+h), x:(x+w)]
  return image, box

def save(image, path):
  image = cv2.resize(image, (160,160))
  cv2.imwrite(path, image)

def convert_path(path):
  extension = os.path.splitext(path)[1]
  return path.replace(extension, 'm'+extension)

#LFW

cnt = len(next(os.walk(lfw_dir))[1])-1
for i, (root, dirs, files) in enumerate(os.walk(lfw_dir)):
  if len(files)==0: continue
  new_root = root.replace(lfw_dir, lfw_dir_new)
  os.makedirs(new_root, exist_ok=True)
  for name in files:
    path = os.path.join(root, name)
    if not is_img(path): continue
    new_path = os.path.join(new_root, name)
    new_path_masked = new_path.replace("_0", "_1")
    image = load(path)
    image_cropped, box = crop(image)
    if isinstance(image_cropped, type(None)):
      save(image, new_path)
      continue
    save(image_cropped, new_path)
    image_masked = add_mask(image)
    if isinstance(image_masked, type(None)): continue
    image_cropped,_ = crop(image_masked, box)
    save(image_cropped, new_path_masked)
  print('\rLFW: adding masks -', "{:.2f}".format(i/cnt*100), "%,", str(i)+"/"+str(cnt), end='')

print('\rLFW: generating pairs', end='')

with open(pairs_filename, 'r') as f:
  for line in f.readlines()[1:]:
    pair = line.strip().split()
    #mask?
    choice = random.choice([0,1,2]) 
    if choice==0:
       mask_1 = mask_2 = False
    elif choice==1:
      mask_1, mask_2 = True, False
    else:
      mask_1 = mask_2 = True

    if len(pair) == 3:
      if mask_1:
        path = os.path.join(lfw_dir_new, pair[0], pair[0] + '_' + '%04d' % (int(pair[1])+1000) + '.jpg')
        if os.path.exists(path): pair[1] = str(int(pair[1])+1000)
      if mask_2:
        path = os.path.join(lfw_dir_new, pair[0], pair[0] + '_' + '%04d' % (int(pair[2])+1000) + '.jpg')
        if os.path.exists(path): pair[2] = str(int(pair[2])+1000)
    else:
      if mask_1:
        path = os.path.join(lfw_dir_new, pair[0], pair[0] + '_' + '%04d' % (int(pair[1])+1000) + '.jpg')
        if os.path.exists(path): pair[1] = str(int(pair[1])+1000)    
      if mask_2:
        path = os.path.join(lfw_dir_new, pair[2], pair[2] + '_' + '%04d' % (int(pair[3])+1000) + '.jpg')
        if os.path.exists(path): pair[3] = str(int(pair[3])+1000)    
    #write to file 
    with open(new_pairs_filename,'a') as file:
      file.write("\t".join(pair) + "\n")

print('\rLFW: finished')

#CASIA

cnt = len(next(os.walk(casia_dir))[1])
for i, (root, dirs, files) in enumerate(os.walk(casia_dir)):
  if len(files)==0: continue
  new_root = root.replace(casia_dir, casia_dir_new)
  os.makedirs(new_root, exist_ok=True)
  for name in files:
    path = os.path.join(root, name)
    if not is_img(path): continue
    new_path = os.path.join(new_root, name)
    new_path_masked = convert_path(new_path)
    image = load(path)
    image_cropped, box = crop(image)
    if isinstance(image_cropped, type(None)): continue
    save(image_cropped, new_path)
    image_masked = add_mask(image)
    if isinstance(image_masked, type(None)): continue
    image_cropped,_ = crop(image_masked, box)
    save(image_cropped, new_path_masked)
  print('\rCASIA: adding masks -', "{:.2f}".format(i/cnt*100), "%,", str(i)+"/"+str(cnt), end='')

print('\rCASIA: finished')
