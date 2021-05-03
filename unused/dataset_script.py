import os 
import random

#LFW PAIRS

result = pd.DataFrame()
file = open('lfw/pairs.txt', 'r')
for line in file.read().splitlines()[1:]: #bez prvniho radku
  h = line.split('\t')
  if len(h) == 3: #positive
    i1 = os.path.join(h[0], h[0] + '_' + str(h[1]).zfill(4) + '.jpg')
    i2 = os.path.join(h[0], h[0] + '_' + str(h[2]).zfill(4) + '.jpg') 
    result = result.append([(i1,i2,1)])
  else:
    i1 = os.path.join(h[0], h[0] + '_' + str(h[1]).zfill(4) + '.jpg')
    i2 = os.path.join(h[2], h[2] + '_' + str(h[3]).zfill(4) + '.jpg')
    result = result.append([(i1,i2,0)])
print(result)
result.to_csv('lfw_pairs.csv', index=False, header=False) 


#CASIA PAIRS

result = pd.DataFrame()
df = pd.read_csv('CASIA/casia2.csv', header=None)
df = df[df[1] > 10200]
#positive
for i in range(500):
  i1 = df.iloc[random.randrange(0,len(df)),:]
  df = df[df[0]!=i1[0]]
  candidates = df[df[1] == i1[1]]
  i2 = candidates.iloc[random.randrange(0,len(candidates)),:]
  df = df[df[0]!=i2[0]]
  result = result.append([(i1[0], i2[0], 1)])
#negative
for i in range(500):
  i1 = df.iloc[random.randrange(0,len(df)),:]
  df = df[df[0]!=i1[0]]
  candidates = df[df[1] != i1[1]]
  i2 = candidates.iloc[random.randrange(0,len(candidates)),:]
  df = df[df[0]!=i2[0]]
  result = result.append([(i1[0], i2[0], 0)])
print(result)
result.to_csv('casia_pairs.csv', index=False, header=False) 


#CASIA TRAIN + TEST

train = pd.read_csv('CASIA/casia2.csv', header=None)
train = train[train[1] <= 10200]
test = pd.DataFrame()
train = train.sample(frac=1) #shuffle
test = train.iloc[0:48423,:]
train = train.iloc[48423:,:]
print(test)
print(train)
train.to_csv('casia_train.csv', index=False, header=False) 
test.to_csv('casia_test.csv', index=False, header=False) 