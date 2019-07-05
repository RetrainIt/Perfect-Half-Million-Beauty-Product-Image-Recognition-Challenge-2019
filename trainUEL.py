from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
import math
import numpy as np

from BatchAverage import BatchAverage, BatchCriterion
from utils import *
import pandas as pd
import faiss

from PIL import Image, ImageChops,ImageFile
from tqdm import tqdm
c
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='train UEL')

# options
parser.add_argument('--train_path', '-train_path', default='./train/', type=str, 
                    help='train path')
parser.add_argument('--test_path', '-test_path', default='./val/', type=str, 
                    help='test path')
parser.add_argument('--tlabel', '-tlabel', default='./val.csv', type=str, 
                    help='test label (.csv)')
					
def get_imlist(path):
    imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist.sort()
    return imlist
	
def pad(im, imsize):
    if im.size[0]>im.size[1]:
        im = im.resize((imsize, int(imsize*im.size[1]/im.size[0])), Image.ANTIALIAS)
    elif im.size[1]>im.size[0]:
        im = im.resize((int(imsize*im.size[0]/im.size[1]), imsize), Image.ANTIALIAS)
    else:
        im = im.resize((imsize, imsize), Image.ANTIALIAS)
    
    new_im = Image.new(im.mode,(imsize, imsize), 'white')

    new_im.paste(im, (int((imsize-im.size[0])/2),
                          int((imsize-im.size[1])/2)))
    
    return new_im

def read_image(path):
    #if memory is insufficient, you can try to save in your computer firstly.
        img_list = []
        file = os.listdir(path) 
        file.sort()
        for file_name in file:
            #print(class_id, file_name)
            img = Image.open(path+file_name)
            img = pad(img,256)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
                pix_array = np.repeat(pix_array, 3, 2)
            if pix_array.shape[2]==4:
                pix_array=pix_array[:,:,:3]
                
            img_list.append(pix_array)
            
        return img_list
    
class DataLoader(data.Dataset):
    """Metric Learning Dataset.
    """
    def __init__(self, path, transform=None, target_transform=None, nnIndex = None):
        
        self.path = path
        self.img_data  = read_image(self.path)
        self.transform = transform
        self.target_transform = target_transform
        self.nnIndex = nnIndex

    def __getitem__(self, index):
        
        img = self.img_data[index]
        img = self.transform(img)

        return img, index
        
    def __len__(self):
        return len(self.img_data)
		
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20:
        lr = lrr
    elif epoch >= 20 and epoch < 40:
        lr = lrr * 0.1
    else:
        lr = lrr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
		
# Training
def train(epoch,best_MAP):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    if freeze_bn:
        net.apply(set_bn_to_eval)
    
    end = time.time()
    for batch_idx, (inputs1, inputs2,  indexes) in enumerate(trainloader):
        net.train()
        data_time.update(time.time() - end)
        inputs1, inputs2,  indexes = inputs1.to(device), inputs2.to(device), indexes.to(device)
        
        inputs = torch.cat((inputs1,inputs2), 0)  
        optimizer.zero_grad()

        features = net(inputs)
        
        outputs = lemniscate(features, indexes)

        loss = criterion(outputs, indexes)
        
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if batch_idx%10 ==0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '.format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
            
        if batch_idx%100 ==0:

            print('----------Evaluation---------')
            start = time.time()
            MAP,test_features,test2_features=Map(net, trainloader, testloader,testloader2,low_dim)
            print("Evaluation Time: '{}'s".format(time.time()-start))

            if MAP > best_MAP:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'lemniscate': lemniscate,
                    'MAP': MAP,
                    'epoch': epoch,
                }
                if not os.path.isdir(model_dir):
                    os.mkdir(model_dir)
                torch.save(state, './'+model_dir + suffix + '_best.t')
                best_MAP = MAP

            print('MAP: {}% \t (best MAP: {})'.format(MAP,best_MAP))
            print('[Epoch]: {}'.format(epoch), file = test_log_file)
            print('MAP: {}% \t (best MAP: {})'.format(MAP,best_MAP), file = test_log_file)
            test_log_file.flush()
    return best_MAP
	
def Map(net, trainloader, testloader,testloader2,low_dim):
    net.eval()
    ptr =0
    test_size = testloader.dataset.__len__()
    test_features = np.zeros((test_size,low_dim))
    with torch.no_grad():
        for batch_idx, (inputs, indexes) in enumerate(testloader):
            batchSize = inputs.size(0)
            real_size = min(batchSize, test_batch)

            batch_feat = net(inputs)
            test_features[ptr:ptr+real_size,:] = np.asarray(batch_feat.cpu())
            ptr += real_size
    feats = np.array(test_features).astype('float32')

    ptr =0
    test2_size = testloader2.dataset.__len__()
    test2_features = np.zeros((test2_size,low_dim))
    with torch.no_grad():
        for batch_idx, (inputs, indexes) in enumerate(testloader2):
            batchSize = inputs.size(0)
            real_size = min(batchSize, test_batch)

            batch_feat = net(inputs)
            test2_features[ptr:ptr+real_size,:] = np.asarray(batch_feat.cpu())
            ptr += real_size
    query_feats = np.array(test2_features).astype('float32')
    # caculate MAP
    one = 0
    apt=0
    tt=0
    ap=0
    index_flat = faiss.IndexFlatL2(feats.shape[1])
    index_flat.add(feats)
    D,I = index_flat.search(query_feats,7)

    for query_num in range(len(val_label)):
        queryDir = query_list[query_num]
        top_num = 7
        imlist = [data_list[index] for i,index in enumerate(I.T[0:top_num,query_num])]
        for num in range(top_num):
            if imlist[num] in set(val_label[query_num]):
                one += 1
                tt += one/float(num+1)
        if one!= 0:
            ap = tt/one
            apt += ap
        tt=0
        one=0
    MAP_score = apt/len(val_label)
    return MAP_score,test_features,test2_features

def main():
    args = parser.parse_args()
    
    # read database
	train_path=args.train_path
	test_path=args.test_path
	tlabel=args.tlabel
	lrr=0.001
	model_dir='checkpoint/'
	low_dim=128
	batch_t=0.1
	batch_size=32
	batch_m=1
	ptr=0
	test_batch=32
	gpu='0,1'
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	freeze_bn = 0
	K_list = [1,2,4,8]
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	best_MAP = 0
	start_epoch = 0
	test_epoch = 1
	#data_read
	train_images = get_imlist(train_path)
	test_images = get_imlist(test_path)
	#list read
	images = [os.path.join(train_path,f) for f in os.listdir(train_path) if f.endswith('.jpg')]
	images.sort()
	data_list=list(map(lambda x:  x.split('/')[-1],images))
	#test label
	val=pd.read_table(tlabel)
	val.columns=[1]
	query_list = list(val[1].map(lambda x: tpath + x.split(',')[0] + '.jpg'))
	val_label = []
	for num in range(len(val[1])):
		t=[]
		for num1 in range(len(val[1][num].split(','))-1):
			t.append(val[1][num].split(',')[num1+1]+'.jpg')
		val_label.append(t)
	#init model
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(size=224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])

	transform_test = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])
	trainset = DataLoader(path = train_path, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
	ntrain = trainset.__len__()
	testset = DataLoader(path = train_path, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
	ntest = testset.__len__()
	
	net = resnet101(pretrained=True,low_dim=128)
	#train set
	lemniscate = BatchAverage(low_dim, batch_t, batch_size)
	if device == 'cuda':
		net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

	criterion = BatchCriterion(batch_m, batch_size)

	net.to(device)
	lemniscate.to(device)
	criterion.to(device)

	optimizer = optim.SGD( net.parameters() , lr=lrr, momentum=0.9, weight_decay=5e-4)
		
	#train
	for epoch in range(start_epoch, start_epoch+100):
		nn_index = np.arange(ndata)
		trainloader.dataset.nnIndex = nn_index
		# training
		best_MAP=train(epoch,best_MAP)

if __name__ == '__main__':
  main()
