import os
import h5py
import pandas as pd
import copy
from RAMAC import extract_feature
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageChops
import torch
from diffusion import Diffusion
from resnet import resnet101
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors

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
        
def obtainf(net, testloader):
    net.cuda()
    net.eval()

    ptr =0
    test_size = testloader.dataset.__len__()
    test_features = np.zeros((test_size,128))
    with torch.no_grad():
        for batch_idx, (inputs, indexes) in enumerate(testloader):
            batchSize = inputs.size(0)
            real_size = min(batchSize, 32)

            batch_feat = net(inputs.cuda())
            test_features[ptr:ptr+real_size,:] = np.asarray(batch_feat.cpu())
            ptr += real_size
    query_feats = np.array(test_features).astype('float32')
    
    return test_features    
	
def main():

	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('train_images_path')
	arg_parser.add_argument('test_images_path')
	arg_parser.add_argument('predictions_path')
	args = arg_parser.parse_args()

	imsize=480
	train_path=args.train_images_path
	test_path=args.test_images_path
	outfile=args.predictions_path
	##read data##
	data_list=os.listdir(ipath)
	train_images = get_imlist(train_path)
	test_images = get_imlist(test_path)
	##RAMAC##
	RAMAC = extract_feature(train_images, 'resnet101', imsize)
	RAMAC_test = extract_feature(test_images, 'resnet101', imsize)
	##UEL##
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
	net = resnet101(pretrained=True,low_dim=128)
	model_path = './model/UEL.t'#After training UEL
	net.load_state_dict(torch.load())
	
	imset = DataLoader(path = train_path, transform=transform_test)
	train_loader = torch.utils.data.DataLoader(imset, batch_size=32, shuffle=False, num_workers=0)
	UEL = obtainf(net, train_loader)
	
	imset = DataLoader(path = test_path, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(imset, batch_size=32, shuffle=False, num_workers=0)
	UEL_test = obtainf(net, test_loader)
	##GEM##
	image_size=1024
	multiscale='[1, 2**(1/2), 1/2**(1/2)]'
	state = torch.load('./model/retrievalSfM120k-vgg16-gem-b4dcdc6.pth')
	
	net_params = {}
	net_params['architecture'] = state['meta']['architecture']
	net_params['pooling'] = state['meta']['pooling']
	net_params['local_whitening'] = state['meta'].get('local_whitening', False)
	net_params['regional'] = state['meta'].get('regional', False)
	net_params['whitening'] = state['meta'].get('whitening', False)
	net_params['mean'] = state['meta']['mean']
	net_params['std'] = state['meta']['std']
	net_params['pretrained'] = False
	# load network
	net = init_network(net_params)
	net.load_state_dict(state['state_dict'])
        
	# if whitening is precomputed
	if 'Lw' in state['meta']:
		net.meta['Lw'] = state['meta']['Lw']
	ms = list(eval(multiscale))
	msp = net.pool.p.item()
	
	net.cuda()
	net.eval()
	# set up the transform
	normalize = transforms.Normalize(
		mean=net.meta['mean'],
		std=net.meta['std']
    )
	transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
	
	GEM = extract_vectors(net,train_images , 480, transform, ms=ms, msp=msp).numpy().T
	GEM_test = extract_vectors(net,test_images , 480, transform, ms=ms, msp=msp).numpy().T
	##Retrieval##
	feats=np.concatenate((RAMAC,UEL,GEM),axis=1).astype('float32')
	query_feat=np.concatenate((RAMAC_test, UEL_test,GEM_test),axis=1).astype('float32')
	##diffusion##
	kq, kd = 7, 50
	gamma=80
	diffusion = Diffusion(feats, '/')
	offline = diffusion.get_offline_results(1024, kd)
	print('[search] 1) k-NN search')
	sims, ids = diffusion.knn.search(query_feat, kq)
	sims = sims ** gamma
	qr_num = ids.shape[0]
	print('[search] 2) linear combination')
	all_scores = np.empty((qr_num, 7), dtype=np.float32)
	all_ranks = np.empty((qr_num, 7), dtype=np.int)
	for i in range(qr_num):
		scores = sims[i] @ offline[ids[i]]
		parts = np.argpartition(-scores, 7)[:7]
		ranks = np.argsort(-scores[parts])
		all_scores[i] = scores[parts][ranks]
		all_ranks[i] = parts[ranks]
	I = all_ranks
	##output##
	out=pd.DataFrame(list(map(lambda x: x.split('/')[-1].split('.jpg')[0],timages)))
	out['1']=pd.DataFrame(I)[0].map(lambda x:data_list[x].split('.')[0] )
	out['2']=pd.DataFrame(I)[1].map(lambda x:data_list[x].split('.')[0] )
	out['3']=pd.DataFrame(I)[2].map(lambda x:data_list[x].split('.')[0] )
	out['4']=pd.DataFrame(I)[3].map(lambda x:data_list[x].split('.')[0] )
	out['5']=pd.DataFrame(I)[4].map(lambda x:data_list[x].split('.')[0] )
	out['6']=pd.DataFrame(I)[5].map(lambda x:data_list[x].split('.')[0] )
	out['7']=pd.DataFrame(I)[6].map(lambda x:data_list[x].split('.')[0] )
	out.to_csv(outfile,index=None,header=None)
	

if __name__ == '__main__':
    main()
