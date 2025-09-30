# coding=gb2312
import math
import os.path
import sys
import skimage.io as sio

import main
import utils.noise

device = 'cuda'
import numpy as np
import matplotlib.pyplot as plt
from main import  loss_func,test_sr_img
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat,savemat
from sampling import pair_downsampler_3d1, pair_downsampler_3d2
from utils.utlis import evaluation,junk_band_remover,tensor2array,Nor
from utils.noise import mixed_noise
from compared import add_noise
from SR_learning_rank import loss_func2_eig2eig,test_sr_dir,SR_learn_HSI,reconstruct_from_subspace,loss_func2_eig2img

def train_sreig2eig_eigloss(model, optimizer, noisy_img,loss_function1=loss_func ,ele = None,rank=None,lossfunc2 =loss_func2_eig2eig):
    loss = loss_function1(model =  model, noisy_img=noisy_img)
    loss2 = lossfunc2(model2 =  model, eigen_image=noisy_img, ele=ele, rank=rank)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train_sr_eig2eig_imgloss(model, optimizer, noisy_img,loss_function=loss_func2_eig2eig ,ele = None,rank=None,lossfunc2 = loss_func):
    loss = loss_function(model2 =  model, eigen_image=noisy_img, ele=ele, rank=rank)
    loss2 = lossfunc2(model =  model, noisy_img=noisy_img)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train_sreig2img(model, optimizer, noisy_img,loss_function=loss_func2_eig2img ,ele = None,rank=None):
    loss = loss_function(model2 =  model, eigen_image=noisy_img, ele=ele, rank=rank)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

class networkeig2eig(nn.Module):
    def __init__(self,n_chan,chan_embed=48):
        super(networkeig2eig, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

class networkeig2img(nn.Module):
    def __init__(self,n_chan,chan_embed=48):
        super(networkeig2img, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)


    def forward(self, x,ele,rank):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = reconstruct_from_subspace(eigenimages=x,V_r=ele,rank=rank)
        return x



def method_proposed1(noisy_img, ele=None,rank=None):
    n_chan = noisy_img.shape[1]
    print(n_chan)
    model = networkeig2eig(n_chan)
    model = model.to(device)
    max_epoch = 3000  # training epochs
    lr = 0.001  # learning rate
    step_size = 1000  # number of epochs at which learning rate decays
    gamma = 0.5  # factor by which learning rate decays

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(max_epoch)):
        loss = train_sreig2eig_eigloss(model, optimizer, noisy_img.to(device),ele=ele.to(device),rank=rank,)
        scheduler.step()
        if epoch ==1 or epoch==max_epoch-1:
            print(epoch,loss)

    return test_sr_dir(model,ele=ele.to(device),noisy_img=noisy_img.to(device),rank=rank)


def method_proposed2(noisy_img, ele=None,rank=None):
    n_chan = noisy_img.shape[1]
    print(n_chan)
    model = networkeig2eig(n_chan)
    model = model.to(device)
    max_epoch = 3000  # training epochs
    lr = 0.001  # learning rate
    step_size = 1000  # number of epochs at which learning rate decays
    gamma = 0.5  # factor by which learning rate decays

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(max_epoch)):
        loss = train_sr_eig2eig_imgloss(model, optimizer, noisy_img.to(device), loss_function=loss_func2_eig2eig,ele=ele.to(device),rank=rank)
        scheduler.step()
        if epoch ==1 or epoch==max_epoch-1:
            print(epoch,loss)

    return test_sr_dir(model,ele=ele.to(device),noisy_img=noisy_img.to(device),rank=rank)


def method_proposed3(noisy_img, ele=None,rank=None):
    n_chan = noisy_img.shape[1]
    print(n_chan)
    model = networkeig2img(n_chan)
    model = model.to(device)
    max_epoch = 3000  # training epochs
    lr = 0.001  # learning rate
    step_size = 1000  # number of epochs at which learning rate decays
    gamma = 0.5  # factor by which learning rate decays

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(max_epoch)):
        loss = train_sreig2img(model, optimizer, noisy_img.to(device), loss_function=loss_func2_eig2img,ele=ele.to(device),rank=rank)
        scheduler.step()
        if epoch ==1 or epoch==max_epoch-1:
            print(epoch,loss)

    return test_sr_img(model,ele=ele.to(device),noisy_img=noisy_img.to(device),rank=rank)

# def method_proposedeig2eigeigloss(noisy_img, ele=None,rank=None):
#     n_chan = noisy_img.shape[1]
#     print(n_chan)
#     model = networkeig2eig(n_chan)
#     model = model.to(device)
#     max_epoch = 3000  # training epochs
#     lr = 0.001  # learning rate
#     step_size = 1000  # number of epochs at which learning rate decays
#     gamma = 0.5  # factor by which learning rate decays
#
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#
#     for epoch in tqdm(range(max_epoch)):
#         train_sr(model, optimizer, noisy_img.to(device), loss_function=loss_func2_eig2eig,ele=ele,rank=rank)
#         scheduler.step()
#
#     return test_sr_dir(model,ele=ele,noisy_img=noisy_img,rank=rank)
#
# def method_proposedeig2eigimgloss(noisy_img, ele=None,rank=None):
#     n_chan = noisy_img.shape[1]
#     print(n_chan)
#     model = networkeig2eig(n_chan)
#     model = model.to(device)
#     max_epoch = 3000  # training epochs
#     lr = 0.001  # learning rate
#     step_size = 1000  # number of epochs at which learning rate decays
#     gamma = 0.5  # factor by which learning rate decays
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#     for epoch in tqdm(range(max_epoch)):
#         train_sr(model, optimizer, noisy_img.to(device), loss_function=loss_func2_eig2eig,ele=ele,rank=rank)
#         scheduler.step()
#     return test_sr_dir(model,ele=ele,noisy_img=noisy_img,rank=rank)

def noise_level_test_pro(noise_level_list,result_name=None,rank  =4,clean_img = None,data_name = None ):
    result_index = np.zeros(shape=(3,len(noise_level_list)))
    ori_index = np.zeros(shape=(3,len(noise_level_list)))
    log_path = './log_SR_' + data_name + '_' + method_name + '.txt'
    for i,noise_level in enumerate(noise_level_list):
        print(noise_level)
        # cc,noisy_img = mixed_noise(clean_img,gau_noise_level=noise_level)
        # # print(uu.compare_psnr(uu.Nor(clean_img,all=False),cc))
        #
        # noisy_img = add_noise(cc,noise_level=noise_level)
        eigenimages, eigenimages_noisy, element = SR_learn_HSI(GT=clean_img, noisy=noisy_img.detach().numpy(), rank=rank)

        ele = torch.tensor(element).unsqueeze(0)

        eigenimages_noisy = main.arr2ten(eigenimages_noisy)

        result = method_proposed1( noisy_img=eigenimages_noisy,rank=rank,ele=ele)
        # result = torch.permute(result,(0,1,3,2))
        noisy_img = noisy_img.cpu().detach().numpy()
        eva_ori = evaluation(cc,noisy_img).ALL(FSIM=False)
        eva_result = evaluation(cc,result).ALL(FSIM=False)
        # result2=method_proposed2( noisy_img=eigenimages_noisy,rank=rank,ele=ele)
        # eva_result2 = evaluation(cc,result2).ALL(FSIM=False)
        # result3=method_proposed2( noisy_img=eigenimages_noisy,rank=rank,ele=ele)
        # eva_result3 = evaluation(cc,result2).ALL(FSIM=False)
        ori_index[:,i] = np.array(eva_ori)
        result_index[:,i] = np.array(eva_result)
        print(eva_ori,'\n',eva_result)
        # print(eva_result2)
        # print(eva_result3)
        log = str(noise_level)+'\n ORI \n '+str(eva_ori)+'\n PRE \n'+str(eva_result)
        with open(log_path, 'a') as file:
            file.write(('\n' + log))

        # print(psnr(cc,noisy_img))
        # print(psnr(result,clean_img))

    result_mat  = {'ori':ori_index,'result':result_index}
    if result_name is not None:
        savemat(mdict=result_mat,file_name=result_name)


def noise_level_test_pro_poi(noise_level_list,rank  =4,clean_img = None, data_name = None):
    result_index = np.zeros(shape=(3,len(noise_level_list)))
    ori_index = np.zeros(shape=(3,len(noise_level_list)))
    log_path = './log_SR_' + data_name + '_' + method_name + '.txt'
    result_name = './result/poi/' + data_name + 'ori.mat'

    for i, noise_level in enumerate(noise_level_list):
        # print(noise_level,'noise_level')
        path = './result/poisson/' + data_name

        noise_level_name = str(i+1)
        print(noise_level_name,noise_level)
        data_mat = loadmat(path + '/' + noise_level_name + '.mat')
        clean_img = data_mat['ori']

        noisy_img = data_mat['noisy']
        # eigenimages, eigenimages_noisy, element = SR_learn_HSI(GT=clean_img, noisy=noisy_img.detach().numpy(), rank=rank)
        #
        # ele = torch.tensor(element).unsqueeze(0)
        #
        # eigenimages_noisy = main.arr2ten(eigenimages_noisy)
        #
        # result = method_proposed1( noisy_img=eigenimages_noisy,rank=rank,ele=ele)
        # # result = torch.permute(result,(0,1,3,2))
        # noisy_img = noisy_img.cpu().detach().numpy()
        eva_ori = evaluation(clean_img,noisy_img,pr=True).ALL(FSIM=False,mean=True)
        # eva_result = evaluation(cc,result).ALL(FSIM=False)
        # result2=method_proposed2( noisy_img=eigenimages_noisy,rank=rank,ele=ele)
        # eva_result2 = evaluation(cc,result2).ALL(FSIM=False)
        # result3=method_proposed2( noisy_img=eigenimages_noisy,rank=rank,ele=ele)
        # eva_result3 = evaluation(cc,result2).ALL(FSIM=False)
        ori_index[:,i] = np.array(eva_ori)
        # result_index[:,i] = np.array(eva_result)
        # print(eva_ori,'\n' )
        # print(eva_result2)
        # print(eva_result3)
        # log = str(noise_level)+'\n ORI \n '+str(eva_ori)+'\n PRE \n'+str(eva_result)
        # with open(log_path, 'a') as file:
        #     file.write(('\n' + log))

        # print(psnr(cc,noisy_img))
        # print(psnr(result,clean_img))

    result_mat  = {'ori':ori_index,'result':result_index}
    if result_name is not None:
        savemat(mdict=result_mat,file_name=result_name)

if __name__ == '__main__':
    def nosie_sample_poisson(data_dic, data_namelist, ):
        noise_level_list = []
        for i in range(20):
            i =  i+1
            i = i / 10
            noise_level = 10**i
            noise_level_list.append(noise_level)
        print(noise_level_list)


        for data_name in data_namelist:
            print(data_name)
            for i, noise_level in enumerate(noise_level_list):
                clean_img = data_dic[data_name]
                cc =  (Nor(clean_img,all=False))
                noisy_img = add_noise(cc, noise_level=noise_level,noise_type='poiss')
                noisy_img = noisy_img.cpu().detach().numpy()
                result_mat = {'ori': cc, 'noisy':noisy_img}
                result_path = './result/poisson/'+data_name+'/'
                if not os.path.exists(result_path): os.makedirs(result_path)
                noise_level_name =i+1
                print(uu.compare_psnr(cc,tensor2array(noisy_img)),noise_level_name,noise_level )
                result_name  = result_path+str(noise_level_name)+'.mat'
                savemat(mdict=result_mat, file_name=result_name)


    def nosie_sample_gau(data_dic, data_namelist, ):
        noise_level_list = []
        for i in range(20):
            i = i+1
            noise_level_list.append(i*5)

        for i, noise_level in enumerate(noise_level_list):
            print(noise_level)
            for data_name in data_namelist:
                clean_img = data_dic[data_name]
                print(data_name)
                cc, noisy_img = mixed_noise(clean_img, gau_noise_level=noise_level)
                noisy_img = add_noise(cc, noise_level=noise_level)
                noisy_img = noisy_img.cpu().detach().numpy()
                result_mat = {'ori': cc, 'noisy': noisy_img}
                print(uu.compare_psnr(cc,noisy_img))
                result_path = './result/Gaussian/' + data_name + '/'
                if not os.path.exists(result_path): os.makedirs(result_path)
                noise_level_name = noise_level
                result_name = result_path + str(noise_level_name) + '.mat'
                savemat(mdict=result_mat, file_name=result_name)

    import utils.utlis as uu
    from compared import datagen

    data_dic, data_namelist = datagen()
    # nosie_sample_gau(data_dic,data_namelist=data_namelist)
    nosie_sample_poisson(data_dic,data_namelist=data_namelist)
    sys.exit()
    method_name = 'propsoed'
    noise_level_list = []
    for i in range(20):
        i = i + 1
        noise_level_list.append(i * 5)
    noise_level_list_poi = []
    for i in range(20):
        i = i
        i = i / 10
        noise_level_list_poi.append(i)
    # for data_name in data_namelist:
    #     clean_img = data_dic[data_name]
    #     re = data_name + '_' + method_name + '.mat'
    #
    #     rank =4
    #     print(rank, data_name)
    #
    #     # noise_level_test_pro(noise_level_list_poi,result_name=data_name+'_'+method_name+'.mat',rank  =rank,clean_img=clean_img)
    #     noise_level_test_pro_poi(noise_level_list_poi,data_name=data_name)

    list = os.listdir('./result/poisson/dc/')
    for filename  in list:
        data_mat = loadmat('./result/poisson/dc/' +filename   )
        clean_img = data_mat['ori']

        noisy_img = data_mat['noisy']
        print(filename)
        eva_ori = evaluation(clean_img, noisy_img, pr=True).ALL(FSIM=False, mean=True)
