import itertools       
import os              
import os.path as osp
import pickle          

from collections import namedtuple  
import numpy as np
import scipy.sparse as sp
import torch           
                       
import torch.nn as nn
import torch.nn.init as init      
import torch.nn.functional as F   
import torch.optim as optim       
import scipy.io as sio
from scipy.sparse import csr_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import tensorflow as tf
import sys

import time         

# Basic Data Settings
col = 64                  # line number of lidar
row = 2250                
# Size of sliding window
win = 2

num_points = col * row
num_pool = int(col / win * row / win) 
# Setting of adjacency matrix
col_side = 1           
row_side = 1           

# Set super parameters
learning_rate = 0.01    
weight_decay = 5e-4     
epochs = 3             
Batch = 3               

# Structure of graph convolution network
class GraphConvolution(nn.Module):
    def __init__(self,input_dim,out_dim,use_bias=True):
        """
        Args:
        --------------
            input_dim:int   
            output_dim:int  
            use_bias:bool,optional 
        """
        super(GraphConvolution,self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim,out_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias',None)     
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)   
        if self.use_bias:
            init.zeros_(self.bias)          

    def forward(self,adjacency,input_feature):
        """
        Args:
        --------
            adjacency:torch.sparse.FloatTensor 
            input_feature:torch.Tensor 
        """
        support = torch.mm(input_feature,self.weight)     
        output = torch.sparse.mm(adjacency,support)       
        if self.use_bias:
            output += self.bias
        return output


class Combine(nn.Module):
    def __init__(self,input_dim,out_dim,use_bias=True):
        """
        Args:
        --------------
            input_dim:int   
            output_dim:int 
            use_bias:bool,optional 
        """
        super(Combine,self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim,out_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias',None)   
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)   
        if self.use_bias:
            init.zeros_(self.bias)          

    def forward(self,input_feature):#adjacency,
        """
        Args:
        --------
            adjacency:torch.sparse.FloatTensor 
            input_feature:torch.Tensor 
        """
        output = torch.mm(input_feature,self.weight)     
        # output = torch.sparse.mm(adjacency,support) 
        if self.use_bias:
            output += self.bias

        return output

class GcnNet(nn.Module):
    """model of GcnNet"""
    def __init__(self,input_dim = 4):         
        super(GcnNet,self).__init__()
        self.gcn1 = GraphConvolution(input_dim,128)
        self.gcn2 = GraphConvolution(132,256)
        self.gcn3 = Combine(260,512)
        self.gcn4 = Combine(516,4)
        
    def forward(self,adjacency,feature,tensor_x3,bing):
        h1 = F.relu(self.gcn1(adjacency,feature))   
        h1 = torch.cat([h1,feature],dim = 1)
        h2 = F.relu(self.gcn2(adjacency,h1))        
        h2 = torch.sparse.mm(bing,h2)
        h2 = torch.cat([h2,tensor_x3],dim = 1)
        h3 = F.relu(self.gcn3(h2))   
        h2 = h3
        h3 = torch.cat([h3,tensor_x3],dim = 1)

        h4 = self.gcn4(h3)  
        logits = h4  

        return logits, h1, h2, h3, h4

def normalization(adjacency):
    """L=D^-0.5*(A+I)*D^-0.5"""
    adjacency += sp.eye(adjacency.shape[0])           
    degree = np.array(adjacency.sum(1))                
    d_hat = sp.diags(np.power(degree,-0.5).flatten()) 
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GcnNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters() , lr = learning_rate , weight_decay = weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



up = 0
down = up + col
left = 0 
right = left + row
result = np.array([up,down,left,right])


order = list(range(0, num_pool))                                      
ORD = np.array(order).reshape(int(col / win) , int(row / win))        
adjacency = np.zeros((num_pool , num_pool))                           
for i in range(ORD.shape[0]):
    for j in range(ORD.shape[1]):
        up_i    = max(0,i - col_side)
        down_i  = min(i + col_side,ORD.shape[0])
        left_j  = max(0,j - row_side)
        right_j = min(j + row_side,ORD.shape[1])
        ord_cut = ORD[up_i:(down_i + 1),left_j:(right_j + 1)]                
        center = ORD[i,j]
        for ii in range(ord_cut.shape[0]):
            for jj in range(ord_cut.shape[1]):
                adjacency[center , ord_cut[ii,jj] ] = 1
                adjacency[ord_cut[ii,jj] , center ] = 1
        adjacency[center , center] = 0
              
adjacency_train = adjacency
adjacency_train = csr_matrix(adjacency_train)
normalize_adjacency2 = normalization(adjacency_train)   
indices2 = torch.from_numpy(np.asarray([normalize_adjacency2.row,normalize_adjacency2.col]).astype('int64')).long()  
values2 = torch.from_numpy(normalize_adjacency2.data.astype(np.float32))
tensor_adjacency2 = torch.sparse.FloatTensor(indices2,values2,(num_pool,num_pool)).to(device) 

order_all = list(range(0, num_points))
ORD_all = np.array(order_all).reshape(col ,row)  
feature_order = np.zeros((int(col / win) * int(row / win) , 4))      
a = 0
for i in range(int(col / win)):
    for j in range(int(row / win)):
        c = ORD_all[(2*i):(2*i + 2),(2*j):(2*j + 2)]
        feature_order[a,0] = c[0][0]
        feature_order[a,1] = c[0][1]
        feature_order[a,2] = c[1][0]
        feature_order[a,3] = c[1][1]
        a = a + 1


bing_sparse_row = np.zeros((1 , num_points))  
bing_sparse_row = bing_sparse_row[0]
bing_sparse_col = np.zeros((1 , num_points))  
bing_sparse_col = bing_sparse_col[0]
bing_sparse_data= np.zeros((1 , num_points))  
bing_sparse_data= bing_sparse_data[0]
a = 0
for i in range(num_pool):
    for j in range(feature_order.shape[1]):
        bing_sparse_row[a] = feature_order[i,j] 
        bing_sparse_col[a] = i                  
        bing_sparse_data[a]= 1                  
        a = a + 1
bing = sp.coo_matrix((bing_sparse_data,(bing_sparse_row,bing_sparse_col)),shape=(num_points,int(num_points / win / win)))
indices = torch.from_numpy(np.asarray([bing.row,bing.col]).astype('int64')).long()   
values = torch.from_numpy(bing.data.astype(np.float32))
bing = torch.sparse.FloatTensor(indices,values,(num_points,int(num_points / win / win))).to(device) 

path = "E:\\SPGCN\\data\\p00\\XYZLR" 
files = os.listdir(path) 
result_path = "E:\\SPGCN\\data\\p00\\result2"

def select_cut(col,row,win,result,file):
    position = path +'\\'+ file
    p     = sio.loadmat(position)

    R     = p['R']
    X     = p['points_X']
    Y     = p['points_Y']
    Z     = p['points_Z']
    FF    = p['points_F']
    label = p['points_label']

    cut_label = label[result[0]:result[1],result[2]:result[3]].copy()
    a = 0    # 计数用
    one_hot = np.zeros((1 , cut_label.shape[0] * cut_label.shape[1]))        
    for i in range(cut_label.shape[0]):
        for j in range(cut_label.shape[1]):

            if cut_label[i,j] == 0:
                one_hot[0,a] = 0              
            elif cut_label[i,j] > 0:
                one_hot[0,a] = cut_label[i,j]         
            a = a + 1
    one_hot_train = one_hot

    a = 0
    mask = np.ones((1,cut_label.shape[0] * cut_label.shape[1]))
    for i in range(cut_label.shape[0]):
        for j in range(cut_label.shape[1]):
            if one_hot[0,a] == 0:  
                mask[0,a] = 0
            a = a + 1
    mask = mask[0]
    mask_train = [mask == 1][0]

    order_all = list(range(0, cut_label.shape[0] * cut_label.shape[1]))
    ORD_all = np.array(order_all).reshape(cut_label.shape[0] ,cut_label.shape[1])   

    cut_R = R[result[0]:result[1],result[2]:result[3]]
    cut_X = X[result[0]:result[1],result[2]:result[3]]
    cut_Y = Y[result[0]:result[1],result[2]:result[3]]
    cut_Z = Z[result[0]:result[1],result[2]:result[3]]
    cut_F = FF[result[0]:result[1],result[2]:result[3]]

    feature = np.zeros((col * row , 4))                     
    a = 0
    for i in range(cut_label.shape[0]):
        for j in range(cut_label.shape[1]):
            feature[a,0] = cut_X[i,j]
            feature[a,1] = cut_Y[i,j]
            feature[a,2] = cut_Z[i,j]
            feature[a,3] = cut_F[i,j]          
            a = a + 1
    feature_combine = feature

    cut_pool_X = np.zeros((int(col / win) , int(row / win)))
    cut_pool_Y = np.zeros((int(col / win) , int(row / win)))
    cut_pool_Z = np.zeros((int(col / win) , int(row / win)))
    cut_pool_F = np.zeros((int(col / win) , int(row / win)))
    
    a = 0
    for i in range(int(col / win)):
        for j in range(int(row / win)):
            a = a + 1
            exist = (cut_R[(2*i):(2*i + 2),(2*j):(2*j + 2)] != 0)
            if exist.any():
                cut_pool_x = cut_X[(2*i):(2*i + 2),(2*j):(2*j + 2)]
                cut_pool_X[i,j] = np.mean(cut_pool_x[exist])
                cut_pool_y = cut_Y[(2*i):(2*i + 2),(2*j):(2*j + 2)]
                cut_pool_Y[i,j] = np.mean(cut_pool_y[exist])
                cut_pool_z = cut_Z[(2*i):(2*i + 2),(2*j):(2*j + 2)]
                cut_pool_Z[i,j] = np.mean(cut_pool_z[exist])
                cut_pool_f = cut_F[(2*i):(2*i + 2),(2*j):(2*j + 2)]
                cut_pool_F[i,j] = np.mean(cut_pool_f[exist])
            else:
                cut_pool_X[i,j] = 0
                cut_pool_Y[i,j] = 0
                cut_pool_Z[i,j] = 0
                cut_pool_F[i,j] = 0
                                    
    feature_pool = np.zeros((int(col / win) * int(row / win) , 4))  
    a = 0
    for i in range(int(col / win)):
        for j in range(int(row / win)):
            feature_pool[a,0] = cut_pool_X[i,j]
            feature_pool[a,1] = cut_pool_Y[i,j]
            feature_pool[a,2] = cut_pool_Z[i,j]
            feature_pool[a,3] = cut_pool_F[i,j]          
            a = a + 1
    feature_train = feature_pool

    tensor_x2 = torch.from_numpy(feature_train).to(device)
    tensor_x2 = tensor_x2.to(torch.float32)

    tensor_x3 = torch.from_numpy(feature_combine).to(device)
    tensor_x3 = tensor_x3.to(torch.float32)

    one_hot_train = one_hot_train[0]
    tensor_y2 = torch.from_numpy(one_hot_train).to(device) 

    tensor_train_mask2 = torch.from_numpy(mask_train).to(device)
    train_y = tensor_y2[tensor_train_mask2]

    return tensor_adjacency2, tensor_x2, tensor_x3,tensor_train_mask2, train_y, tensor_y2


def test(mask,logits):
    model.eval()
    with torch.no_grad():                         
        predict = logits.max(1)[1]                 
        predict_write = np.array(predict).reshape(col ,row)
        predict_y = predict[mask]
        accuracy = torch.eq(predict_y,tensor_y2[mask]).float().mean()
    return accuracy, predict_write

model.train()
loss_history = []
train_acc_history = []
time_history = []


for epoch in range(epochs):
    a = 0
    for file in files[0:3]: 
        head = file[0:6]
        tensor_adjacency2, tensor_x2, tensor_x3, tensor_train_mask2, train_y, tensor_y2 = select_cut(col,row,win,result,file)    
        
        for batch in range(Batch):

            tic = time.time()
            logits, h1, h2, h3, h4 = model(tensor_adjacency2,tensor_x2,tensor_x3,bing)
            toc = time.time()
            time_once = toc - tic
            print(time_once)

            train_mask_logits = logits[tensor_train_mask2]   

            train_acc, predict_write = test(tensor_train_mask2,logits)
           
            train_acc_history.append(train_acc.item())
   
            train_y = train_y.long() 
            loss = criterion(train_mask_logits,train_y)      

            optimizer.zero_grad()                            

            loss_history.append(loss.item())

            loss.backward()
                               
            optimizer.step()                                 
        
            print("Epoch {:02d} File {:04d} Batch {:04d}: Loss {:.4f}, TrainAcc {:.4f}".format(
                epoch, a, batch, loss.item(), train_acc.item()
            ))
            a = a + 1

        result_write = result_path +'\\'+ 'result_' + head + '.mat'
        scipy.io.savemat(result_write, {'predict_write': predict_write})

       
scipy.io.savemat('E:/SPGCN/code/loss&acc.mat', {'loss_history': loss_history, 'train_acc_history': train_acc_history})

plt.plot(train_acc_history)
plt.grid(True)   
plt.axis('tight') 
plt.xlabel('batch')
plt.ylabel('acc')
plt.title('Train Accurancy')
plt.show()

plt.plot(loss_history)
plt.grid(True) 
plt.axis('tight') 
plt.xlabel('batch')
plt.ylabel('loss')
plt.title('Loss History')
plt.show()

plt.plot(train_acc_history)
plt.grid(True)  
plt.axis('tight') 
plt.xlabel('batch')
plt.ylabel('acc')
plt.title('Train Accurancy')
plt.show()