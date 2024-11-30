import torch

from util import *

from torch_geometric.nn import GCNConv
import torch.nn.parameter
import torch
import torch.nn.functional as F
import torch.nn as nn

import ot
import random

#l2_dis

def eucdis(x1,x2):
    norm1=torch.sum(x1*x1,dim=1).reshape(-1,1)
    norm2=torch.sum(x2*x2,dim=1).reshape(-1,1)
    norm1=norm1.repeat(1,x2.shape[0])
    norm2=norm2.repeat(1,x1.shape[0])
    return torch.sqrt(torch.relu(norm1+norm2.t()-2*torch.mm(x1,x2.t())))

def eucdis_all(x1,x2):
    norm1=torch.sum(x1*x1,dim=2).unsqueeze(2)
    norm2=torch.sum(x2*x2,dim=2).unsqueeze(2)
    norm1=norm1.repeat(1,1,x2.shape[1])
    norm2=norm2.repeat(1,1,x1.shape[1])
    return torch.sqrt(torch.relu(norm1+norm2.transpose(dim0=1,dim1=2)-2*torch.matmul(x1,x2.transpose(dim0=1,dim1=2)))+1e-10)


def prepare(graph1_e,graph2_e,train_size,l_edge_index, r_edge_index,bt_size=32):
    neia_ts=[]
    neia_ss=[]
    neib_ts=[]
    neib_ss=[]
    len_a=[]
    len_b=[]
    ags=[]
    bgs=[]
    masks=[]
    neia_aps=[]
    neib_aps=[]
    for i in range(len(l_edge_index)):
        
        if(len(graph1_e[l_edge_index[i]])>bt_size):
            nei1=random.sample(graph1_e[l_edge_index[i]],bt_size)
        else:
            nei1=graph1_e[l_edge_index[i]]
        if(len(graph2_e[r_edge_index[i]])>bt_size):
            nei2=random.sample(graph2_e[r_edge_index[i]],bt_size)
        else:
            nei2=graph2_e[r_edge_index[i]]    
        
        #if(len(nei1)<=len(nei2)):
        #    neia=nei1
        #    neib=nei2
        #    ag=0
        #    bg=1
        #else:
        #    neia=nei2
        #    neib=nei1
        #    ag=1
        #    bg=0
        
        neia=nei1
        neib=nei2
        ag=0
        bg=1
        
        
        neia_t=[i for i in neia if i <train_size]
        neia_s=[i-train_size  for i in neia if i>=train_size]
         
        neib_t=[i for i in neib if i<train_size]
        neib_s=[i-train_size for i in neib if i>=train_size]
        
        tmp=torch.zeros(bt_size,bt_size)
        tmp[:len(neia),:len(neib)]=1
        
        neia_aps+=neia+[0]*(bt_size-len(neia))
        neib_aps+=neib+[0]*(bt_size-len(neib))
        
        
        
        
        masks.append(tmp.reshape(1,bt_size,bt_size))
        
        neia_ts.append(neia_t)
        neia_ss.append(neia_s)
        neib_ts.append(neib_t)
        neib_ss.append(neib_s)
        len_a.append(len(neia))
        len_b.append(len(neib))
        ags.append(ag)
        bgs.append(bg)
        
    return neia_ts,neia_ss,neib_ts,neib_ss,len_a,len_b,ags,bgs,torch.LongTensor(neia_aps),torch.LongTensor(neib_aps),torch.cat(masks,dim=0)



#model

class MatchNet(torch.nn.Module):
    def __init__(self,graphx_com,graph1x_dis,graph2x_dis,dim1,dim2,f_dim,x_dim=256):
        super(MatchNet, self).__init__()
        
        
        self.graphxcom=nn.Parameter(torch.randn(graphx_com.shape[0],x_dim))
        self.graph1x=nn.Parameter(torch.randn(graph1x_dis.shape[0],x_dim))
        self.graph2x=nn.Parameter(torch.randn(graph2x_dis.shape[0],x_dim))
        
        self.conv1 = GCNConv(f_dim, dim1)
        self.conv2 = GCNConv(dim1, dim2)
        
        
        

    def forward(self, l_edge_index, r_edge_index,x1_input,x2_input,edge1,edge2,graph1_e,graph2_e,train_size,onlyGCN):
        x1=x1_input
        x2=x2_input
        x1 = F.relu(self.conv1(x1, edge1))
        
        x1 = self.conv2(x1, edge1)
        x1mid=x1/torch.norm(x1,p=2,dim=1).reshape(-1,1)
         
        
        x2 = F.relu(self.conv1(x2,edge2))
        
        x2 = self.conv2(x2,edge2)
        x2mid=x2/torch.norm(x2,p=2,dim=1).reshape(-1,1)
        
        
        neia_ts,neia_ss,neib_ts,neib_ss,len_a,len_b,ags,bgs,neia_aps,neib_aps,masks=prepare(graph1_e,graph2_e,train_size,l_edge_index, r_edge_index)    
        
        x1midmid=torch.cat([self.graphxcom,self.graph1x])
        x2midmid=torch.cat([self.graphxcom,self.graph2x])
        sims=self.calc_sink_bt(graph1_e,graph2_e,neia_ts,neia_ss,neib_ts,neib_ss,len_a,len_b,ags,bgs,neia_aps,neib_aps,masks,x1mid,x2mid,x1midmid,x2midmid,onlyGCN)
        
        return sims,x1mid,x2mid
    
    def calc_sink(self,nei1,nei2,x1,x2,train_size):
        #if(len(nei1)<=len(nei2)):
        #    neia=nei1
        #    neib=nei2
        #    ag=0
        #    bg=1
        #else:
        #    neia=nei2
        #    neib=nei1
        #    ag=1
        #    bg=0
        neia=nei1
        neib=nei2
        ag=0
        bg=1
        neia_t=[i for i in neia if i <train_size]
        neia_s=[i-train_size  for i in neia if i>=train_size]
         
        neib_t=[i for i in neib if i<train_size]
        neib_s=[i-train_size for i in neib if i>=train_size]
        
        if(ag==0):
            ax0=torch.cat([self.graphxcom[neia_t],self.graph1x[neia_s]],dim=0)
            bx0=torch.cat([self.graphxcom[neib_t],self.graph2x[neib_s]],dim=0)
            ax1=x1[neia]
            bx1=x2[neib]
            ax=torch.cat([ax0,ax1],dim=1)
            bx=torch.cat([bx0,bx1],dim=1)
        else:
            bx0=torch.cat([self.graphxcom[neib_t],self.graph1x[neib_s]],dim=0)
            ax0=torch.cat([self.graphxcom[neia_t],self.graph2x[neia_s]],dim=0)
            bx1=x1[neib]
            ax1=x2[neia]
            ax=torch.cat([ax0,ax1],dim=1)
            bx=torch.cat([bx0,bx1],dim=1)
        
        dist=eucdis(ax,bx)
        lena=len(neia)
        lenb=len(neib)
        r=torch.ones(lena)/lena
        c=torch.ones(lenb)/lenb
        p=ot.sinkhorn(r, c, dist,0.5, method='sinkhorn', numItermax=5, stopThr=1e-06) 
        return torch.sum(dist*p)
    
    
    
    def calc_sink_bt(self,graph1_e,graph2_e,neia_ts,neia_ss,neib_ts,neib_ss,lens1,lens2,ags,bgs,neia_aps,neib_aps,masks,x1,x2,x1midmid,x2midmid,onlyGCN,max_pend=1,c=0.5,bt_size=32):
        if(onlyGCN):
            axs=torch.index_select(x1,0,neia_aps).reshape(-1,bt_size,x1.shape[1])
            bxs=torch.index_select(x2,0,neib_aps).reshape(-1,bt_size,x2.shape[1])
            
            
        else:
            axs=torch.cat([torch.index_select(x1,0,neia_aps).reshape(-1,bt_size,x1.shape[1]),torch.index_select(x1midmid,0,neia_aps).reshape(-1,bt_size,x1midmid.shape[1])],dim=2)
            bxs=torch.cat([torch.index_select(x2,0,neib_aps).reshape(-1,bt_size,x2.shape[1]),torch.index_select(x2midmid,0,neib_aps).reshape(-1,bt_size,x2midmid.shape[1])],dim=2)
            #axs=torch.index_select(x1midmid,0,neia_aps).reshape(-1,bt_size,x1midmid.shape[1])
            #bxs=torch.index_select(x2midmid,0,neib_aps).reshape(-1,bt_size,x2midmid.shape[1])
        
        
        
        
        dist=eucdis_all(axs,bxs)
        
            
        dist_all=torch.exp(-dist/c)*(masks+1e-30)
        ori_all=dist*masks
        lens1=torch.LongTensor(lens1)
        lens2=torch.LongTensor(lens2)
        
        p=bt_sinkhorn(dist_all,lens1,lens2,c=c,bt_size=dist_all.shape[0])
        
                
        sum_dist_p=torch.sum(torch.sum(ori_all*p,dim=2),dim=1)
        
        
        return sum_dist_p
    
    
    
#ranking_loss

def ranking(out_dist_exp,label_here,neg_s,rank_l):    
    pos=out_dist_exp[:out_dist_exp.shape[0]//(neg_s+1)]
    neg=out_dist_exp[out_dist_exp.shape[0]//(neg_s+1):].reshape(-1,neg_s)
    return torch.sum(torch.relu(pos.reshape(-1,1)-neg+rank_l))/(neg.shape[0]*neg.shape[1])


#row_sinkhorn

bt_size=32
def bt_sinkhorn(p_lambda,lens1,lens2,c=0.5,numItermax=5,bt_size=32):
    for ep in range(numItermax):
        p_lambda/=torch.sum(p_lambda,dim=2).reshape(bt_size,-1,1).repeat(1,1,p_lambda.shape[2])*lens1.reshape(bt_size,1,1).repeat(1,p_lambda.shape[1],p_lambda.shape[2])+1e-10
        p_lambda/=torch.sum(p_lambda,dim=1).reshape(bt_size,1,-1).repeat(1,p_lambda.shape[1],1)*lens2.reshape(bt_size,1,1).repeat(1,p_lambda.shape[1],p_lambda.shape[2])+1e-10
    return p_lambda
    
    
def calc_sink_row(node1,numnode2,graphxs,graph1_e,graph2_e,max_pend=1,c=0.5):
    #t1=time.time()
    if(len(graph1_e[node1])>bt_size):
        nei1=random.sample(graph1_e[node1],bt_size)
    else:
        nei1=graph1_e[node1]
    
    
    dist_all=[]
    ori_all=[]
    lens1=[]
    lens2=[]
    for i in range(numnode2):
        if(len(graph2_e[i])>bt_size):
            nei2=random.sample(graph2_e[i],bt_size)
        else:
            nei2=graph2_e[i]
        
        if(len(nei1)<=len(nei2)):
            neia=nei1
            neib=nei2
            ag=0
            bg=1
        else:
            neia=nei2
            neib=nei1
            ag=1
            bg=0
        #print(time.time()-t1)
        dist=eucdis(graphxs[ag][neia],graphxs[bg][neib])
        #print(time.time()-t1)
        lena=len(neia)
        lenb=len(neib)
        tmp_dist=torch.zeros(bt_size,bt_size)
        tmp_dist[:dist.shape[0],:dist.shape[1]]+=torch.exp(-dist/c)
        tmp_dist1=torch.zeros(bt_size,bt_size)
        tmp_dist1[:dist.shape[0],:dist.shape[1]]+=dist
        
        dist_all.append(tmp_dist.reshape(1,tmp_dist.shape[0],tmp_dist.shape[1]))
        
        ori_all.append(tmp_dist1.reshape(1,tmp_dist.shape[0],tmp_dist.shape[1]))
        lens1.append(lena)
        lens2.append(lenb)
        
    dist_all=torch.cat(dist_all,dim=0)
    ori_all=torch.cat(ori_all,dim=0)
    lens1=torch.LongTensor(lens1)
    lens2=torch.LongTensor(lens2)
    p=bt_sinkhorn(dist_all,lens1,lens2,c=c,bt_size=dist_all.shape[0])
    sum_dist_p=torch.sum(torch.sum(ori_all*p,dim=2),dim=1)
    
    
    return sum_dist_p
