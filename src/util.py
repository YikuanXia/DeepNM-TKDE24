import torch
import numpy as np
import codecs
import ot
from model import *
import random
from tqdm import tqdm
#set_seed

def seed_all(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
#read_edges

def process_relations(rel_file):

    fi = codecs.open(rel_file, 'r', encoding='utf-8', errors='ignore')

    total_line = parser_error =node_error =0

    rel_list = set()

    for line in fi:
        total_line = total_line + 1

        sub_lines = line.split()

        if len(sub_lines)==0:
            parser_error = parser_error+1
            continue

        from_node = int(sub_lines[0].lower())
        to_node = int(sub_lines[1].lower())

        rel_list.add((from_node, to_node))
        rel_list.add((to_node, from_node))

        if total_line %50000 ==0:
            print("%d lines is processed, lines %d"
                        %(total_line, len(rel_list)))

    print("Total lines is %d,  Parser errors is %d, User Error is %d" % ( total_line,  parser_error, node_error))

    fi.close()

    return np.array(list(rel_list))

#build_degree_onehot

def build_degree(graph1_e,graph2_e,n1,n2,max_degree):
    f1=torch.zeros(n1,max_degree)
    f2=torch.zeros(n2,max_degree)
    for i in range(n1):
        f1[i][min(max_degree,len(graph1_e[i]))-1]=1
    for i in range(n2):
        f2[i][min(max_degree,len(graph2_e[i]))-1]=1
    return f1,f2


#naive_sinkhorn

def calc_sink(nei1,nei2,graphxs):
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
    dist=eucdis(graphxs[ag][neia],graphxs[bg][neib])
    lena=len(neia)
    lenb=len(neib)
    r=torch.ones(lena)/lena
    c=torch.ones(lenb)/lenb
    p=ot.sinkhorn(r, c, dist, 0.5, method='sinkhorn', numItermax=5, stopThr=1e-06)    
    return torch.sum(dist*p)


#evaluate_func

def evaluate(S,topk,train_size,al_len):
    
    S_tmp=S.clone()
    S_tmp[:,:train_size]=-100
    values,indices=torch.topk(S_tmp,topk)
    true=0
    for i in range(al_len):
        if(i>train_size):
            if(i in indices[i]):
                true+=1
    return true/(al_len-train_size)


def evaluate_eval(S,topk,train_size,al_len,eval_set):
    
    S_tmp=S.clone()
    S_tmp[:,:train_size]=-100
    values,indices=torch.topk(S_tmp,topk)
    true=0
    for i in range(al_len):
        if(i in eval_set):
            if(i in indices[i]):
                true+=1
    return true/len(eval_set)


#generate_anchor

def generate_anchor(tmp,budget,anchor_size,x1out,x2out,model,al_len,train_size,graph1_e,graph2_e):
    with torch.no_grad():
        values,indices=torch.topk(tmp,budget,dim=1)  
        value_dict=dict()
        
        for i in tqdm(range(train_size,al_len)):
            value_dict[i]=[]
            for j in indices[i]:
                value_dict[i].append((j,model.calc_sink(graph1_e[i],graph2_e[j],x1out,x2out,train_size)))
        
        value_dict1=[]
        for i in range(train_size,al_len):
            value_dict[i].sort(key=lambda x:x[1])
            value_dict1.append(([i,value_dict[i][0][0].item()],value_dict[i][0][1]))
        
        value_dict1.sort(key=lambda x:x[1])
        add_anchor=[]
        tabu0=[]
        tabu1=[]
        anchor_cnt=0
        for i in value_dict1:
            
            if(i[0][0] not in tabu0 and i[0][1] not in tabu1):
                add_anchor.append(i[0])
                anchor_cnt+=1
                tabu0.append(i[0][0])
                tabu1.append(i[0][1])
                if(anchor_cnt==anchor_size):
                    return add_anchor
        return add_anchor


#generate anchor fast


def generate_anchor_f(tmp,budget,anchor_size,x1out,x2out,model,al_len,train_size,graph1_e,graph2_e):
    with torch.no_grad():
        values,indices=torch.topk(tmp,budget,dim=1)  
        value_dict=dict()
        
        left=torch.LongTensor([[i]*budget for i in range(train_size,al_len)]).reshape(-1)
        right=indices[train_size:al_len].reshape(-1)
        
        
        neia_ts,neia_ss,neib_ts,neib_ss,len_a,len_b,ags,bgs,neia_aps,neib_aps,masks=prepare(graph1_e,graph2_e,train_size,left,right) 
        
        
        x1midmid=torch.cat([model.graphxcom,model.graph1x])
        x2midmid=torch.cat([model.graphxcom,model.graph2x])
        
        sims=model.calc_sink_bt(graph1_e,graph2_e,neia_ts,neia_ss,neib_ts,neib_ss,len_a,len_b,ags,bgs,neia_aps,neib_aps,masks,x1out,x2out,x1midmid,x2midmid,False).reshape(-1,budget)
        
        
        
        for i in tqdm(range(train_size,al_len)):
            sims_h=sims[i-train_size].numpy().tolist()
            ind_h=indices[i].numpy().tolist()
            value_dict[i]=[(ind_h[j],sims_h[j]) for j in range(len(sims_h)) ]
        
                
        value_dict1=[]
        for i in range(train_size,al_len):
            value_dict[i].sort(key=lambda x:x[1])
            value_dict1.append(([i,value_dict[i][0][0]],value_dict[i][0][1]))
        
        value_dict1.sort(key=lambda x:x[1])
        
        add_anchor=[]
        tabu0=[]
        tabu1=[]
        anchor_cnt=0
        for i in value_dict1:
            
            if(i[0][0] not in tabu0 and i[0][1] not in tabu1):
                add_anchor.append(i[0])
                anchor_cnt+=1
                tabu0.append(i[0][0])
                tabu1.append(i[0][1])
                if(anchor_cnt==anchor_size):
                    return add_anchor
        return add_anchor
    
#negative_sampling

def sampling(pos_here,sims,numnode1,numnode2,neg_s,outrage):
    result_s=[[],[]]    
    if(sims==None):
        all_node2=[i for i in range(numnode2)]
        for i in range(pos_here.shape[1]):
            sam=random.sample(all_node2,neg_s+5)
            if(pos_here[1][i].item() in sam):
                sam.remove(pos_here[1][i])
            result_s[1]+=sam[:neg_s]
            result_s[0]+=[pos_here[0][i].item()]*neg_s
    else:
        values,indices=torch.topk(sims,neg_s+outrage)
        for i in range(pos_here.shape[1]):
            sam=random.sample(indices[pos_here[0][i]].numpy().tolist(),neg_s+5)
            if(pos_here[1][i].item() in sam):
                sam.remove(pos_here[1][i].item())
            result_s[1]+=sam[:neg_s]
            result_s[0]+=[pos_here[0][i].item()]*neg_s
    return torch.LongTensor(result_s)
            
            
         
    
