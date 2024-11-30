from util import *
from model import *

import argparse
import torch
import random
import torch.utils.data as Data
from torch_geometric.utils import remove_self_loops

parser = argparse.ArgumentParser()

parser.add_argument('--al_len', type=int, default=1118)
parser.add_argument('--e1_path', type=str, default="../data/douban_online_e")
parser.add_argument('--e2_path', type=str, default="../data/douban_offline_e")
parser.add_argument('--f1_path', type=str, default="../data/douban_online_feat.npy")
parser.add_argument('--f2_path', type=str, default="../data/douban_offline_feat.npy")
parser.add_argument('--model_path', type=str, default="best_model_douban")
parser.add_argument('--n1', type=int, default=3906)
parser.add_argument('--n2', type=int, default=1118)
parser.add_argument('--max_degree', type=int, default=32)
parser.add_argument('--train_size', type=int, default=223)
parser.add_argument('--dim1', type=int, default=128)
parser.add_argument('--dim2', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--neg_s', type=int, default=3)
parser.add_argument('--rank_l', type=float, default=0.11)
parser.add_argument('--half_anchors', type=bool, default=True)
parser.add_argument('--load_from_add', type=bool, default=True)
parser.add_argument('--load_models', type=bool, default=True)
parser.add_argument('--all_epoch', type=int, default=75+25*11+1)
parser.add_argument('--onlyGCN_epoch', type=int, default=35)
parser.add_argument('--init_epoch', type=int, default=75)
parser.add_argument('--train_epoch', type=int, default=25)
parser.add_argument('--sp_k', type=int, default=80)
parser.add_argument('--budget', type=int, default=80)
parser.add_argument('--lambdaa', type=float, default=0)
args = parser.parse_args(args=[])

seed_all(seed=0) #set_random_seed

#hyper_parameters
lam=args.lambdaa
model_path=args.model_path
al_len=args.al_len
e1_path=args.e1_path
e2_path=args.e2_path
f1_path=args.f1_path
f2_path=args.f2_path
n1=args.n1
n2=args.n2
max_degree=args.max_degree
train_size=args.train_size
dim1=args.dim1
dim2=args.dim2
lr=args.lr
batch_size=args.batch_size
neg_s=args.neg_s
rank_l=args.rank_l
half_anchors=args.half_anchors
load_from_add=args.load_from_add
load_models=args.load_models
all_epoch=args.all_epoch
init_epoch=args.init_epoch
train_epoch=args.train_epoch
sp_k=args.sp_k
budget=args.budget
onlyGCN_epoch=args.onlyGCN_epoch


eval_epoch=all_epoch-1

#load_data
graph12 = torch.Tensor()

graph1 = torch.Tensor()
graph2 = torch.Tensor()

g1_edges = process_relations(e1_path)
g2_edges = process_relations(e2_path)

graph1.edge_index = torch.from_numpy(g1_edges).long().T
graph2.edge_index = torch.from_numpy(g2_edges).long().T

if(f1_path):
    g1_features=np.load(f1_path)
    graph1.x = torch.FloatTensor(g1_features)
if(f2_path):
    g2_features=np.load(f2_path)
    graph2.x = torch.FloatTensor(g2_features)

graph12.graph1 = graph1
graph12.graph2 = graph2

map_rel=[]
for i in range(al_len):
    map_rel.append([i, i])
    
maplinks = np.array(map_rel)

test_size=al_len-train_size

true_labels = np.ones(maplinks.shape[0])

pos_x_train = maplinks[:train_size, :]
pos_x_test = maplinks[-test_size:, :]
pos_y_train = true_labels[:train_size]
pos_y_test = true_labels[-test_size:]

graph12.pos_x_train = torch.LongTensor(pos_x_train).T
graph12.pos_x_test = torch.LongTensor(pos_x_test).T
graph12.pos_y_train = torch.LongTensor(pos_y_train)
graph12.pos_y_test = torch.LongTensor(pos_y_test)
    
graph12.graph1.edge_index, _ = remove_self_loops(graph12.graph1.edge_index)
graph12.graph2.edge_index, _ = remove_self_loops(graph12.graph2.edge_index)    

#build_index_degree
graph1_e=[[] for i in range(n1)]
graph2_e=[[] for i in range(n2)]

for i in range(graph12.graph1.edge_index.shape[1]):
    s=graph12.graph1.edge_index[0][i].item()
    e=graph12.graph1.edge_index[1][i].item()
    if(s!=e):
        graph1_e[s].append(e)

for i in range(graph12.graph2.edge_index.shape[1]):
    s=graph12.graph2.edge_index[0][i].item()
    e=graph12.graph2.edge_index[1][i].item()
    if(s!=e):
        graph2_e[s].append(e)
        
if not f1_path:
    graph12.graph1.x,graph12.graph2.x=build_degree(graph1_e,graph2_e,n1,n2,max_degree)
    

#build_graphxs_demo

graphx_com=graph12.graph1.x[:train_size]

graph1x_dis=graph12.graph1.x[train_size:]
graph2x_dis=graph12.graph2.x[train_size:]

graph1x=torch.cat([graphx_com,graph1x_dis],dim=0)
graph2x=torch.cat([graphx_com,graph2x_dis],dim=0)

graphxs=[graph1x,graph2x]


#build model&optimizer

model = MatchNet(graphx_com,graph1x_dis,graph2x_dis,dim1,dim2,graph12.graph1.x.shape[1])
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

eval_set=set(random.sample([i for i in range(train_size,al_len)],int(0.1*(al_len-train_size))))

if(onlyGCN_epoch!=-1):
    oFlag=True
    onlyGCN=True
else:
    oFlag=False
    onlyGCN=True



#training_process

add_anchor=[]
cnt_diff=0
best_eval=-100

tmpsim=None
result=0



for epoch in range(all_epoch):
    if(oFlag):
        if(epoch==onlyGCN_epoch):
            onlyGCN=False
        
            for i in model.conv1.parameters():
                i.requires_grad=False
            for i in model.conv2.parameters():
                i.requires_grad=False
                
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    model.train()
    
    add_anchor_ts=torch.LongTensor(add_anchor).t()
    
    train_idxs=torch.cat([graph12.pos_x_train,add_anchor_ts],dim=1)
    if(onlyGCN):
        train_loader = Data.DataLoader(dataset=[i for i in range(train_idxs.shape[1])], batch_size=train_idxs.shape[1], shuffle=True)
    else:
        train_loader = Data.DataLoader(dataset=[i for i in range(train_idxs.shape[1])], batch_size=batch_size, shuffle=True)
    
    loss_all=0
    
    loader_cnt=0
    for idx in train_loader:
        loader_cnt+=1
        pos_here=train_idxs[:,idx]    
        
        
        neg_edge_index = sampling(pos_here,tmpsim,graph12.graph1.x.shape[0],graph12.graph2.x.shape[0],neg_s,20)
        
        cat_here=torch.cat([pos_here,neg_edge_index],dim=1)
        
        label_here=torch.cat([torch.ones(pos_here.shape),-torch.ones(neg_edge_index.shape)/neg_s],dim=1)       
        
        
        out_dist,x1out,x2out=model(cat_here[0],cat_here[1],graph12.graph1.x,graph12.graph2.x,graph12.graph1.edge_index,graph12.graph2.edge_index,graph1_e,graph2_e,train_size,onlyGCN)
        #out_dist=torch.cat(out_dist_raw)
        out_dist_exp=1-torch.exp(-out_dist)
        loss=ranking(out_dist_exp,label_here,neg_s,rank_l)
        
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        loss_all+=loss.item()
        
    with torch.no_grad():
        x1=torch.cat([model.graphxcom,model.graph1x],dim=0)
        x2=torch.cat([model.graphxcom,model.graph2x],dim=0)
        
        if(True):
            tmpsc=torch.mm(x1,x2.t())
            print("struc",evaluate(tmpsc,1,train_size,al_len),evaluate(tmpsc,5,train_size,al_len),evaluate(tmpsc,10,train_size,al_len))
            tmpat=torch.mm(x1out,x2out.t())
            print("neirong",evaluate(tmpat,1,train_size,al_len),evaluate(tmpat,5,train_size,al_len),evaluate(tmpat,10,train_size,al_len))
        x1=torch.cat([x1,x1out],dim=1)
        x2=torch.cat([x2,x2out],dim=1)
        
        tmpsim=torch.mm(x1,x2.t())
        
        print("only emb",evaluate(tmpsim,1,train_size,al_len),evaluate(tmpsim,5,train_size,al_len),evaluate(tmpsim,10,train_size,al_len))
        eval_result=evaluate_eval(tmpsim,1,train_size,al_len,eval_set)
        if(eval_result>best_eval):
            import pickle
            best_eval=eval_result
            
            with open(model_path,"wb") as f:
                pickle.dump(model,f)
            
        
    if(epoch==eval_epoch):
        model.eval()
        
        with torch.no_grad():
            x1=graph12.graph1.x
            x2=graph12.graph2.x
            x1 = F.relu(model.conv1(x1, graph12.graph1.edge_index))
            
            x1 = model.conv2(x1, graph12.graph1.edge_index)          
             
            x1/=torch.norm(x1,p=2,dim=1).reshape(-1,1)
            x2 = F.relu(model.conv1(x2,graph12.graph2.edge_index))
            
            x2 = model.conv2(x2,graph12.graph2.edge_index)
            x2/=torch.norm(x2,p=2,dim=1).reshape(-1,1)
            tmp1=torch.cat([model.graphxcom,model.graph1x],dim=0)
            tmp2=torch.cat([model.graphxcom,model.graph2x],dim=0)
            tmp1=torch.cat([tmp1,x1],dim=1)
            tmp2=torch.cat([tmp2,x2],dim=1)
            graphxs=[tmp1,tmp2]
            sinksim_sim=torch.zeros(n1,n2) 
            for i in tqdm(range(al_len)):
                sinksim_sim[i]=calc_sink_row(i,n2,graphxs,graph1_e,graph2_e,c=0.1)
            for i in range(11):
                result=evaluate(-sinksim_sim*0.1*i+tmpsim*(1-0.1*i),1,train_size,al_len)      
                print("evaluate","epoch",epoch, "loss",loss_all/loader_cnt,"test_acc",result,0.1*i)
    else:
        print("train","epoch",epoch, "loss",loss_all/loader_cnt,"test_acc",result)
    
    if(epoch>init_epoch-1 and epoch%train_epoch==0):
        cnt_diff+=1
        add_anchor=[]
        if(load_models):
            with open(model_path,"rb") as f:
                model=pickle.load(f)
            if(load_from_add):
                best_eval=-100
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        
        with torch.no_grad():
            x1=torch.cat([model.graphxcom,model.graph1x],dim=0)
            x2=torch.cat([model.graphxcom,model.graph2x],dim=0)
            tmp=torch.mm(x1,x2.t())
            tmp[:,:train_size]=-100
            tmp[:train_size]=-100
            evaluate(tmp,1,train_size,al_len)
            add_anchor=generate_anchor_f(tmp,sp_k,budget*cnt_diff,x1out,x2out,model,al_len,train_size,graph1_e,graph2_e)
            right_anchor=0
            for i in add_anchor:
                if(i[0]==i[1]):
                    right_anchor+=1
            
           
            print("adding","epoch",epoch,add_anchor,"right ratio:",right_anchor/(cnt_diff*budget),len(add_anchor))
            if(half_anchors):
                for pair in add_anchor:
                    half=(model.graph1x[pair[0]-train_size]+model.graph2x[pair[1]-train_size])/2
                    model.graph1x[pair[0]-train_size]=half
                    model.graph2x[pair[1]-train_size]=half
    with torch.no_grad():        
        model.graph1x/=torch.norm(model.graph1x,p=2,dim=1).reshape(-1,1)
        model.graph2x/=torch.norm(model.graph2x,p=2,dim=1).reshape(-1,1)
        model.graphxcom/=torch.norm(model.graphxcom,p=2,dim=1).reshape(-1,1)

        
#final_eval

with torch.no_grad():
    with open(model_path,"rb") as f:
        model=pickle.load(f)
    x1=graph12.graph1.x
    x2=graph12.graph2.x
    x1 = F.relu(model.conv1(x1, graph12.graph1.edge_index))
    
    x1 = model.conv2(x1, graph12.graph1.edge_index)          
     
    x1/=torch.norm(x1,p=2,dim=1).reshape(-1,1)
    x2 = F.relu(model.conv1(x2,graph12.graph2.edge_index))
    
    x2 = model.conv2(x2,graph12.graph2.edge_index)
    x2/=torch.norm(x2,p=2,dim=1).reshape(-1,1)
    tmp1=torch.cat([model.graphxcom,model.graph1x],dim=0)
    tmp2=torch.cat([model.graphxcom,model.graph2x],dim=0)
    tmp1=torch.cat([tmp1,x1],dim=1)
    tmp2=torch.cat([tmp2,x2],dim=1)
    graphxs=[tmp1,tmp2]
    sinksim_sim=torch.zeros(n1,n2) 
    for i in tqdm(range(al_len)):
        sinksim_sim[i]=calc_sink_row(i,n2,graphxs,graph1_e,graph2_e,c=0.1)
    sinksim_sim=sinksim_sim*lam-torch.mm(tmp1,tmp2.t())*(1-lam)
print("final_eval",evaluate(-sinksim_sim,1,train_size,al_len),evaluate(-sinksim_sim,5,train_size,al_len),evaluate(-sinksim_sim,10,train_size,al_len))    
