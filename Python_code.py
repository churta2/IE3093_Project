import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import random
import copy
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression



batch_size=10


# If already generate a file with the feautures. 

df=pd.read_csv('Train.csv')  
Train_matrix=np.array(df)[:,1:7]


# Function to Generate feature vectors and actual slopes and intercepts.

def simulated_graph(params):
    
    
    
    G = nx.Graph()
    N=int(params[0])
    n=list(range(N))
    Rate=params[1]
    
    size_edges = list(np.random.poisson(Rate, N))
    
    
    

    
    edges=set({})
    
    for i in range(N):
        
        copy_n=copy.deepcopy(n)
        copy_n.remove(i)
        sampled=random.sample(copy_n, size_edges[i])
        new_edges=set(list(itertools.product([i],sampled)))
        edges=edges.union(new_edges)
    
    
    
    G.add_nodes_from(n)
    G.add_edges_from(list(edges))
    

    n_vaccines=int(np.ceil(params[3]*N))
    
    tested=np.random.uniform(0.35,0.6,1)[0]
    n_infected=int(np.ceil(params[2]*tested*N))
    
    n_batches=max(min(int(np.ceil(n_vaccines/batch_size)),5),2)
    
    
    matrix_X=np.zeros((n_batches,2))
    matrix_X[:,0]=list(range(1,n_batches+1))
    
    
    get_infected=set(random.sample(n,n_infected))
    
    get_vaccine=set({})
    
    
    for i in range(n_batches):
        
        get_vaccine=get_vaccine.union(set(random.sample(n,batch_size)))
    
        list_vaccinated=list(get_vaccine)
        
        benefit=set({})
        for k in list_vaccinated:
            
            benefit=benefit.union(set(nx.bfs_tree(G, source=k, depth_limit=1).nodes))
            
        
        reduction_count=get_infected.difference(benefit)
        matrix_X[i,1]=(n_infected-len(reduction_count))/(n_infected)
        
            
        
     
    slope, intercept, r, p, std_err = stats.linregress(matrix_X[:,0], matrix_X[:,1])  
    
   
        
    
    
    H = nx.Graph()
    n=list(range(N))
    Rate=params[1]+1
    
    size_edges = list(np.random.poisson(Rate, N))
    

    
    edges=set({})
    
    for i in range(N):
        
        copy_n=copy.deepcopy(n)
        copy_n.remove(i)
        sampled=random.sample(copy_n, size_edges[i])
        new_edges=set(list(itertools.product([i],sampled)))
        edges=edges.union(new_edges)
    
    
    
    H.add_nodes_from(n)
    H.add_edges_from(list(edges))
    
    
    betwness=(nx.betweenness_centrality(H))
    
    avg_between=sum([betwness[i ]for i in betwness])/N
    
    
    return np.array([N,avg_between,params[2],n_vaccines,slope,intercept])


# Function to Generate the Data points

def simulated_graph_points(params):
    
    G = nx.Graph()
    N=int(params[0])
    n=list(range(N))
    Rate=params[1]
    
    size_edges = list(np.random.poisson(Rate, N))
    
    
    

    
    edges=set({})
    
    for i in range(N):
        
        copy_n=copy.deepcopy(n)
        copy_n.remove(i)
        sampled=random.sample(copy_n, size_edges[i])
        new_edges=set(list(itertools.product([i],sampled)))
        edges=edges.union(new_edges)
    
    
    
    G.add_nodes_from(n)
    G.add_edges_from(list(edges))
    

    n_vaccines=int(np.ceil(params[3]*N))
    
    tested=np.random.uniform(0.35,0.6,1)[0]
    n_infected=int(np.ceil(params[2]*tested*N))
    
    n_batches=max(min(int(np.ceil(n_vaccines/batch_size)),5),2)
    
    
    matrix_X=np.zeros((n_batches,2))
    
    
    
    get_infected=set(random.sample(n,n_infected))
    
    get_vaccine=set({})
    
    
    for i in range(n_batches):
        
        
        actual_vaccines=int(np.ceil(batch_size*np.random.normal(1,0.25,1)[0]))
        
        
        get_vaccine=get_vaccine.union(set(random.sample(n,actual_vaccines)))
        matrix_X[i,0]=len(get_vaccine)/batch_size
    
        
        list_vaccinated=list(get_vaccine)
        
        benefit=set({})
        for k in list_vaccinated:
            
            benefit=benefit.union(set(nx.bfs_tree(G, source=k, depth_limit=1).nodes))
            
        
        reduction_count=get_infected.difference(benefit)
        matrix_X[i,1]=(n_infected-len(reduction_count))/(n_infected)
        
            
        
    
    
    
    return matrix_X


# Parameters for generation of Contagion cascades.
# The last 3 rows correspond to the test set




Data=np.zeros((103,4))
Data[:,0]=np.random.randint(low=80,high=800, size=103)  #Size of the Network
Data[:,1]=np.random.randint(low=0,high=3, size=103)   # Lamba Poisson Distr  .
Data[:,2]=np.random.uniform(0.08,0.35,103)              #Positivity Test
Data[:,3]=np.random.uniform(0.2,0.6,103)                #PIC




def save_dataset(Data):

    
    Train_matrix=np.zeros((103,6))
    
    for t in range(103):
        
        print(t)
        
        Train_matrix[t,:]=simulated_graph(Data[t,:])
    
    DF = pd.DataFrame(Train_matrix)
     
    # save the dataframe as a csv file
    DF.to_csv("Train.csv")
    




def save_dataset_points(Data):
    
    Points_data_set=np.zeros((1,2))

    
    for t in range(103):
        
        Points_data_set=np.concatenate((Points_data_set,simulated_graph_points(Data[t,:])), axis=0)
    
    
    
    DF = pd.DataFrame(Points_data_set)
     
    # save the dataframe as a csv file
    DF.to_csv("Data_points.csv")
        
    



### This function returns the optimal solution of the problem when the Cost vector is known. 
### In paper's notation, this function returns  w^*(c)


def opt_with_known_parameter(Beta):
    
    
    N=np.array([Data[-1,0],Data[-1,0],Data[-2,0],Data[-2,0],Data[-3,0],Data[-3,0]])   
    Beta=np.multiply(N,Beta)        
    m=gp.Model()
    x=m.addMVar(6, lb=0,vtype=GRB.INTEGER) 
    e=np.array([1,0,1,0,1,0])

    C=10
    
    m.addConstr(e@x<=C) 
    
    slopes=[0,2,4]
    intercepts=[1,3,5]
    
    for i in slopes:
        m.addConstr(x[i]<=C*0.5)
        

    for i in intercepts:
        
        m.addConstr(x[i]==1)
 
    
   
    m.setObjective(Beta@x,GRB.MINIMIZE)
    # m.write('master.lp')
    m.optimize()
    
    
    x_v=[]
    for i in range(6):
        x_v.append(m.getVars()[i].x)
    
    x_v=np.array(x_v)
    
        
    return x_v
    
    
    
# Actual Betas and Actual Optimal Solutions

Beta=np.array([-Train_matrix[-1,4],-Train_matrix[-1,5],-Train_matrix[-2,4],-Train_matrix[-2,5],-Train_matrix[-3,4],-Train_matrix[-3,5]])    
x_star=opt_with_known_parameter(Beta)



# Reshape Feature matrix to be use in the SPO framework 

    
where=random.sample(list(range(100)), 50)
D1=Train_matrix[where,0:4]
K1=Train_matrix[where,4:6]

where=random.sample(list(range(100)), 50)

D2=Train_matrix[where,0:4]
K2=Train_matrix[where,4:6]

where=random.sample(list(range(100)), 50)

D3=Train_matrix[where,0:4]
K3=Train_matrix[where,4:6]



new_X=np.concatenate((D1,D2),axis=1)
new_X=np.concatenate((new_X,D3),axis=1)


new_Beta=np.concatenate((K1,K2),axis=1)
new_Beta=np.concatenate((new_Beta,K3),axis=1)




# Function that Calculate l_SPO_+ for each observation


def loss_SPO_plus(x_star,Beta, Theta,x_obs):

  

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        
        with gp.Model(env=env) as m:
            
            
            # m=gp.Model()
            x=m.addMVar(6, lb=0,vtype=GRB.INTEGER) 
            e=np.array([1,0,1,0,1,0])
            C=10
            
            
            m.addConstr(e@x==C) 
            
            slopes=[0,2,4]
            intercepts=[1,3,5]
            
            for i in slopes:
                m.addConstr(x[i]<=C*0.5)
                

            for i in intercepts:
                
                m.addConstr(x[i]==1)
         
            
            Beta_hat=-Theta@x_obs

            
            m.setObjective(Beta@x -2*Beta_hat@x,GRB.MAXIMIZE) 
            # m.write('master.lp')
            m.optimize()
            
            
            lso_plus=m.objVal+2*Beta_hat@x_star
            

    return (lso_plus,Beta_hat)



##  In this section we solve for the Betas that minimize the LSO_plus
##  I WAS TRYING TO USE SUBGRADIENTS BUT IT DID NOT WORK
##  INSTEAD I'M USING SIMULATED ANNEALING. 


ERM=100000

#Initial Values


Theta=np.array([[-2.12156094e-04, -3.03922153e+00,  3.20705495e-02,  1.64471532e-05]*3,
                [3.23558652e-05,  4.52328308e+00,  1.21866415e-01, -1.39586984e-04]*3,
                [-2.12156094e-04, -3.03922153e+00,  3.20705495e-02,  1.64471532e-05]*3,
                [ 3.23558652e-05,  4.52328308e+00,  1.21866415e-01, -1.39586984e-04]*3,
                [-2.12156094e-04, -3.03922153e+00,  3.20705495e-02,  1.64471532e-05]*3,
                [ 3.23558652e-05,  4.52328308e+00,  1.21866415e-01, -1.39586984e-04]*3])


sd=np.array([[1e-2,2,2e-1,2e-4]*3,
            [1e-2,2,2e-1,2e-4]*3,
            [1e-2,2,2e-1,2e-4]*3,
            [1e-2,2,2e-1,2e-4]*3,
            [1e-2,2,2e-1,2e-4]*3,
            [1e-2,2,2e-1,2e-4]*3])


F_theta=Theta.reshape((1, 72))
F_sd=sd.reshape((1, 72))


Theta_star=Theta


#++++SIMULATED ANNEALING

for u in range(10000):
    
      
    F_theta=Theta_star.reshape((1, 72))
    F_sd=sd.reshape((1, 72))


    new_Theta=np.array([np.random.normal(F_theta[0,i],F_sd[0,i],1)[0] for i in range(72)])

    
    new_Theta=new_Theta.reshape((6,12))

    
    comp=sum([loss_SPO_plus(x_star,new_Beta[i,:],new_Theta, new_X[i,:])[0] for i in range(50)])
    
    if comp<ERM:
        
        ERM=comp
        Theta_star=new_Theta
        print("Improving at iteration N: " + str(u) + "   value " +str(ERM))
        
  

    
        
    
### ===============  LSE ESTIMATION ==========  

X=Train_matrix[:,0:4]
Y=Train_matrix[:,4]
reg = LinearRegression().fit(X, Y)


X=Train_matrix[:,0:4]
Y=Train_matrix[:,5]
reg0= LinearRegression().fit(X, Y)



B1=Train_matrix[100,0:4]@reg.coef_+reg.intercept_
B0=Train_matrix[100,0:4]@reg0.coef_+reg0.intercept_
B2=Train_matrix[101,0:4]@reg.coef_+reg.intercept_
B02=Train_matrix[101,0:4]@reg0.coef_+reg0.intercept_
B3=Train_matrix[102,0:4]@reg.coef_+reg.intercept_
B03=Train_matrix[102,0:4]@reg0.coef_+reg0.intercept_

np.array([B1, B0, B2, B02, B3, B03]) ## BETA _HAT WITH LSE



### ===============  SPO ESTIMATION ==========  

Matrix_test=np.concatenate((Train_matrix[100,0:4],Train_matrix[101,0:4], Train_matrix[102,0:4]))



-Theta_star@Matrix_test  ## BETA _HAT WITH SPO


