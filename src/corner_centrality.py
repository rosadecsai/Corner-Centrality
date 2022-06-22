
from utilidades import *
import pandas as pd
import operator
import numpy as np
import copy

class Corner_Centrality():
    """ Corner Centrality measure.
    Args:
    *path: path to the database
    *file_conf: configuration file with the number of layers and nodes file name and edges file.
    """

    def __init__(self,path,file_conf):
        self.path = path
        n_f = self.path+file_conf;
        self.get_forlayer=0
        self.scores=[]
        with open(n_f,'rt') as f:
            l = f.readline()
            self.n_layers = int(l)
            l=f.readline() #file with the edges
            name_fedges = self.path+l[:-1]
            #read the information of the nodes in every layer
            self.vertices=[] #list of vertices
            for i in range(self.n_layers):
                l=f.readline()
                if (l[-1]=='\n'):
                    l=l[:-1]
                self.vertices.append(self.__read_vertices(l))
            #list  of incidences matrix
            self.incidences=[]
            for i in range(self.n_layers):
                filas = list(self.vertices[i].keys())
                nnodes1=len(filas)
                for j in range(i,self.n_layers):
                    cols = list(self.vertices[j].keys())
                    nnodes2=len(cols)
                    aux = np.zeros((nnodes1,nnodes2),dtype='uint8')
                    inci = pd.DataFrame(aux,index=filas,columns=cols)
                    self.incidences.append(inci)
            with open(name_fedges,'rt') as f:
                #format: node1 layer1 node2 layer2
                lines =f.readlines()
                for l in lines:
                    aux=l.split()
                    l1=int(aux[0]) 
                    n1=aux[1]
                    l2=int(aux[2])
                    n2=aux[3]
                    pos=self.__getpos(l1,l2) #ravel the position of the matrix 
                    self.incidences[pos].loc[n1,n2]=1

                    

    def set_target_layer(self, target_layer):
        """ 
            Update the target layer.
            Args:
                *target_layer: identification to the new target layer
                
        """
        self.get_forlayer=target_layer

    def __read_vertices(self, name):
        """ 
            Read the name of vertices and  the value initial of importance for every one
            Args:
                *name: file name 
            
        """
        d_out={}
        nn=self.path+name
        with open(nn,'rt') as f:
            #read name of node and weight (initial value of importance)
            lines = f.readlines()
            for l in lines:
                aux = l.split()
                print(aux)
                importance =float(aux[1])
                d_out.update({aux[0]:importance})
        return d_out

        
    def __getpos(self,l1,l2):
        """ 
            Ravel the position of the incidence matrix 
            Args:
                *l1: row layer 
                *l2: column layer 
            *Note :
                Suposse a matrix of incidences matrix with nxn layers
        """
        res =int((l1)/2.0*(l1-1))
        pos = (l1-1)*self.n_layers+(l2-1)-res
        return pos

  
    def save_initial_sorted(self):
        """ 
            Save the nodes sorted by initial value of importance
            
        """
        for i in range(self.n_layers):
            aux = dict(sorted( self.vertices[i].items(), key=operator.itemgetter(1),reverse=True))
            rank_aux = dict({k:i+1  for i,k in enumerate(aux)})  
            nout=self.path+'/initial_sort_L'+str(i)+'.txt'
            with open(nout, 'wt') as f:
                s = sum(aux.values())
                for ll in self.words:
                    k=aux[ll]
                    k=k/s
                    txt=ll+" "+str(k)+" "+str(rank_aux[ll])+'\n'
                    f.write(txt)        
    
    
    
    def getscore(self,debug=False):
        """
        Get the importace of vertice in the target layer
        
        """
        
        pos_forlayer = self.__getpos(self.get_forlayer+1,self.get_forlayer+1)
        self.words = self.incidences[pos_forlayer].index;
        cols = self.words;
        v_target=np.array([self.vertices[self.get_forlayer][i] for i in cols])
        D_target = np.zeros(self.incidences[pos_forlayer].shape,dtype=np.int16)
        D_target_pd = pd.DataFrame(D_target,index=cols,columns=cols)
        n=len(v_target)
        Get_D2(v_target,v_target,n,n,D_target_pd.values)
        if (debug):
            print("Incidence matrix ")
            print(self.incidences[pos_forlayer])
            print("Layer 1  incidence")
            print(D_target_pd)
        v=np.zeros(len(self.words),dtype='float')
        Get_W(D_target_pd.values,self.incidences[pos_forlayer].values,len(self.words),len(self.words),v)
        if (debug):
            print("Suma importancia relativa")
            print(dict({k:v[i] for i,k in enumerate(self.incidences[pos_forlayer].index)}))
        #Normalization
        v= (v-np.mean(v))/np.std(v)
      
        I_layers=[]
        I_layers.append(dict({k:v[i] for i,k in enumerate(self.incidences[pos_forlayer].index)}))
        if (debug):
            print("Capa 0 I",I_layers[0])
           
        del D_target_pd

        #for the rest of layers
        v_layers=[]
        v_layers.append(v_target);
        set_k = set([kk for kk in self.incidences[pos_forlayer].index])
        for i in range(0,self.n_layers):
            if (i!=self.get_forlayer):

                pos_multiplex = self.__getpos(i+1,i+1) 
                if (np.all(self.incidences[pos_multiplex].values==0)): #inter-links?
                    if (i<self.get_forlayer):
                        l1=i;
                        l2=self.get_forlayer
                        pos_ref=self.__getpos(l1+1,l2+1)        
                        cols = self.incidences[pos_ref].index;
                        index=self.incidences[pos_ref].columns
                    else:
                        l1=self.get_forlayer
                        l2=i
                        pos_ref=self.__getpos(l1+1,l2+1)        
                        index = self.incidences[pos_ref].index;
                        cols=self.incidences[pos_ref].columns
                else:#only intra-links
                    pos_ref=pos_multiplex        
                    index = self.incidences[pos_ref].index;
                    cols=self.incidences[pos_ref].columns
                
                set_i = set(cols)
                v_other=np.array([self.vertices[i][j] for j in cols])
                if (set_k!=set_i):#multilayer network
                    v_target=np.array([self.vertices[pos_forlayer][j] for j in index])
                else: #multiplex network
                    v_target=v_other;
                
                D_target = np.zeros(self.incidences[pos_ref].shape,dtype=np.int16)
                D_target_pd = pd.DataFrame(D_target,index=index,columns=cols)
                n=len(v_target)
                m=len(v_other)
                Get_D2(v_target,v_other,n,m,D_target_pd.values)
               
                v=np.zeros(len(self.words),dtype='float') 
                if i<self.get_forlayer:
                    m_aux = np.transpose(self.incidences[pos_ref].values)
                else:
                    m_aux = self.incidences[pos_ref].values    
                Get_W(D_target_pd.values,m_aux,n,m,v)
                if (debug):
                    print("Capa ",i, "v = ",v)
                
                
                v= (v-np.mean(v))/np.std(v)
                #importance of every node in target layer from the layer i
                I_layers.append(dict({k:v[j] for j,k in enumerate(index)}))
                if (debug):
                    print("Capa ",len(I_layers) ," I " ,I_layers[len(I_layers)-1])
                   
                del D_target_pd
        #Get the final importance        
        auxI=[]
        for i in range(len(I_layers)):                               
            aux= np.array([i for i in I_layers[i].values()])
            auxI.append(aux)
        n = len(self.incidences[pos_forlayer].index)
        m = len(self.incidences[pos_forlayer].columns)
        v=np.zeros(len(self.words),dtype='float')
        mlocal = np.zeros((self.n_layers,self.n_layers),'float')
        get_score(self.incidences[pos_forlayer].values,auxI,self.n_layers,mlocal,n,m,v)
        self.v=dict({k:v[i] for i,k in enumerate(self.words)})                         
        
        self.v = dict(sorted(self.v.items(), key=operator.itemgetter(1),reverse=True))
        self.scores.append(self.v)
        if (debug):
            print("Centrallity Corner ")
            print(self.v)
        rank =dict({k:i  for i,k in enumerate(self.v)})     
        #save the results
        nout=self.path+'score_L'+str(self.get_forlayer)+'.txt'    
        with open(nout, 'wt') as f:
            s_v=sum(self.v.values())
            for i,k in self.v.items():
                k=k/s_v
                txt=i+" "+str(k)+'\n'
                f.write(txt)
        nout=self.path+'ranking_L'+str(self.get_forlayer)+'.txt' 
        with open(nout, 'wt') as f:
           for w in self.words:
               txt=w+" "+str(self.v[w])+" "+str(rank[w]+1)+'\n'
               f.write(txt)
    
    def min_scores(self):
        """
        For multiplex networks, with the same nodes in every layer, get the minimum corner centrality values
        for every node across all the layers
        
        """

        self.v=copy.deepcopy(self.scores[0])
        for i in range(1,self.n_layers):
            for k in self.v.keys():
                if (self.v[k]>self.scores[i][k]):
                    self.v[k]=self.scores[i][k]
        self.v = dict(sorted(self.v.items(), key=operator.itemgetter(1),reverse=True))
        nout=self.path+'ranking_min.txt'
        rank =dict({k:i  for i,k in enumerate(self.v)})  
        with open(nout, 'wt') as f:
           for w in self.words:
               txt=w+" "+str(self.v[w])+" "+str(rank[w]+1)+'\n'
               f.write(txt)
   

if __name__ == '__main__': 
    #Example with multiplex centrality
    # cc=Corner_Centrality("../data/exp_1/",'config.txt') 
    # cc.set_target_layer(0)
    # cc.getscore(True)
    # cc.set_target_layer(1)
    # cc.getscore(True)
    # cc.min_scores()
    
    #Example Florentine
    cc_florentine=Corner_Centrality("../data/Florentine_Family/",'config.txt') 
    cc_florentine.set_target_layer(0)
    cc_florentine.getscore(True)
    cc_florentine.set_target_layer(1)
    cc_florentine.getscore(True)
    cc_florentine.min_scores()

    #Example in Author Keyword KeyWordsPlus keywrords multilayer network with interlinks
    #cc_ak=Corner_Centrality("../data/exp_3/",'config.txt') 
    #cc_ak.set_target_layer(0)
    #cc_ak.getscore(False)