from numba import jit
import numpy as np



#Functions to get the relevance of a word
#_______________________________________________________
#Function to get the context gradient

# v1: number of context (or papers where particpates the word) for each keyword in set one
# v2: context for each keyword in set two
# n: number of keywords in set one
# m: number of keywords in set two
# m_val: ouput  matrix with the context gradient
@jit(nopython=True)
def Get_D2(v1,v2,n,m,m_val):
    for i in range(n):
        for j in range(m):
            m_val[i,j]=v1[i]-v2[j]
          
        #if (i%10==0):
        #    print("Fila ",i)
          
    
    
#get the dijoint global context or the global context grdient.
#D is the diferent in contexts between two words
#Min incidence matrix between two words
#n number of words in the first set
#m number of words in the second set
#I the result importance for each word 
@jit(nopython=True)
def Get_W(D,Min,n,m,I):
    for i in range(n):
        I[i]=0
       
        for j in range(m):
            #if (i!=j):
                I[i]+=D[i,j]*Min[i,j]
            #else:
            #    I[i]+=D[i,i] #is zero
            
        
#Get Vowner score        
#M th incidence matrix of words
#I1 the importance of the words according to the one criterion (for example set of author words )
#I2 the importance of the words according to the second criterion ( for example the keywords Plus set)
@jit(nopython=True)
def getV_fast(M,I1,I2,f,c1,v):
    
    for i in range(f):
        sumaA=0;
        sumaB1=0;
        sumaC=0;
        for j in range(c1):
            if M[i,j]==1 or i==j: 
                sumaA+=I1[j]*I1[j]
                sumaB1+=I1[j]*I2[j]
                sumaC+=I2[j]*I2[j]
            
        sB=sumaB1*sumaB1
        d= sumaA*sumaC-sB
        t= sumaA+sumaC
    
        v[i]=d/(t+1.0e-7)         

@jit(nopython=True)
def get_score(M,Ilayers,nlayers,mlocal,f,c1,v):
    
    for i in range(f):
        mlocal[:,:]=0;        
        for j in range(c1):
            for l1 in range(nlayers):
                for l2 in range(nlayers):
                    if M[i,j]==1 or i==j: 
                        mlocal[l1,l2]+=(Ilayers[l1][j]*Ilayers[l2][j])
                        
        d= np.linalg.det(mlocal);
        t= np.trace(mlocal)    

        v[i]=d/(t+1.0e-7)         

