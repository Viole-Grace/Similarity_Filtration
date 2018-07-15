import numpy as np 
import scipy.stats
from scipy import spatial
from operator import itemgetter
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.weightstats import CompareMeans
import statsmodels.api as sm
#from Data import givenset

# flipkart uses user-user based Cosine Similarity function. Then if one item matches it after a recommendation, that very user is recommended to it.
# Each user has been given 'n' items to evaluate. 
#User that hasnt evaluated an item rates it '0'. Scale is out of 5.
# R here shows sample data. This can be extracted from any Data module with values.
# This can be used as a hybrid filter for a recommendation system as well

#sample array
R = np.array([  
    [2, 3, 0, 1],
    [1, 0, 0, 2],
    [4, 1, 0, 5],
    [5, 0, 0, 4],
    [0, 1, 5, 4],  
  ])
r2=[]
for i in R:
    r1=[]
    for j in i:
        r1.append(j/5)
    r2.append(r1)
R=np.array(r2)

#plt.plot(R)
x=range(1,10)
n=1
for user in R:
    l="User:"+str(n)
    plt.plot(x,user,label = l)
    n+=1
plt.ylabel('Rating')
plt.xlabel('Product Number')
plt.legend()
plt.show()

d={}
for i in range(1,6):
    d[i]=list()

print("\nCosine Similarity : User to User\n")
Ai = []
for i in range(len(R)):
    for j in range(i, len(R)):
        desc= 1-spatial.distance.cosine(R[i],R[j])
        if desc != 1:
            print("Cosine similarity for user ",i+1," and user ",j+1," is : %1.5f"%(desc))
            Ai.append([i+1,j+1,desc])
            l=d.get(i+1)
            l.append([i+1,j+1,desc])
            d[i+1]=l
print ("\nCosine Similarity calculated\n");totval=0
for item in Ai:
    totval=totval+item[2]
    print("CS relevance of user-user : U1 - ",item[0]," U2 - ",item[1]," Similarity - %1.5f"%(item[2]))
print("User-User Similarity : ",totval/len(Ai))

for user in d:
    values=d.get(user)
    if values==list():
        continue
    info=max(values,key=itemgetter(2))
    a=info[0]-1
    b=info[1]-1
    x=range(1,10)
    n=a+1
    l="User:"+str(n)
    plt.plot(x,R[a],label = l)
    n=b+1
    l="User:"+str(n)
    plt.plot(x,R[b],label = l)
    plt.ylabel('Rating')
    plt.xlabel('Product Number')
    plt.legend()
    plt.show()
    
    print("Pearson:",scipy.stats.pearsonr(R[a], R[b]))

    print(CompareMeans(DescrStatsW(R[a]), DescrStatsW(R[b])).summary())
    
    results = sm.OLS(R[b], R[a]).fit()
    print(results.summary())

    for i in range(0,9):
        if R[a][i]==0:
            if R[b][i]>=3:                
                print("product",i+1,"recommended to user",a+1)
    for i in range(0,9):
        if R[b][i]==0:
            if R[a][i]>=3:
                print("product",i+1,"recommended to user",b+1)
