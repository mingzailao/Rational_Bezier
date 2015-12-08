

from numpy import *
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt
import operator




def c(n,k):
    if k!=0:
        return  reduce(operator.mul, range(n - k + 1, n + 1)) /reduce(operator.mul, range(1, k +1))
    else:
        return 1
def Bernstein(n,k,t):
    return c(n,k)*(t**k)*((1-t)**(n-k))
class Rational_BezierBase:
    def __init__(self,degree=3,weight=np.asarray([
                               [1,1,1,1],
                               [1,1,1,1],
                               [1,1,1,1],
                               [1,1,1,1]
                           ])):
        self.degree=degree
        self.weight=weight
    def Get(self,i,t):
        sum=0;
        for j in range(self.weight.shape[0]):
            sum=sum+self.weight[j]*Bernstein(self.degree,j,t)
        return self.weight[i]*Bernstein(self.degree,i,t)/sum



def Rational_BezierSurface(Points1=np.asarray([[[0,0,0],[1,0,1],[2,0,1.5],[3,0,-1]],
                                              [[0,1,2],[1,1,4],[2,1,2.5],[3,1,0]],
                                              [[0,2,1],[1,2,3],[2,2,2.5],[3,2,2]],
                                              [[0,3,0.5],[1,3,0],[2,3,1],[3,3,1]]]),
                           weight=np.asarray([
                               [1,1,1,1],
                               [1,1,1,1],
                               [1,1,1,1],
                               [1,1,1,1]
                           ]),U=r_[0:1:0.01],V=r_[0:1:0.01]):
    ans=np.zeros([U.size,V.size,3])
    for s in range(U.size):
        u=U[s]
        for t in range(V.size):
            v=V[t]
            sum=0
            tmp=np.zeros([1,3])
            for i in range(Points1.shape[0]):
                for j in range(Points1.shape[1]):
                    sum=sum+weight[i,j]*Bernstein(weight.shape[0],i,u)*Bernstein(weight.shape[1],j,v)
                    tmp=tmp+weight[i,j]*Points1[i,j]*Bernstein(weight.shape[0],i,u)*Bernstein(weight.shape[1],j,v)
            ans[s,t]=tmp/sum
    return ans

ans=Rational_BezierSurface()
x=ans[:,:,0]
y=ans[:,:,1]
z=ans[:,:,2]
x=x.reshape(100,100)
y=y.reshape(100,100)
z=z.reshape(100,100)



fig = plt.figure()
# ax=fig.add_subplot(111,projection='3d')

ax=p3.Axes3D(fig)
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

plt.show()
