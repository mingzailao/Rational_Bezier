

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



Points1=np.asarray([[[0,0,0],[1,0,1],[2,0,1.5],[3,0,-1]],
                    [[0,1,2],[1,1,4],[2,1,2.5],[3,1,0]],
                    [[0,2,1],[1,2,3],[2,2,2.5],[3,2,2]],
                    [[0,3,0.5],[1,3,0],[2,3,1],[3,3,1]]])
ans=Rational_BezierSurface()
x=ans[:,:,0]
y=ans[:,:,1]
z=ans[:,:,2]
x=x.reshape(100,100)
y=y.reshape(100,100)
z=z.reshape(100,100)



fig = plt.figure()

ax=p3.Axes3D(fig)
ax.plot_surface(x, y, z,rstride=4,cstride=4, color='r')


P0=Points1[:,0].transpose()
P1=Points1[:,1].transpose()
P2=Points1[:,2].transpose()
P3=Points1[:,3].transpose()
Q0=Points1[0,:].transpose()
Q1=Points1[1,:].transpose()
Q2=Points1[2,:].transpose()
Q3=Points1[3,:].transpose()
ax.plot(P0[0],P0[1],P0[2],linewidth=2,color='b')
ax.plot(P1[0],P1[1],P1[2],linewidth=2,color='b')
ax.plot(P2[0],P2[1],P2[2],linewidth=2,color='b')
ax.plot(P3[0],P3[1],P3[2],linewidth=2,color='b')
ax.plot(Q0[0],Q0[1],Q0[2],linewidth=2,color='b')
ax.plot(Q1[0],Q1[1],Q1[2],linewidth=2,color='b')
ax.plot(Q2[0],Q2[1],Q2[2],linewidth=2,color='b')
ax.plot(Q3[0],Q3[1],Q3[2],linewidth=2,color='b')

plt.show()
