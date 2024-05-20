import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

import sys
import time

# Coloque aqui o seu número USP
nusp = 13684130

nusp_str=str(nusp).replace("0","")

magic_ilong = int(nusp_str)
magic_ishort = int(str(magic_ilong)[:2])

# Ultimo digito não nulo do nusp
magic_int = int(list(set(nusp_str))[-1])


print("magic_ilong  =", magic_ilong)
print("magic_ishort =", magic_ishort)
print("magic_int =", magic_int)
#####################################################################

#Função
def f(x,y):
  # c=1
  return x**2+y**2 - 1.0


# Amostra de pontos no plano
x = np.arange(-2.0, 2.0, 0.01)
y = np.arange(-2.0, 2.0, 0.01)
X, Y = np.meshgrid(x, y)

# Funcão
Z = f(X,Y)

#Gráfico
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, [0.0])
#ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Curva de $f(x,y)=0$')
ax.axis('equal')

plt.show()
###############################################

def f1(x,y):
  return x**2+y**2 -1

def f2(x,y):
  return x**2/2+3*y**2 - 2

# Amostra de pontos no plano
x = np.arange(-3.0, 3.0, 0.01)
y = np.arange(-3.0, 3.0, 0.01)
X, Y = np.meshgrid(x, y)

# Funcão
Zf1 = f1(X,Y)
Zf2 = f2(X,Y)

#Gráfico
fig, ax = plt.subplots()
CSf = ax.contour(X, Y, Zf1, [0.0])
CSg = ax.contour(X, Y, Zf2, [0.0], colors=('r'))
ax.set_title('Curvas de $f_1(x,y)=0$ e $f_2(x,y)=0$')
ax.axis('equal')

plt.show()

############################################################
#AQUI COMEÇA O QUE TU TEM QUE FAZER#
############################################################

M = magic_int
# F(x,y)
def F(x,y):
    f1 = (x**2)+(y**2)-1
    f2 = ((x**2)/M)+ 3*(y**2)-2
    vF = np.array([f1, f2])
    return vF

# F'(x,y)
def derivF(x,y):
    dxf1 = 2*x
    dyf1 = 2*y
    dxf2 = (2*x)/M
    dyf2 = 6*y
    vdF = np.array([[dxf1, dyf1], [dxf2, dyf2]])
    return vdF

# Sistema Linear
def sistlin(xk, yk):
    m1 = F(xk, yk) * -1
    m2 = derivF(xk, yk)
    sl = np.linalg.solve(m2, m1)
    return sl
    
# Método de Newton
def newton(x0, y0, lim=1e-6, inter=100):
    x = x0
    y = y0
    for i in range (inter):
        sl = sistlin(x, y)
        x = x + sl[0]
        y = y + sl[1]
        if abs(sl[0]) < lim and abs(sl[1]) < lim:
            break
    return x, y

xi = float(input("chute inicial do valor de x0: "))
yi = float(input("chute inicial do valor de y0: "))
x0 = xi
y0 = yi
rQ1 = newton(x0, y0)
print('Para o 1Q: ',rQ1)

x0 = -xi
y0 = yi
rQ2 = newton(x0, y0)
print('Para o 2Q: ',rQ2)

x0 = -xi
y0 = -yi
rQ3 = newton(x0, y0)
print('Para o 3Q: ',rQ3)

x0 = xi
y0 = -yi
rQ4 = newton(x0, y0)
print('Para o 4Q: ',rQ4)

#seguindo o exemplo acima: 
def f1(x,y):
  return x**2+y**2 -1

def f2(x,y):
  return x**2/2+3*y**2 - 2

# Amostra de pontos no plano
x = np.arange(-3.0, 3.0, 0.01)
y = np.arange(-3.0, 3.0, 0.01)
X, Y = np.meshgrid(x, y)

# Funcão
Zf1 = f1(X,Y)
Zf2 = f2(X,Y)

#Grafico
fig, ax = plt.subplots()
CSf = ax.contour(X, Y, Zf1, [0.0])
CSg = ax.contour(X, Y, Zf2, [0.0], colors=('r'))
ax.set_title('Curvas de $f_1(x,y)=0$ e $f_2(x,y)=0$')
ax.axis('equal')
plt.grid()
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.plot(rQ1[0], rQ1[1], 'ro',
         rQ2[0], rQ2[1], 'ro',
         rQ3[0], rQ3[1], 'ro',
         rQ4[0], rQ4[1], 'ro')

plt.show()