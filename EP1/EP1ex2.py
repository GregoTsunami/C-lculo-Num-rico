import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

import sys
import time

dados = {
    'aluno': ['João', 'José', 'Maria', 'Rosa', 'Pedro', 'Carlos', 'Daniel', 'Mario', 'Ronaldo', 'André', 'Gabriel', 'Nelson', 'Cláudia', 'Fábio', 'Cláudio', 'Armando'],
    'univ': ['USP', 'USP', 'UNICAMP', 'USP', 'UNICAMP', 'UNICAMP', 'USP', 'UNICAMP', 'USP', 'USP', 'UNICAMP', 'USP', 'USP', 'UNICAMP', 'USP', 'UNICAMP'],
    'nota_mat':  [9, 10, 7, 10, 3, 6, 9, 7, 7, 5, 3, 8, 9, 5, 6, 5],
    'nota_port': [7,  6, 8,  4, 7, 9, 5, 9, 5, 3, 9, 4, 3, 6, 5, 8]
}

df = pd.DataFrame(dados)

# Definir cores com base na variável 'local'
cores = {'USP': 'blue', 'UNICAMP': 'red'}

# Criar o gráfico de dispersão
plt.figure(figsize=(8, 6))
for local, cor in cores.items():
    plt.scatter(df[df['univ'] == local]['nota_mat'], df[df['univ'] == local]['nota_port'], label=local, color=cor, alpha=0.5)

# Adicionar rótulos e título
plt.xlabel('Nota Matemática')
plt.ylabel('Nota Português')
plt.legend(title='Universidade')
plt.axis([2, 10, 2, 10])

# Exibir o gráfico
#print(df)
#plt.show()

####################################################################################################################################################################################
plt.figure(figsize=(8, 6))

# Criar o gráfico de dispersão
for local, cor in cores.items():
    plt.scatter(df[df['univ'] == local]['nota_mat'], df[df['univ'] == local]['nota_port'], label=local, color=cor, alpha=0.5)


#Reta (chutamos esses valores!)
# ax+by-c = 0,  wx - c = 0,  w= (a, b)
b = 3.5
a = -2.8
c = 2.0

# Fazendo o gráfico da reta usando 2 pontos da reta
x0 = 0
y0 = (c - a*x0)/b
x1 = 10
y1 = (c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='green', linestyle='--', label='Reta de Separação')

# Adicionar rótulos e título
plt.xlabel('Nota Matemática')
plt.ylabel('Nota Português')
plt.legend(title='Universidade')
plt.axis([2, 10, 2, 10])
#plt.show()

####################################################################################################################################################################################
plt.figure(figsize=(8, 6))

# Criar o gráfico de dispersão
for local, cor in cores.items():
    plt.scatter(df[df['univ'] == local]['nota_mat'], df[df['univ'] == local]['nota_port'], label=local, color=cor, alpha=0.5)

#Reta (chutada)
# ax+by-c = 0 ou  wx - c = 0,  w= (a, b)
b = 3.5
a = -2.8
c = 2.0

# Gráficos das retas
#2 pontos da reta principal
x0 = 2
y0 = (c - a*x0)/b
x1 = 10
y1 = (c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='green', linestyle='--', label='Reta de Separação')

#2 pontos da reta de margem
# ax+by-c = 1
x0 = 2
y0 = (1+c - a*x0)/b
x1 = 10
y1 = (1+c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='red', linestyle=':', label='Reta de Margem +1')

#2 pontos da reta de margem
# ax+by-c = -1
x0 = 2
y0 = (-1+c - a*x0)/b
x1 = 10
y1 = (-1+c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='red', linestyle=':', label='Reta de Margem -1')


# Adicionar rótulos e título
plt.axis([2, 10, 2, 10])
plt.xlabel('Nota Matemática')
plt.ylabel('Nota Português')
plt.legend(title='Universidade')
#plt.show()

####################################################################################################################################################################################
############################################################
#AQUI COMEÇA O QUE TU TEM QUE FAZER#
############################################################
# Coloque seu código aqui
def dist(a, b):
    w = np.array([a, b])
    d = 2/np.linalg.norm(w)
    return d

#Teste da função:
# a = float(input("Valor de a: "))
# b = float(input("Valor de b: "))
# d = dist(a,b)
# print('Distância entre retas: ',d)

#Para o 1° exemplo:
t1 = dist(-2.8,3.5)
print('Distância do 1°: ', t1)

#Para o 2° exemplo:
e1 = dist(-1.6,2.0)
print('Distância do 2°: ', e1)

########################################################################################################################################################################################
# Criar o gráfico de dispersão
for local, cor in cores.items():
    plt.scatter(df[df['univ'] == local]['nota_mat'], df[df['univ'] == local]['nota_port'], label=local, color=cor, alpha=0.5)

#Reta
b = float(input("Insira o valor de b: ")) #b > 0 ou b < 0
a = float(input("Insira o valor de a: ")) #a < 0 ou a > 0
c = float(input("Insira o valor de c: "))
d = dist(a,b)
print("Distância = ", d)

#2 pontos da reta principal
x0 = 2
y0 = (c - a*x0)/b
x1 = 10
y1 = (c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='green', linestyle='--', label='Reta de Separação')

#2 pontos da reta de margem
# ax+by-c = 1
x0 = 2
y0 = (1+c - a*x0)/b
x1 = 10
y1 = (1+c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='red', linestyle=':', label='Reta de Margem +1')

#2 pontos da reta de margem
# ax+by-c = -1
x0 = 2
y0 = (-1+c - a*x0)/b
x1 = 10
y1 = (-1+c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='red', linestyle=':', label='Reta de Margem -1')


# Adicionar rótulos e título
plt.axis([2, 10, 2, 10])
plt.xlabel('Nota Matemática')
plt.ylabel('Nota Português')
plt.legend(title='Universidade')
plt.show()