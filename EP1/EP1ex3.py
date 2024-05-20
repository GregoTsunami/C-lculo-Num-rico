import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from sklearn.svm import LinearSVC


import sys
import time

t = np.arange(0.0, 5.0, 0.01)

plt.figure()

plt.subplot(221)
s = (t-1)*(t-1)

plt.plot(t, s)
plt.title('mínimo em $z>0$ (derivada nula)   ')

plt.subplot(222)
s = (t+1)*(t+1)

plt.plot(t, s)
plt.title('   mínimo em $z=0$ (derivada não-negativa)')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

#plt.show()


##################################################################################################################################################

def F(X, y, w, c, lam, mu, s, r):
  # Recebe:
  # X - matrix (n x p) com cada linha uma amostra/ponto diferente (x_i), e cada ponto tendo dimensão "p".
  # y - vetor (n x 1) com classificação dos pontos +-1 para cada ponto
  # w - vetor (p x 1) com os pesos para a equação do hiperplano
  # c - constante do hiperplano
  # lam - vetor (n x 1) dos multiplicadores de Lagrange (variável auxiliar)
  # mu  - vetor (n x 1) multiplicadores (variável auxiliar)
  # s - vetor (n x 1) vetor de variáveis de folga (variável auxiliar)
  # r - escalar auxiliar
  #
  # Devolve o vetor da função F(z) que queremos achar o zero.
  # Observação z=[w,c,s,lam,mu]
  #  Vetor  : F(z)

  #Matriz de dados
  n, p = X.shape

  y = y.reshape(n,1)

  yr = np.repeat( y, p, axis=1 )
  YX = np.multiply(yr, X)

  lamYXt = np.dot(lam.T, YX).T
  wtX = np.dot(w.T, X.T).reshape(n,1)
  lamy = np.dot(lam.T, y).reshape(1, 1)

  # F(z)
  Fz = np.block([
    [y*(wtX-c)-s-1],
    [w+lamYXt],
    [-lamy],
    [-lam-mu],
    [mu*s-r]
  ])

  # Retorna o valor da função que queremos achar o zero.
  return Fz


##################################################################################################################################################

def sistema_newton(X, y, w, c, lam, mu, s, r):
  # Recebe:
  # X - matrix (n x p) com cada linha uma amostra/ponto diferente (x_i), e cada ponto tendo dimensão "p".
  # y - vetor (n x 1) com classificação dos pontos +-1 para cada ponto
  # w - vetor (p x 1) com os pesos para a equação do hiperplano
  # c - constante do hiperplano
  # lam - vetor (n x 1) dos multiplicadores de Lagrange (variável auxiliar)
  # mu  - vetor (n x 1) multiplicadores (variável auxiliar)
  # s - vetor (n x 1) vetor de variáveis de folga (variável auxiliar)
  # r - escalar auxiliar
  #
  # Devolve Solução do sistema do método de Newton com direções na ordem
  # [dw,
  #  dc,
  #  ds,
  #  dlam,
  #  dmu]

  # Observação z=[w,c,s,lam,mu]
  #  Matriz : F'(z)
  #  Vetor  : F(z)
  # onde F' é a Jacobiana do problema de Newton e F é a função que queremos encontrar o zero.
  # O sistema linear resolvido é F'(z) d = - F(z)

  #Montar a matrix para o método de Newton
  n, p = X.shape

  y = y.reshape(n,1)

  yr = np.repeat( y, p, axis=1 )
  YX = np.multiply(yr, X)

  Onxn = np.zeros((n,n))
  Opxn = np.zeros((p,n))
  Onxp = np.zeros((n,p))
  Opx1 = np.zeros((p,1))
  O1xp = np.zeros((1,p))
  O1xn = np.zeros((1,n))
  Onx1 = np.zeros((n,1))
  Inxn = np.eye(n)
  Ipxp = np.eye(p)
  M = np.diag(mu.reshape(n))
  S = np.diag(s.reshape(n))

  #Matrix
  A=np.block([
      [YX,  -y,   -Inxn,   Onxn,   Onxn],
      [Ipxp, Opx1, Opxn,   YX.T,   Opxn],
      [O1xp, 0,    O1xn,   -y.T,   O1xn],
      [Onxp, Onx1, Onxn,  -Inxn,  -Inxn],
      [Onxp, Onx1,    M,   Onxn,   S   ]
      ])


  #Resolve sistema linear usando Numpy
  z = np.linalg.solve(A, -F(X, y, w, c, lam, mu, s, r))

  dw = z[:p]
  dc = z[p]
  ds = z[p+1:p+1+n]
  dlam = z[p+1+n:p+1+2*n]
  dmu = z[p+1+2*n:p+1+3*n]

  # Retorna solução do sistema linear (incremento do passo de Newton)
  return dw, dc, ds, dlam, dmu


##################################################################################################################################################

# Método de pontos interiores

def pontos_interiores_SVM(X, y, tolresid = 1e-12, itmax = 1000):
  #
  # Recebe:
  #  X : matriz nxp de dados
  #  y : vetor nx1 de classificações +-1
  #
  #  Opcionais:
  #  tolresid : critério de parada (tolerância do resíduo - valor de ||F(z)||)
  #  itmax    : máximo de iterações
  #
  # Retorna:
  #  w  : vetor de definição do hiperplano
  #  c  : constante associada ao hiperplano de separação
  #

  #Inicialização - dimensões
  n, p = X.shape

  #Chute inicial para hiperplano (zero)
  w = np.zeros(p).reshape(p,1)
  c = 0

  # Chute inicial das variáveis auxiliares

  # Vetor de folga
  s = np.ones(n).reshape(n,1)

  #Multiplicadores
  mu = np.ones(n).reshape(n,1)
  lam = np.zeros(n).reshape(n,1)

  # Parâmetro inicial de controle da complementaridade
  r = np.matmul(s.T,mu)/n


  # Loop principal do Método

  #Iteração
  resid = 1
  it = 0
  while (resid>tolresid) and (it<itmax):
    it=it+1

    # Resolver o sistema linear de Newton
    dw, dc, ds, dlam, dmu = sistema_newton(X, y, w, c, lam, mu, s, r)

    # Imposição das restrições de desigualdades em s e mu
    t_s = 1.0
    if np.any(ds<0):
      t_s = 0.99*np.min(-s[ds<0]/ds[ds<0])
    t_mu = 1.0
    if np.any(dmu<0):
      t_mu = 0.99*np.min(-mu[dmu<0]/dmu[dmu<0])
    t = np.min([t_s, t_mu])

    # Atualizar as variáveis
    w=w+t*dw
    c=c+t*dc
    s=s+t*ds
    lam=lam+t*dlam
    mu=mu+t*dmu

    # Atualizar o parâmetro r
    r=0.2*r

    #Cálculo do resíduo no sistema original (com r=0)
    resid = np.linalg.norm(F(X, y, w, c, lam, mu, s, 0))

    # print()
    # print("Iteração: ",  it, " Resíduo:", resid)
    # print("  Hiperplano w:", w.T," c:", c)

#   print()
#   print("Solução final. Hiperplano w:", w.T," c:", c)
  return w, c


##################################################################################################################################################

dados = {
    'aluno': ['João', 'José', 'Maria', 'Rosa', 'Pedro', 'Carlos', 'Daniel', 'Mario', 'Ronaldo', 'André', 'Gabriel', 'Nelson', 'Cláudia', 'Fábio', 'Cláudio', 'Armando'],
    'univ': ['USP', 'USP', 'UNICAMP', 'USP', 'UNICAMP', 'UNICAMP', 'USP', 'UNICAMP', 'USP', 'USP', 'UNICAMP', 'USP', 'USP', 'UNICAMP', 'USP', 'UNICAMP'],
    'nota_mat': [9, 10, 7, 10, 3, 6, 9, 7, 7, 5, 3, 8, 9, 5, 6, 5],
    'nota_port': [7, 6, 8, 4, 7, 9, 5, 9, 5, 3, 9, 4, 3, 6, 5, 8]
}


df = pd.DataFrame(dados)
x1 = np.array(df['nota_mat'].values)
x2 = np.array(df['nota_port'].values)

# Transformar nomes das universidade em números +-1
df.loc[df["univ"] == "USP", "univ_num"] = -1
df.loc[df["univ"] == "UNICAMP", "univ_num"] = +1
y = np.array(df['univ_num'].values)

# Matriz X
X = np.array((x1, x2)).transpose()
n, p = X.shape

# Vetor de classificação y
y = y.reshape(n,1)


##################################################################################################################################################

# Teste do método de pontos interiores
w, c = pontos_interiores_SVM(X, y)


##################################################################################################################################################

plt.figure(figsize=(8, 6))

# Definir cores com base na variável 'local'
cores = {'USP': 'blue', 'UNICAMP': 'red'}

# Criar o gráfico de dispersão
for local, cor in cores.items():
    plt.scatter(df[df['univ'] == local]['nota_mat'], df[df['univ'] == local]['nota_port'], label=local, color=cor, alpha=0.5)

#Reta
# wx - c = 0, com w= (a, b)
a = w[0]
b = w[1]
#2 pontos da reta principal
x0 = 2
y0 = (c - a*x0)/b
x1 = 10
y1 = (c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='green', linestyle='--', label='Reta de Separação')

#2 pontos da reta de margem
# wx - c = 1,  com w= (a, b)
x0 = 2
y0 = (1+c - a*x0)/b
x1 = 10
y1 = (1+c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='red', linestyle=':', label='Reta de Margem +1')

#2 pontos da reta de margem
# wx - c = -1, com w= (a, b)
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


##################################################################################################################################################

#Dados (os mesmos de antes)
n, p = X.shape
x_train = X
y_train = y.reshape(n)

# Primeiro dizemos que tipo de modelo que queremos (SVM linear)
# Colocamos C bem grande, para que ele não permita erros de classificação
# a função loss é escolhida como 'hinge' para coincidir com o modelo apresentado
model = LinearSVC(loss='hinge',C=1e12, max_iter=10000)

# Agora pedimos para ele ajustar o modelo, ou seja,
#  resolver o problema de otimização e achar o hiperplano
svmfit = model.fit(x_train,y_train)

# Esses são os parâmetros do hiperplano ajustado pelo sklearn
w_sklearn = svmfit.coef_[0]

# **Cuidado**, o nosso c é -c do sklearn!!!
c_sklearn = - svmfit.intercept_

# print(" w , c ")
# print(w_sklearn, c_sklearn)


##################################################################################################################################################

plt.figure(figsize=(8, 6))

# Definir cores com base na variável 'local'
cores = {'USP': 'blue', 'UNICAMP': 'red'}

# Criar o gráfico de dispersão
for local, cor in cores.items():
    plt.scatter(df[df['univ'] == local]['nota_mat'], df[df['univ'] == local]['nota_port'], label=local, color=cor, alpha=0.5)

#Reta
# wx - c = 0, com w= (a, b)
a = w_sklearn[0]
b = w_sklearn[1]

#2 pontos da reta principal
x0 = 2
y0 = (c - a*x0)/b
x1 = 10
y1 = (c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='green', linestyle='--', label='Reta de Separação')

#2 pontos da reta de margem
# wx - c = 1, com w= (a, b)
x0 = 2
y0 = (1+c - a*x0)/b
x1 = 10
y1 = (1+c - a*x1)/b
plt.plot([x0, x1], [y0, y1], color='red', linestyle=':', label='Reta de Margem +1')

#2 pontos da reta de margem
# wx - c = -1, com w= (a, b)
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


##################################################################################################################################################

############################################################
#AQUI COMEÇA O QUE TU TEM QUE FAZER#
############################################################

#Se r = 0:
def pontos_interiores_SVM_com_r0(X, y, tolresid = 1e-12, itmax = 1000):
  #
  # Recebe:
  #  X : matriz nxp de dados
  #  y : vetor nx1 de classificações +-1
  #
  #  Opcionais:
  #  tolresid : critério de parada (tolerância do resíduo - valor de ||F(z)||)
  #  itmax    : máximo de iterações
  #
  # Retorna:
  #  w  : vetor de definição do hiperplano
  #  c  : constante associada ao hiperplano de separação
  #

  #Inicialização - dimensões
  n, p = X.shape

  #Chute inicial para hiperplano (zero)
  w = np.zeros(p).reshape(p,1)
  c = 0

  # Chute inicial das variáveis auxiliares

  # Vetor de folga
  s = np.ones(n).reshape(n,1)

  #Multiplicadores
  mu = np.ones(n).reshape(n,1)
  lam = np.zeros(n).reshape(n,1)

  # Parâmetro inicial de controle da complementaridade
  r = 0


  # Loop principal do Método

  #Iteração
  resid = 1
  it = 0
  while (resid>tolresid) and (it<itmax):
    it=it+1

    # Resolver o sistema linear de Newton
    dw, dc, ds, dlam, dmu = sistema_newton(X, y, w, c, lam, mu, s, r)

    # Imposição das restrições de desigualdades em s e mu
    t_s = 1.0
    if np.any(ds<0):
      t_s = 0.99*np.min(-s[ds<0]/ds[ds<0])
    t_mu = 1.0
    if np.any(dmu<0):
      t_mu = 0.99*np.min(-mu[dmu<0]/dmu[dmu<0])
    t = np.min([t_s, t_mu])

    # Atualizar as variáveis
    w=w+t*dw
    c=c+t*dc
    s=s+t*ds
    lam=lam+t*dlam
    mu=mu+t*dmu

    # Atualizar o parâmetro r
    r=0.2*r

    #Cálculo do resíduo no sistema original (com r=0)
    resid = np.linalg.norm(F(X, y, w, c, lam, mu, s, 0))

    # print()
    # print("Iteração: ",  it, " Resíduo:", resid)
    # print("  Hiperplano w:", w.T," c:", c)

#   print()
#   print("Solução final. Hiperplano w:", w.T," c:", c)
#   print("Número de Iterações: ", it)
  return w, c

w, c = pontos_interiores_SVM_com_r0(X,y)

##############################################################

def pontos_interiores_SVM_com_gam(X, y, gam = 0.2, tolresid = 1e-12, itmax = 1000):
  #
  # Recebe:
  #  X : matriz nxp de dados
  #  y : vetor nx1 de classificações +-1
  #
  #  Opcionais:
  #  tolresid : critério de parada (tolerância do resíduo - valor de ||F(z)||)
  #  itmax    : máximo de iterações
  #
  # Retorna:
  #  w  : vetor de definição do hiperplano
  #  c  : constante associada ao hiperplano de separação
  #

  #Inicialização - dimensões
  n, p = X.shape

  #Chute inicial para hiperplano (zero)
  w = np.zeros(p).reshape(p,1)
  c = 0

  # Chute inicial das variáveis auxiliares

  # Vetor de folga
  s = np.ones(n).reshape(n,1)

  #Multiplicadores
  mu = np.ones(n).reshape(n,1)
  lam = np.zeros(n).reshape(n,1)
  
  # Parâmetro inicial de controle da complementaridade
  r = np.matmul(s.T,mu)/n


  # Loop principal do Método

  #Iteração
  resid = 1
  it = 0

  while (resid>tolresid) and (it<itmax):
    it=it+1

    # Resolver o sistema linear de Newton
    dw, dc, ds, dlam, dmu = sistema_newton(X, y, w, c, lam, mu, s, r)

    # Imposição das restrições de desigualdades em s e mu
    t_s = 1.0
    if np.any(ds<0):
      t_s = 0.99*np.min(-s[ds<0]/ds[ds<0])
    t_mu = 1.0
    if np.any(dmu<0):
      t_mu = 0.99*np.min(-mu[dmu<0]/dmu[dmu<0])
    t = np.min([t_s, t_mu])

    # Atualizar as variáveis
    w=w+t*dw
    c=c+t*dc
    s=s+t*ds
    lam=lam+t*dlam
    mu=mu+t*dmu

    # Atualizar o parâmetro r
    r=gam*r

    #Cálculo do resíduo no sistema original (com r=0)
    resid = np.linalg.norm(F(X, y, w, c, lam, mu, s, 0))

    # print()
    # print("Iteração: ",  it, " Resíduo:", resid)
    # print("  Hiperplano w:", w.T," c:", c)

    print()
#   print("Solução final. Hiperplano w:", w.T," c:", c)
    # print("Gama: ", gam, "; Iterações: ", it)
  return w, c

conj_gam = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# print("Gama/Iterações")
for gam in conj_gam:
  pontos_interiores_SVM_com_gam(X, y, gam)
#   print()
  
##############################################################
#3.
def pontos_interiores_SVM_com_criterio(X, y, tolresid=1e-12, tolnorm=1e-12, itmax=1000):
    n, p = X.shape
    w = np.zeros((p, 1))
    c = np.array([[0]])
    s = np.ones((n, 1))
    mu = np.ones((n, 1))
    lam = np.zeros((n, 1))
    r = np.dot(s.T, mu) / n
    resid = 1
    it = 0
    z_prev = np.vstack([w, c, s, lam, mu])

    while (resid > tolresid or np.linalg.norm(z_prev - np.vstack([w, c, s, lam, mu])) > tolnorm) and (it < itmax):
        it += 1
        z_prev = np.vstack([w, c, s, lam, mu])
        dw, dc, ds, dlam, dmu = sistema_newton(X, y, w, c, lam, mu, s, r)
        t_s = 1.0
        if np.any(ds < 0):
            t_s = 0.99 * np.min(-s[ds < 0] / ds[ds < 0])
        t_mu = 1.0
        if np.any(dmu < 0):
            t_mu = 0.99 * np.min(-mu[dmu < 0] / dmu[dmu < 0])
        t = np.min([t_s, t_mu])
        w = w + t * dw
        c = c + t * dc
        s = s + t * ds
        lam = lam + t * dlam
        mu = mu + t * dmu
        r = 0.2 * r
        resid = np.linalg.norm(F(X, y, w, c, lam, mu, s, 0))

    # print()
    # print("Solução final. Hiperplano w:", w.T," c:", c, "N° it:", it)
    return w, c

w, c = pontos_interiores_SVM_com_criterio(X,y)

##############################################################

#4. 
def F_semlam(X, y, w, c, mu, s, r):
    n, p = X.shape

    y = y.reshape(n, 1)

    yr = np.repeat(y, p, axis=1)
    YX = np.multiply(yr, X)

    muYXt = np.dot(mu.T, YX).T
    wtX = np.dot(w.T, X.T).reshape(n, 1)
    muy = np.dot(mu.T, y).reshape(1, 1)

    # F(z)
    Fz = np.block([
        [y * (wtX - c) - s - 1],
        [w - muYXt],
        [muy],
        [mu - mu],
        [mu * s - r]
    ])

    return Fz

def sistema_newton_semlam(X, y, w, c, mu, s, r):
    n, p = X.shape
    y = y.reshape(n, 1)
    yr = np.repeat(y, p, axis=1)
    YX = np.multiply(yr, X)
    Onxn = np.zeros((n, n))
    Opxn = np.zeros((p, n))
    Onxp = np.zeros((n, p))
    Opx1 = np.zeros((p, 1))
    O1xp = np.zeros((1, p))
    O1xn = np.zeros((1, n))
    Onx1 = np.zeros((n, 1))
    Inxn = np.eye(n)
    Ipxp = np.eye(p)
    M = np.diag(mu.reshape(n))
    S = np.diag(s.reshape(n))
    A = np.block([
        [YX, -y, -Inxn, Onxn, Onxn],
        [Ipxp, Opx1, Opxn, YX.T, Opxn],
        [O1xp, 0, O1xn, -y.T, O1xn],
        [Onxp, Onx1, Onxn, -Inxn, -Inxn],
        [Onxp, Onx1, M, Onxn, S]
    ])
    z = np.linalg.solve(A, -F_semlam(X, y, w, c, mu, s, r))
    dw = z[:p]
    dc = z[p]
    ds = z[p + 1:p + 1 + n]
    dmu = z[p + 1 + n:p + 1 + 2 * n]

    # Retorna a solução do sistema linear (incremento do passo de Newton)
    return dw, dc, ds, dmu

def pontos_interiores_SVM_semlam(X, y, tolresid=1e-12, itmax=1000):
    #
    # Recebe:
    #  X : matriz nxp de dados
    #  y : vetor nx1 de classificações +-1
    #
    #  Opcionais:
    #  tolresid : critério de parada (tolerância do resíduo - valor de ||F(z)||)
    #  itmax    : máximo de iterações
    #
    # Retorna:
    #  w  : vetor de definição do hiperplano
    #  c  : constante associada ao hiperplano de separação

    # Inicialização - dimensões
    n, p = X.shape

    # Chute inicial para hiperplano (zero)
    w = np.zeros(p).reshape(p, 1)
    c = 0

    # Chute inicial das variáveis auxiliares

    # Vetor de folga
    s = np.ones(n).reshape(n, 1)

    # Multiplicadores
    mu = np.ones(n).reshape(n, 1)
    lam = np.zeros(n).reshape(n, 1)

    # Parâmetro inicial de controle da complementaridade
    r = np.matmul(s.T, mu) / n

    # Loop principal do Método

    # Iteração
    resid = 1
    it = 0
    while (resid > tolresid) and (it < itmax):
        it = it + 1

        # Resolver o sistema linear de Newton
        dw, dc, ds, dmu = sistema_newton_semlam(X, y, w, c, mu, s, r)

        # Imposição das restrições de desigualdades em s e mu
        t_s = 1.0
        if np.any(ds < 0):
            t_s = 0.99 * np.min(-s[ds < 0] / ds[ds < 0])
        t_mu = 1.0
        if np.any(dmu < 0):
            t_mu = 0.99 * np.min(-mu[dmu < 0] / dmu[dmu < 0])
        t = np.min([t_s, t_mu])

        # Atualizar as variáveis
        w = w + t * dw
        c = c + t * dc
        s = s + t * ds
        mu = mu + t * dmu

        # Atualizar o parâmetro r
        r = 0.2 * r

        # Cálculo do resíduo no sistema original (com r=0)
        resid = np.linalg.norm(F_semlam(X, y, w, c, mu, s, 0))
        # print()
        print("Iteração: ",  it, " Resíduo:", resid)
        print("  Hiperplano w:", w.T," c:", c)

    print()
    print("Solução final. Hiperplano w:", w.T," c:", c)

    return w, c

w,c = pontos_interiores_SVM_semlam(X,y)