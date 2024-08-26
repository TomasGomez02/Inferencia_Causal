import numpy as np

def pr(r):
    return 1/3

def pc(c):
    return 1/3

def ps_rM0(s, r):
    return (s != r) * 1/2

# Prob de la pista modelo Monty Hall
def ps_rcM1(s,r,c):
    if r != c:
        return (s != r) * (s != c) * 1
    else:
        return (s != r) * 1/2

prcs_M = np.array([
    np.zeros((3,3,3)),
    np.zeros((3,3,3))
])
h = np.arange(3)
for r in h:
    for c in h:
        for s in h:
            prcs_M[0,r,c,s] = pr(r) * pc(c) * ps_rM0(s, 1)

def pc_M(m, c):
    return np.sum(prcs_M[m,:,c,:])

def ps_cM(s,c,m):
    return np.sum(prcs_M[m,:,c,s]) / np.sum(pc_M(m, c))

def pr_scM(r, s, c ,m):
    return prcs_M[m,r,c,s] / np.sum(prcs_M[m,:,c,s])

def pEpisodio_M(c,s,r,m):
    if m == 0:
        return pr(r) * pc(c) * ps_rM0(s, r)
    elif m == 1:
        return pr(r) * pc(c) * ps_rcM1(s, r, c)
    
def simular(episodios = 16, seed = 0):
    np.random.seed(seed)
    datos = []
    for _ in range(episodios):
        r = np.random.choice(3, p=[pr(hr) for hr in h])
        c = np.random.choice(3, p=[pc(hc) for hc in h])
        s = np.random.choice(3, p=[ps_rcM1(hs,r,c) for hs in h])
        datos.append((c, s, r))
    return datos

datos = simular()

def secuencia_de_prediciones(datos, m):
    p_datos_M = [1]
    for t in range(len(datos)):
        c, s, r = datos[t]
        p_datos_M.append(pEpisodio_M(c, s, r, m))
    return p_datos_M

def p_datos_M(datos, m):
    return np.prod(secuencia_de_prediciones(datos, m))

print(p_datos_M(datos, 0))
print(p_datos_M(datos, 1))

log_evicencia_m0 = np.log10(p_datos_M(datos, 0))
log_evicencia_m1 = np.log10(p_datos_M(datos, 1))

print(log_evicencia_m1 - log_evicencia_m0)

print(10**(log_evicencia_m1 - log_evicencia_m0))

print(10**(log_evicencia_m1 / (len(datos) * 3)))
print(10**(log_evicencia_m0 / (len(datos) * 3)))

def pM(m):
    return 1/2

pDatos_M0 = secuencia_de_prediciones(datos, 0)
pDatos_M1 = secuencia_de_prediciones(datos, 1)

p_datos_m = [np.cumprod(pDatos_M0) * pM(0),
             np.cumprod(pDatos_M1) * pM(1)]

p_datos = p_datos_m[0] + p_datos_m[1]

pM_datos = [p_datos_m[0] / p_datos,
            p_datos_m[1] / p_datos]

from matplotlib import pyplot as plt

plt.plot(pM_datos[0], label='M0: Base')
plt.plot(pM_datos[1], label='M1: Monty Hall')
plt.legend()
plt.show()