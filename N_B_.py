import numpy as np
from numpy import exp
from scipy.stats import gamma, norm, poisson, nbinom
from scipy.special import gamma as gm
from scipy.special import loggamma
import matplotlib.pyplot as plt
from pytwalk import pytwalk
import pandas as pd
#from numba import jit

df = pd.read_csv("Datos/Libro1_Covid19Updated.csv") 
count_t =  np.array(df["Counts"])#[:-33]
count_p =  np.array(df["Counts"])
data_p = np.diff(count_p)
obs_p = np.arange(0, len(count_p))

date_generated = pd.date_range(start="2020-03-01",end="2020-07-25")
dates = []
for date in date_generated:
    dates.append(date.strftime("%d-%m"))
    #print(date.strftime("%d-%m-%Y"))

mm = 0
#dates = dates#[mm:]
count_t = count_t[mm:]
data = np.diff(count_t)


#n_d = 5
#ff = len(count_t)%n_d
#dat_w = count_t[:-ff]
#week = dat_w.reshape((len(dat_w)//n_d,n_d))
#count_t = np.sum(week, axis=1)
#data = np.diff(count_t)
#plt.plot(obs_p[:-ff:5][:-1], data)

m = len(data) # Tamaño de muestra
mt = 1
obs_time = np.arange(0, m+1) / mt  # Observation time

# Fixed parameters
K = 500000 # 73980
N = 300000
Model = 3
t0 = 0
X0 = datadat_w = count_t[:-4]
#week = dat_w.reshape((len(dat_w)//5,5))
#count_t = np.sum(week, axis=1)

#data = np.diff(count_t)[0]

#@jit(nopython=True)    
def Mod3(theta, grid):
    """La función sigmoide sirve como la base de la función de Gompertz, en la
    que el crecimiento inicial es rápido seguido de una nivelación."""
    t0 = np.repeat(grid[0],len(grid))
    a = theta[0]
    b = theta[1]
    c = theta[2]   
    return  a * exp(-b * exp(-c * (grid-t0)))
#@jit(nopython=True)  
def Mod4(theta, grid):
    """Modelo Bertalanffy—M4."""
    t0 = np.repeat(grid[0],len(grid))
    a = theta[0]
    b = theta[1]
    c = theta[2]

    res = a*(1 - exp(-b * (grid-t0)))**c
    return res

if Model == 3:
    Mod = Mod3
    a_a = 2; b_a = a_a/K
    a_b = 2; b_b = a_b/11.66
    a_c = .5; b_c = a_c/0.02

if  Model == 4:
    Mod = Mod4
    a_a = 7; b_a = a_a/K
    a_b = 1; b_b = a_b/0.1
    a_c = 2; b_c = a_c/7

a_alph = 0.5
b_alph = a_alph / 0.04
# Hiperparameters
alpha = np.array([a_a, a_b, a_c, a_alph])
beta = np.array([b_a, b_b, b_c, b_alph])

def log_gam(data, alp):
    vec = [0.]*m
    for i, dat in enumerate(data):
        vec[i] = np.sum(np.log(np.arange(dat)+alp))
    return np.sum(np.array(vec))

#@jit(nopython=True)  
def loglikelihood(data, x):
    theta = x[:-1]
    alph = x[-1]
    alp_1 = 1 / alph
    dif = Mod(theta, obs_time)
    mu = np.diff(dif)
    a_mu = alph*mu
    #v1 = np.sum(loggamma(data + alp_1)  -  loggamma(alp_1))
    #v1 = log_gam(data, alp_1)    
    #v2 = np.sum((alp_1 + data)*np.log(1 + a_mu))
    #v3 = np.sum(np.log(a_mu)*data)
    v1 = loggamma(data + alp_1)  -  loggamma(alp_1)  
    v2 = (alp_1 + data)*np.log(1 + a_mu)
    v3 = np.log(a_mu)*data
    
    
    return  np.sum(v1 - v2 + v3)

#@jit(nopython=True)
def logprior(x):  
    log_p = (alpha - 1)*np.log(x) - beta*x
    return np.sum(log_p)
#@jit(nopython=True)
def Energy( x, data=data):
    return -1*(loglikelihood(data,x) + logprior(x))

def Supp(x):
    return all(x > 0.0)

def LG_Init(): ###Simulate from the prior
    sim = gamma.rvs(alpha, scale=1/beta)
    return sim.ravel()
#"""
d = len(alpha) # number of parameters
LG_twalk = pytwalk(n=d, U=Energy, Supp=Supp)

LG_twalk.Run( T=500000, x0=LG_Init(), xp0=LG_Init())

# Posterior Analisys
burnin = 50000
thini = 300
Output = LG_twalk.Output[burnin::thini, :]
Output_theta = Output[:,:d]

#plt.figure()
#plt.plot(LG_twalk.Output[burnin:,-1])


# Estadísticas
i = np.argmax(-LG_twalk.Output[:,-1])
th_map_ = LG_twalk.Output[ i, :]; th_map = th_map_[:d]
th_mean_ = np.median(Output[:, :],axis=0); th_mean = th_mean_ [:d]
Q025 = np.quantile(Output_theta,0.025,axis=0)
Q50 = np.quantile(Output_theta,0.5,axis=0)
Q975 = np.quantile(Output_theta,0.975,axis=0)

sumar = np.array([Q025,th_mean,Q975]).T
sumary = pd.DataFrame(data=np.round(sumar,2), columns=["Q025","Mean","Q975"], index=["a", "b", "c", "alpha"])
print(sumary)
#sumary.to_csv(r'Figures/Sumary.csv', index = False)


# Graficamos los histogramas de las simulaciones de cada parametro
plt.figure()
n_plot = 221
#plt.subplot(224); plt.plot(LG_twalk.Output[burnin:,-1])
for i in range(d):
    xpri = np.linspace(min(Output[:,i]),max(Output[:,i]),400)
    plt.subplot(n_plot)
    LG_twalk.Hist(par=i, start=burnin, density=True)
    plt.plot(xpri, gamma.pdf(xpri,alpha[i], scale=1/beta[i]))
    n_plot += 1 

time_f = np.arange(len(dates))#mt
n_pre = len(time_f)
def Mod_theta(theta): return np.diff(Mod(theta,time_f))

# Graficamos la incertidumbre de las predicciones diarias
def Mod_NB(x):
    theta = x[:-1]
    r = 1/x[-1]
    FM = Mod(theta,time_f)
    mu = np.diff(FM)
    p = r / (mu+r) #mu/(mu+r)
    FM_error = nbinom.rvs(r, p)
    return FM_error

# Prediccion esperada de casos diarios y su varianza
Eval_theta = np.apply_along_axis(Mod_theta,axis=1, arr = Output_theta)
cases_mean2 = np.mean(Eval_theta,axis=0) # X(Y_fut) = E_theta( E_Y (Y_fut|theta) ) = E_theta( F(theta) )
# cases_mean = np.mean(simu_pred,axis=0) # Is the same approximaty
#sd_sim = np.sqrt(cases_mean)
#VarF = np.var(Eval_theta, axis=0)
#Var_pred = cases_mean + VarF
#Sd_Pred = np.sqrt(Var_pred)

# Simulaciones de prediccion diaria
simu_pred = np.apply_along_axis(Mod_NB, axis=1, arr = Output_theta)
cases_mean = np.mean(simu_pred,axis=0) # Is the same approximaty
cases      = Mod_theta(th_map) # MAP
cases_med  = np.median(simu_pred,axis=0)
cases_Q025 =  np.quantile(simu_pred,0.05,axis=0)
cases_Q975 = np.quantile(simu_pred,0.95,axis=0)

# Prediccion de los incrementos diarios
def plot_Incre():
    m_pr = 0
    time_pred = time_f[m_pr+1:]
    #maxi = np.argmax(cases)

    plt.figure()
    #plt.plot(time_pred, (simu_pred.T)[m_pr:,:], '-', color="gray", alpha=0.05)
    plt.plot(time_f[1:], cases, 'black', label="Predicción media")
    plt.bar(time_f[1:m+1], data, label = "Valores reales")
    #plt.plot(time_f[1:], cases_med, 'black', label="Prediccion media")
    plt.plot(time_pred, cases_Q025[m_pr:], "-.", color='darkmagenta', label="Intervalo de  predicción")
    plt.plot(time_pred, cases_Q975[m_pr:], "-.", color='darkmagenta')
    plt.fill_between(time_f[1:], cases_Q025, cases_Q975, color = "darkmagenta")

    #plt.vlines(time_f[maxi+1],ymin=0,ymax=cases[maxi], color="blue")#, label="Fecha probable del pico máximo")
    plt.xticks(time_f[1::3],dates[1::3], rotation=90)
    plt.ylabel("Casos confirmados diarios")
    plt.legend()
    plt.title("Casos diarios de COVID") 
    #plt.title("Predicción de casos confirmados diarios Modelo M%i: BN" %Model)   
    
    return  np.append(1,cases_Q025), np.append(1,cases_mean), np.append(1,cases_Q975)

diff_Q025, diff_mean, diff_Q975 = plot_Incre()
#plt.plot(np.diff(count_p), color="red")

# Tabla de predicciones hasta el 30 de mayo del 2020
def plot_uq():
    m1 = 0
    m2 = 30
    time_pred = time_f[m1+1:-m2]
    Nt = np.cumsum(simu_pred,axis=1)
    Nt_mean = np.mean(Nt,axis=0) #np.append(1,np.mean(Nt,axis=0)) #np.cumsum(diff_mean)
    Nt_MAP  = np.cumsum(cases)
    Nt_Q500 = np.quantile(Nt,0.5,axis=0)
    Nt_Q025 = np.quantile(Nt,0.05,axis=0) # np.append(1,np.quantile(Nt,0.05,axis=0))
    Nt_Q975 = np.quantile(Nt,0.95,axis=0)
    
    plt.figure()
    #plt.plot(time_pred, (Nt.T)[m1:,:], '-', color="gray", alpha=0.05)
    #plt.plot(time_pred, Nt_mean[m1:-m2], 'green', label="Prediccion media")
    plt.plot(time_pred, Nt_Q500[m1:-m2],'black', label="Mediana")
    plt.plot(time_pred, Nt_Q025[m1:-m2], "-.", color='darkmagenta', label="Intervalos de  predicción")
    plt.plot(time_pred, Nt_Q975[m1:-m2], "-.", color='darkmagenta')
    plt.step(obs_time[m1+1:], count_t[m1+1:],marker=".",label = "Valores reales", where = "pre")
    plt.fill_between(time_pred, Nt_Q025[m1:-m2], Nt_Q975[m1:-m2], color = "darkmagenta")

    plt.xticks(time_f[::3],dates[::3], rotation=90)
    plt.ylabel("Casos confirmados acumulados")
    plt.title("No. de casos de COVID acumulados en México")
    #plt.title("Modelo M%i" %Model)
    plt.legend()
    #plt.savefig("Predic_Incertidumbre.png")
    return np.append(1,Nt_mean), np.append(1,Nt_MAP), np.append(1,Nt_Q025), np.append(1,Nt_Q975) 
Nt_mean, Nt_MAP, Nt_Q025, Nt_Q975 = plot_uq()
#plt.plot(count_p, color="red")
    
print_cases = Nt_mean #cases # cases_mean

def print_table(fromi=45, toi=50):
    Obs_com = np.repeat(0, len(print_cases))
    Obs_com[:m+1] = count_t
    dif_obs = np.append(1,Obs_com[1:]-Obs_com[:-1]); dif_obs[m+1:] = 0
    dat = np.round([Obs_com, Nt_Q025, print_cases, Nt_Q975,dif_obs,diff_Q025, diff_mean,diff_Q975],0).T  
    covid = pd.DataFrame(data=dat[fromi:toi], columns=["Obs","Q025","Mean","Q975","Day_obs","Day_Q025","Day_mean","Day_QQ975"], index=dates[fromi:toi])
    print(covid.astype(int))
    #covid_2 = pd.DataFrame(data=dat, columns=["Obs","Q025","Mean","Q975","Daily_obs","Daily_Q025","Daily_mean","Daily_QQ975"], index=dates)
    #covid_2.to_csv('Figures/covid_2.csv', index = False)
#print_table(fromi=45, toi=85)
#print_table(fromi=0, toi=92)
print_table(fromi=70, toi=100)
#"""


# np.array([5.735868e+04, 1.108075e+01, 3.513101e-02])