import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from collections import namedtuple

# funcao usada no find_best_b antigo do Rodrigo
def exponential_cicle(x, a, b, c):
    return -a*np.exp(-b*x)+c

def calc_fit_error(fit_func, params, xdata, ydata):
    error_lst = []
    for idx,x in enumerate(xdata):
        y = fit_func(x,*params)
        error_lst.append( np.sqrt((y-ydata[idx])**2) )
    erro = np.mean(error_lst)
    return erro

def find_best_b(ser, b_percentage_range:tuple=(0.95,1.05), step:float=0.0001):
   
    params = ser["param"]
    fit_func = ser["model"].function
    
    #Especifico para sigmoid
    # sigmoid(x, a, b, c, d): a + b/(1 + np.exp(-(x-c)/d))   
    a, b, c, d = params 
    
    b_esperado = np.exp(c/d)/(b)
    
    volume_ZEEP_esperado = fit_func(0, *params)
    
    #print( (b_esperado,[0,volume_ZEEP_esperado]) )
    
    return b_esperado, [0, volume_ZEEP_esperado]

def plot_exponentials(ser, aux_func):

    #print(ser)
    best_b = ser["best_b"]
    
    func = partial(aux_func, best_b)
    #func = aux_func
    
    guess_zero = ser["guess_zero"]
    raw_data   = ser["raw_data"]
    qtd_steps  = ser["qtd_steps"]
    
    v_zeep = guess_zero[1]
    
    new_pressures = np.arange(int(guess_zero[0]), 150, 1)
    colors = ["b","g","r","m","y"]
    
    exponential_lst = []
    exponential_tuple = namedtuple("exponential",['x', 'y'])
    
    ###################################################################################################### 
    ##Plotando best fit
    plt.figure(figsize = (14,8))
    plt.title(f'Caso {ser.name}: {ser.subject}: {ser.manobra}: {ser.qtd_steps} passos; Model error: {ser.error}')
    plt.xlabel("pressure")
    plt.ylabel("volume")
    
    model_func = ser["model"].function
    model_volumes = model_func(new_pressures,*ser["param"])
    plt.scatter(new_pressures,model_volumes-v_zeep, c = 'black', marker = '+', label = "Fit")


    print(model_volumes[-1])
    ######################################################################################################
    #Plot zero
    plt.scatter(guess_zero[0],guess_zero[1]-v_zeep, c = 'k', s = 100, label = "ZERO")

    for idx, (fst_point, snd_point, cor) in enumerate(zip(raw_data[0::2][:qtd_steps],raw_data[1::2][:qtd_steps], colors)):
        
        run = np.vstack([fst_point, snd_point])
        run = np.vstack([guess_zero,run])
        
        parameters, pcov = curve_fit(f      = func, 
                                     xdata  = run[:2,0],  
                                     ydata  = run[:2,1], 
                                     method = "trf",
                                     #p0     = model.inits,
                                     bounds = ([-np.inf,-np.inf],[np.inf,model_volumes[-1]]),
                                    )
        
        new_volumes = func(new_pressures, *parameters)
        exponential_lst.append(exponential_tuple(new_pressures, new_volumes))
        
        plt.plot(new_pressures, new_volumes-v_zeep, c=cor, label = f"exponential {idx}")
        plt.scatter(fst_point[0], fst_point[1]-v_zeep, c=cor, label = f"Pair {idx}")
        plt.scatter(snd_point[0], snd_point[1]-v_zeep, c=cor)
        
    ######################################################################################################

    plt.legend()
    plt.show()
    return exponential_lst
    