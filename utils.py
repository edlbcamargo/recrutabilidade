import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from collections import namedtuple

def select_best(df:pd.DataFrame):
    best_fit = df.sort_values(by = "error").reset_index().iloc[-1]

    a = best_fit["param"][0]
    b = best_fit["param"][1]
    best_fit_func = best_fit["function"]
    
    return best_fit["param"], best_fit_func


def find_zero(offSet:float, best_fit_func, params:np.ndarray, pressure, learning_rate=0.01):
    
    volume=1000
    
    while(True):
        if offSet < volume:
            volume = best_fit_func(pressure, *params)
            pressure-=learning_rate
        else:
            break
            
    return np.array([pressure,volume])

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
    fit_func = ser["function"]
    
    #Especifico para sigmoid
    # sigmoid(x, a, b, c, d): a + b/(1 + np.exp(-(x-c)/d))   
    a, b, c, d = params 
    
    b_esperado = np.exp(c/d)/(b)
    
    volume_ZEEP_esperado = fit_func(0, *params)
    
    #print( (b_esperado,[0,volume_ZEEP_esperado]) )
    
    return b_esperado, [0, volume_ZEEP_esperado]
    

    

    
#def find_best_b(ser, b_percentage_range:tuple=(0.05,0.125), step:float=0.001):
def find_best_b2(ser, b_percentage_range:tuple=(0.001,0.125), step:float=0.0001):
    
    params = ser["param"]
    fit_func = ser["function"] # sigmoid(x, a, b, c, d): a + b/(1 + np.exp(-(x-c)/d))
    raw_data = ser["raw_data"]
   
    a = params[0]
    b = params[1]

    b_var_lst = []
    b_percentage_lst = []
    guess_zero_lst = []
    error_lst = []

    b_initial, b_end = b_percentage_range

    for b_percentage in np.arange(b_initial, b_end, 0.001):

        offSet = (b_percentage*b+a)
        initial_pressure = raw_data[0,0]

        guess_zero = find_zero(offSet, fit_func, params, initial_pressure, learning_rate = 0.01)
        guess_zero_lst.append(guess_zero)
        
        try:
            """
            ESTOU IGNORANDO O PRIMEIRO PAR DE PONTOS, POIS COMECA NO 4 ESSE ARRAY
            Erick: Talvez primeiro par seja muito sujeito a tidal recruitment, e o segundo ponto P,V desse par
                   fica acima do esperado, por isso acaba não fitando uma exponencial com esse ponto.
            """
            b_exp_run_lst = []
            error_run_lst = []

            for i in range(4, len(raw_data)+1, 2):
                #Select first two points
                run = raw_data[i-2:i,:]
                #Add guess_zero
                run = np.vstack([guess_zero,run])
                #Fit curve with 3 points
                parameters, pcov = curve_fit(exponential_cicle, run[:,0], run[:,1], method="dogbox", p0=[1500, 0.04, 1500-guess_zero[1]], bounds=([100,0.01,-np.inf], [10000, 0.1, np.inf]))
                b_exp_run_lst.append(parameters[1])
                perr = calc_fit_error(exponential_cicle, parameters, run[:,0], run[:,1])
                error_run_lst.append(perr)
                
            error_lst.append(np.mean(error_run_lst))

            b_var_lst.append(np.var(b_exp_run_lst)) # o melhor b não é o com menor variância, e sim com menor erro!?!
            b_percentage_lst.append(b_percentage)
            b_exp_run_lst = []
            error_run_lst = []

        except Exception as e:
            print(e)

    #index_best_b = np.argmin(b_var_lst)
    index_best_b = np.argmin(error_lst)
    
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    axs[0].plot(b_percentage_lst,b_var_lst,c='r')
    axs[0].set_title('Variance')
    axs[1].plot(b_percentage_lst,error_lst,c='b')
    axs[1].set_title('Error')
    axs[2].plot(b_percentage_lst,error_lst,c='b')
    axs[2].set_ylim( (np.min(error_lst),np.min(error_lst)*20) )
    axs[2].set_title('Error (zoom)')
    print((index_best_b, b_percentage_lst[index_best_b], guess_zero_lst[index_best_b]))
                    
    return b_percentage_lst[index_best_b], guess_zero_lst[index_best_b]


def plot_exponentials(ser, aux_func):

    #print(ser)
    best_b = ser["best_b"]
    
    func = partial(aux_func, best_b)
    #func = aux_func
    
    guess_zero = ser["guess_zero"]
    raw_data = ser["raw_data"]
    
    v_zeep = guess_zero[1]
    
    new_pressures = np.arange(int(guess_zero[0]), 150, 1)
    colors = ["b","g","r","m","y"]
    
    exponential_lst = []
    exponential_tuple = namedtuple("exponential",['x', 'y'])
    
    ######################################################################################################
    plt.figure(figsize = (14,8))
    plt.scatter(guess_zero[0],guess_zero[1]-v_zeep, c = 'k', s = 100, label = "ZERO")

    for i, cor in zip(range(2, raw_data.shape[0]+1, 2), colors):

        run = raw_data[i-2:i,:]
        run = np.vstack([guess_zero,run])
        
        
        #print(f'run: {run}')

        parameters, pcov = curve_fit(func, run[:2,0], run[:2,1], method="dogbox")
        #parameters, pcov = curve_fit(func, run[:2,0], run[:2,1], method="dogbox",bounds=([-np.inf, -np.inf],[np.inf, np.inf]))
        new_volumes = func(new_pressures, *parameters)
        exponential_lst.append(exponential_tuple(new_pressures, new_volumes))
        
        plt.plot(new_pressures, new_volumes-v_zeep, c=cor, label = f"exponential {i}")
        plt.scatter(raw_data[i-2:i,0], raw_data[i-2:i,1]-v_zeep, c=cor, label = f"Pair {i}")
        ##

    ##Plotando best fit
    new_volumes = ser["function"](new_pressures,*ser["param"])
    plt.scatter(new_pressures,new_volumes-v_zeep, c = 'black', marker = '+', label = "Fit")

    plt.xlabel("pressure")
    plt.ylabel("volume")
    print(f"Best b: {best_b}")
    print(f"Best zero: {guess_zero}")
    plt.title(f'Caso {ser.name}: {ser.subject}: {ser.manobra}: {ser.n_point} passos; erro {ser.error}')
    plt.legend()
    plt.show()
    return exponential_lst
    