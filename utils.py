import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def select_best(df:pd.DataFrame):
    best_fit = df.sort_values(by = "error").reset_index().iloc[0]

    a = best_fit["param"][0]
    b = best_fit["param"][1]
    best_fit_func = best_fit["function"]
    
    return best_fit["param"], best_fit_func


def find_zero(offSet:float, best_fit_func, params:np.ndarray, pressure:int, learning_rate=0.01):

    volume=0
    
    while(True):
        if offSet < volume:
            volume = best_fit_func(pressure, *params)
            pressure-=learning_rate
        else:
            break
            
    return np.array([pressure,volume])

def exponential_cicle(x, a, b, c):
    return -a*np.exp(-b*x)+c


def find_best_b(df:pd.DataFrame, raw_pressures:np.ndarray,b_percentage_range:tuple=(0.05,0.125), step:float=0.001, plot_figure:bool=True):

    params, best_fit_func = select_best(df)

    a = params[0]
    b = params[1]

    b_var_lst = []
    b_percentage_lst = []
    guess_zero_lst = []

    b_initial, b_end = b_percentage_range

    for b_percentage in np.arange(b_initial, b_end, 0.001):

        offSet = (b_percentage*b+a)
        initial_pressure = raw_pressures[0,0]

        guess_zero = find_zero(offSet, best_fit_func, params, initial_pressure, learning_rate = 0.01)
        guess_zero_lst.append(guess_zero)
        
        try:
            """
            ESTOU IGNORANDO O PRIMEIRO PAR DE PONTOS, POIS COMECA NO 4 ESSE ARRAY
            """
            b_exp_run_lst = []

            for i in range(4, len(raw_pressures)+1, 2):
                #Select first two points
                run = raw_pressures[i-2:i,:]
                #Add guess_zero
                run = np.vstack([guess_zero,run])
                #Fit curve with 3 points
                parameters, pcov = curve_fit(exponential_cicle, run[:,0], run[:,1], method="lm")
                b_exp_run_lst.append(parameters[1])

            b_var_lst.append(np.var(b_exp_run_lst))
            b_percentage_lst.append(b_percentage)
            b_exp_run_lst = []

        except Exception as e:
            print(e)
            
    if plot_figure:
        data=pd.DataFrame([b_percentage_lst,b_var_lst]).T.rename(columns = {0:"b_percentage", 1:"b_var"}).astype(float)
        
        fig = plt.figure(figsize = (8,6))
        ax = sns.scatterplot(x="b_percentage", y="b_var", data=data)
        plt.tight_layout()
        plt.show()
        
    index_best_b = np.argmin(b_var_lst)
    return b_percentage_lst[index_best_b], guess_zero_lst[index_best_b]


def plot_exponentials(best_b, guess_zero, df, raw_data, func):
    
    ######################################################################################################
    plt.figure(figsize = (14,8))
    plt.scatter(guess_zero[0],guess_zero[1], c = 'k', s = 100, label = "ZERO")

    new_pressures = range(int(guess_zero[0]), 100, 7)

    colors = ["b","g","r","m","y"]

    for i, cor in zip(range(2, raw_data.shape[0]+1, 2), colors):

        run = raw_data[i-2:i,:]
        run = np.vstack([guess_zero,run])

        parameters, pcov = curve_fit(func, run[:,0], run[:,1], method="lm")

        new_volumes = func(new_pressures, *parameters)
        plt.plot(new_pressures, new_volumes, c=cor, label = f"exponential {i}")
        plt.scatter(raw_data[i-2:i,0], raw_data[i-2:i,1], c=cor, label = f"Pair {i}")
        ##

    ##Plotando best fit
    best_fit = df.sort_values(by = "error").reset_index().iloc[0]
    new_volumes = best_fit["function"](new_pressures,*best_fit["param"])
    plt.scatter(new_pressures,new_volumes, c = 'black', marker = '+', label = "Fit")

    plt.xlabel("pressure")
    plt.ylabel("volume")
    print(f"Best b: {best_b}")
    print(f"Best zero: {guess_zero}")
    plt.legend()
    plt.show()
    