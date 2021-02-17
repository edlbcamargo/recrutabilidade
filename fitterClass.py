import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.legend import Legend

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from sklearn.metrics import mean_squared_error


"""
Função usada para fittar os dados.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
"""

"""
Atributos:

- As variáveis `pressure` e `volume` contém as pressões e volumes utilizados para fitar as funções sigmoid e exponencial. Espera-se que para um pulmão saudável o melhor fit seja exponencial, enquanto que para o doente espera-se um melhor fit da sigmoid. !!Entretanto vale notar que passamos somente os pontos de volume e pressão mínimo (PEEP) de cada passo!!.

- A variável `raw_data` contém uma cópia com todos os dados de pressão e volume.

- As váriaveis `subject`, `manobra`, e `n_point` servem para identificação do caso, respectivamente, nome do porco, manobra e passo. Exemplo, na manobra D existem 5 passos, na C existem 4 e na B existem 3.

- A variável `pmin` contém a PEEP mínima utilzida no fit das sigmoids e exponenciais

- A variável `estimators` é uma lista de diferentes estimadores para o fit das sigmoids e exponenciais. Exemplo: "lm" e "dogbox".

- A variável `interpolate` !!!não foi testada!!! e está setada como False, isto é, !!!No momento não estamos interpolando!!!. Ela é responsável pela interpolação dos dados visando melhorar o fit por meio da criação de mais pontos. 

Funções:

_interpolate: cria as versões interpoladas das variáveis `pressure` (`interp_pressures`) e `volumes` (`interp_volumes`).

_get_error: Calcula o "root mean square", volume fitado vs volume

_make_fit_report: Constrói o dataframe que contém a análise do caso (codificado por subject, manobra, n_point). Cada linha do dataframe contém todas informações necessárias daquele caso em específico.

fit: Essa função funciona como uma interface para o usuário, chamando a função `_make_fit_report`, setando os parâmetros necessarios e retornando um DataFrame.

make_plot: Plota os "n" melhores casos do dataframe retornado pela função `fit`. Os melhores são definidos como os que têm os menores erros.

"""

class funcFitter:
    
    def __init__(self, subject:str, manobra:str, raw_data, data:np.ndarray, n_point:int=5, estimators:list=["lm"]):
        
        """
            A variável `raw_data` está formatada de forma que a coluna 0 contém as pressões e a 1 contém os volumes.
            A variável `data` contém somente os pontos minímos dos passos, selecionados a partir do raw da seguinte forma: data = raw_data[0::2,:].
        """
        
        self.raw_data    = raw_data.copy()     # Copia dos dados raw.
        self.n_point     = n_point             # Quantidade de Passos (Exemplo: Manobra C tem 4 passos).
        self.manobra     = manobra             # Manobra.
        self.subject     = subject             # Nome do porco.
        self.estimators  = estimators          # Lista de estimadores.
        self.pressures   = data[:,0]           # Seleciona somente as PEEP de cada passo.
        self.volumes     = data[:,1]           # Seleciona somente os volumes minimos de cada passo
        self.pmin        = min(self.pressures) #
        self.interpolate = False
        
    def _interpolate(self, n_interp_point:int):
        last_point = self.pressures[:self.n_point][-1]
        self.interp_pressures = np.linspace(self.pmin, last_point, n_interp_point, endpoint=True)
        interp_func = interp1d(self.pressures[:self.n_point], self.volumes[:self.n_point], kind=self.interp_method)
        self.interp_volumes = interp_func(self.interp_pressures)
        
    def _get_error(self, func, parameters):
        hat_volumes = func(self.pressures[:self.n_point], *parameters)        
        return mean_squared_error(np.array(self.volumes[:self.n_point]), np.array(hat_volumes),squared=False)
        
    def _make_fit_report(self, funcs:list, estimators:list, n_interp_point:int):
        subject = []
        n_point = []
        interp_data = []
        interp_run = []
        data = []
        run = []
        
        cols = ["subject","manobra","n_point","function", "function_name", "estimator", "error", "param", "raw_data"]
        interp_cols = ["subject","manobra","n_point", "function", "function_name", "estimator", "error", "param", "interp_point", "interp_pressure", "interp_volume","raw_data"]
        
        #meus_bounds = ([-np.inf,100,-np.inf,-np.inf],[np.inf,5000,np.inf,np.inf])
        meus_bounds = ([-np.inf,100,10,2],[np.inf,5000,40,15])
        
        for func in funcs:
            for estimator in self.estimators:
                if self.interpolate:
                    for point in range(5, n_interp_point, 5):
                        try:
                            self._interpolate(point)
                            parameters, pcov = curve_fit(func, 
                                                         self.interp_pressures,  
                                                         self.interp_volumes, 
                                                         method=estimator,
                                                         bounds=meus_bounds)
                            err = self._get_error(func=func, parameters=parameters)

                            interp_run.append(self.subject)
                            interp_run.append(self.manobra)
                            interp_run.append(self.n_point)
                            interp_run.append(func)
                            interp_run.append(func.__name__)
                            interp_run.append(estimator)
                            interp_run.append(err)
                            interp_run.append(parameters)
                            interp_run.append(point)
                            interp_run.append(self.interp_pressures)
                            interp_run.append(self.interp_volumes)
                            interp_run.append(self.raw_data)
                            interp_data.append(interp_run)
                            interp_run = []
                            
                        except Exception as e:
                            pass
                else:
                    try:
                        parameters, pcov = curve_fit(func, 
                                                     self.pressures[:self.n_point],  
                                                     self.volumes[:self.n_point], 
                                                     method=estimator)
                        err = self._get_error(func=func, parameters=parameters)

                        run.append(self.subject)
                        run.append(self.manobra)
                        run.append(self.n_point)
                        run.append(func)
                        run.append(func.__name__)
                        run.append(estimator)
                        run.append(err)
                        run.append(parameters)
                        run.append(self.raw_data)
                        data.append(run)
                        run = []
                        
                    except Exception as e:
                        pass
               
        if self.interpolate:             
            return pd.DataFrame(interp_data, columns=interp_cols)    
        else:
            return pd.DataFrame(data, columns=cols)    

    def fit(self, funcs, interpolate:bool=False, n_interp_point:int=30, interp_method:str="linear"):
        
        self.n_interp_point = n_interp_point
        self.interp_method = interp_method           
        self.interpolate = interpolate

        return self._make_fit_report(funcs=funcs, estimators=self.estimators, n_interp_point=n_interp_point)
        
    def make_plot(self, df:pd.DataFrame, n_best = 6):
        
        if len(df) == 0:
            print("Does not exist available plot")
            return None
        
        n_col = 2
        n_row = int(np.ceil(n_best/n_col))
        
        df.reset_index(drop = True, inplace = True)
        best_fits= df["error"].nsmallest(n_best).index
        
        fig, axs = plt.subplots(n_row, n_col, figsize = (5*n_col,4*n_row))

        for row, ax in zip(df.iloc[best_fits].iterrows(), axs.flatten()):
            new_pressures = range(0,100,7)

            _, data = row
            new_pressures = range(0,100,7)
            ax.set_title(f"Model: {data['function_name']} Error: {round(data['error'], 2)}")
            ax.scatter(self.pressures, self.volumes, c = 'r', label = "Original")
            if self.interpolate:
                ax.scatter(data["interp_pressure"], data["interp_volume"], c = 'k', marker = '.', label = "Interpolated")
                ax.text(0.98, 0.6, f"n interp points: {data['interp_point']}",
                horizontalalignment='right',
                verticalalignment='bottom',
                transform = ax.transAxes)
            ax.scatter(new_pressures, data["function"](new_pressures, *data["param"]), c = 'g', marker = '+', label = "Fit")
            ax.set_xlabel('Pressure')
            ax.set_ylabel('Volume')
            ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
            
            ax.text(0.98, 0.5, f"Estimator: {data['estimator']}",
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform = ax.transAxes)
            
        plt.tight_layout()
        plt.show()