# Este arquivo contém as funções usadas para ajustar as curvas PV
# e outras funções úteis

############################################################### BIBLIOTECAS:
import numpy as np         # para fazer contas e mexer com matrizes
import pandas as pd        # para montar DataFrames (tabelas de bancos de dados)

from pathlib import Path   # para trabalhar com diretorios e arquivos
import pickle              # para gravar e ler dados

import matplotlib.pyplot as plt        # para gráficos

from scipy.optimize import curve_fit   # para ajuste das curvas dos modelos

import math                            # para erf()
from scipy.interpolate import interp1d # para interpolar os pontos PV


############################################################### MODELOS:

# função usada para fitar o modelo PV sigmoide (doente)
#                b                                   b
# V(x) = a + ----------------------   = a + ------------------------
#            1 + exp(-(x/d) + (c/d)          1 + exp(-x/d).exp(c/d)
#
# lim (x-> inf)  V(x) = a + b
def sigmoidvenegas1(x, a, b, c, d):
    return a + b/(1 + np.exp(-(x-c)/d))

def sigmoidpaiva(x,TLC,k1,k2):
    return TLC/(1+(k1*np.exp(-k2*x)))

# modificação nossa: incluindo offset
def sigmoidpaivaoffset(x,TLC,k1,k2,offset):
    return TLC/(1+(k1*np.exp(-k2*x))) + offset

# baseado no artigo original do paiva1975, e incluindo offset:
def sigmoidpaivaoffset2(x,TLC,k1,k2,offset):
    return TLC/(1+(k1*TLC*np.exp(-k2*x))) + offset

def sigmoidvenegas2(x,TLC,B,k,c,d):
    return (TLC-(B*np.exp(-k*x)))/(1 + np.exp(-(x-c)/d))

# modificação nossa: incluindo offset
def sigmoidvenegas2offset(x,TLC,B,k,c,d,offset):
    return (TLC-(B*np.exp(-k*x)))/(1 + np.exp(-(x-c)/d)) + offset

# Modelo exponencial simples de curva PV pulmonar (Salazar 1964)
# Volume = Vmax*(1-e^(-K*Paw))
# Paw = pressão na via aérea
# K = 'constante de tempo' da exponencial
def expsalazar(x,Vo,K):
    return Vo*(1-np.exp(-K*x))

# modelo de unidades recrutadas com erf()
# ajustando a função para uma entrada array (para curve_fit)
def meu_erf_vec(Paw,mi,sigma):
    saida_lst = []
    for x_in in Paw:
        x = (x_in-mi)/(sigma*1.5)
        merf = math.erf(x)
        saida_lst.append((merf/2)+0.5)
    return np.array(saida_lst)

# modelo proposto pelo grupo (nós)
def sigmoid_recruit_units(Paw,K,Vmax,mi,sigma,offset):
    Vmax_recrutado = Vmax*meu_erf_vec(Paw,mi,sigma)
    V = Vmax_recrutado*(1-np.exp(-K*Paw)) + offset
    return V


############################################################### FUNÇÕES:

'''
Carrega os arquivos .pickle das subpastas da pasta './porquinhos/'
e retorna um DataFrame com os dados.

As manobras C contém apenas 4 passos, e as D, apenas 5 passos.
'''
def carrega_pickles(folder = 'porquinhos'):
    dataframes_lst = [] # lista de dataframe: Cada elemento da lista corresponde a um dataframe de um porco/manobra/dados PV

    for file_name in Path(folder).rglob('*.pickle'):

        print(f"\rLendo {file_name.name}\t\t\t")

        with open(file_name, "rb") as file: # abre o arquivo.pickle

            porquinho = pickle.load(file)
            for manobra in porquinho: #Para cada manobra 

                if manobra == "D": # Posso fazer 3,4,5 passos
                    n_steps = 5
                elif manobra == "C": # Posso fazer 3,4 passos
                    n_steps = 4
                elif manobra == "B": # Posso fazer 3 passos
                    n_steps = 3

                # Formato os dados de entrada
                format_data = []

                for pi, pe, wi, we in zip(porquinho[manobra]["p_i"], porquinho[manobra]["p_e"],
                                          porquinho[manobra]["w_i"], porquinho[manobra]["w_e"]):

                    format_data.extend([pi,wi,pe,we])

                format_data = np.array(format_data).reshape(-1,2) # monta matriz de N linhas e 2 colunas


                ##########################################################
                caso = []
                caso.append(porquinho.name)
                caso.append(manobra)
                caso.append(format_data)
                caso.append(n_steps)
                casodf = pd.DataFrame(caso, index = ['Animal', 'Manobra', 'Dados', 'n_steps']).T
                dataframes_lst.append(casodf)    
    
    # Junta todos os dataframes da lista em um único DataFrame:
    dadosdf = pd.concat(dataframes_lst, ignore_index=True)
    
    # Extrai os dados de pressão e volume dos dados raw dos arquivos pickle:
    pv_lst = []
    for idx,caso in dadosdf.iterrows():
        pv = []
        ps,vs = Data2PV(caso.Dados)
        pv.append(ps)
        pv.append(vs)
        pvdf = pd.DataFrame([pv], columns = ['Pressoes', 'Volumes'])
        pv_lst.append(pvdf)
        
    pvdf_all = pd.concat(pv_lst, ignore_index=True)
    
    dadosdf_completo = pd.concat((dadosdf,pvdf_all),axis=1)
    
    # inclui uma coluna para volume esperado...
    dadosdf_completo["volume_esperado"] = 0
    
    return dadosdf_completo

'''
Retorna os vetores de pressão e volume a partir dos dados raw disponíveis nos pickles
'''
def Data2PV(data):
    data2 = data[0::2, :]
    pressures = data2[:,0]
    volumes = data2[:,1]
    return pressures,volumes


# metodos : lm, dogbox, trf
def testa_modelo(df, modelo, meu_p0 = [], metodo = 'lm', n_colunas = 4, texto = '', TLC_index = 0, meus_bounds = [], n_points_interp=0, debug=True): 
    numero_de_casos = len(df)
    fig = plt.figure(figsize=(25,5*numero_de_casos/n_colunas))
    
    erro_vec = []
    n_fitted = 0
    
    for caso_teste in range(numero_de_casos):
        
        p_in = df.iloc[caso_teste].Pressoes
        v_in = df.iloc[caso_teste].Volumes

        # interpola pontos (se n_points_interp==0, a função não interpola)
        p, v = interpola_PV(p_in,v_in,n_points_interp)

        plt.subplot(int(numero_de_casos/n_colunas)+1,n_colunas,caso_teste+1)
        fig.tight_layout()
        if (n_points_interp > 0):
            plt.scatter(p,v,label='interp',c='k',marker='x')
        plt.scatter(p_in,v_in,label='raw')
        try:
            if (meu_p0 == []):                          # sem p0
                if (meus_bounds == []): # sem bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo)
                else:                   # com bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo, bounds=meus_bounds)
            else:                                       # com p0
                if (meus_bounds == []): # sem bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo, p0 = meu_p0)
                else:                   # com bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo, p0 = meu_p0, bounds=meus_bounds)
            if debug:
                textop = ""
                for p in parameters:
                    if ( np.abs(p) > 1 ):
                        textop = textop + f'{p:7.1f}' + ' '
                    else:
                        textop = textop + f'{p:.3f}' + ' '
                print(f'Testando caso {caso_teste}: {df.iloc[caso_teste].Animal}: [{textop}]')
            meu_p = range(1,100)
            meu_v = modelo(meu_p,*parameters)
            plt.plot(meu_p,meu_v,'r',label='fit')
            n_fitted = n_fitted + 1
            if ( df.iloc[caso_teste]["volume_esperado"] == 0 ):
                plt.title(f'Case: {df.iloc[caso_teste].Animal}. TLC = {parameters[TLC_index]:.0f} mL')
            else:
                v_esperado = df.iloc[caso_teste]["volume_esperado"]
                erro = 100*(parameters[TLC_index]-v_esperado)/v_esperado
                erro_vec.append(erro)
                plt.title(f'Case: {df.iloc[caso_teste].Animal}. TLC = {parameters[TLC_index]:.0f} mL. Error: {erro:.1f}%')
        except Exception as e:
            print(f'\tCaso {caso_teste} ({df.iloc[caso_teste].Animal}) deu erro...')
            plt.title(f'Case: {df.iloc[caso_teste].Animal}. Error fitting.')

        plt.xlabel('Pressure [cmH2O]')
        plt.ylabel('Volume [mL]')
        plt.legend()
    
    fig.suptitle(f'PV Graph. Model: {modelo.__name__}. {texto}', fontsize=16, y=1.05)
    plt.show()
    
    if ( len(erro_vec) > 0 ):
        erro_medio = np.mean(np.abs(erro_vec))
        erro_norm = np.linalg.norm(erro_vec)
    else:
        erro_medio = -1
        erro_norm = -1
        
    if debug:
        print(f'Norma(erro): {erro_norm:.1f}. Erro médio: {erro_medio:.2f}%. Ajustados: {n_fitted}.')
    
    return erro_norm, erro_medio, n_fitted

# o mesmo que a função anterior, mas não mostra gráficos ou mensagens... para uso dentro de loops...
# metodos : lm, dogbox, trf
def testa_modelo_loop(df, modelo, meu_p0 = [], metodo = 'lm', n_colunas = 4, texto = '', TLC_index = 0, meus_bounds = [], n_points_interp=0): 
    numero_de_casos = len(df)
    
    erro_vec = []
    n_fitted = 0
    
    for caso_teste in range(numero_de_casos):
        
        p_in = df.iloc[caso_teste].Pressoes
        v_in = df.iloc[caso_teste].Volumes

        # interpola pontos (se n_points_interp==0, a função não interpola)
        p, v = interpola_PV(p_in,v_in,n_points_interp)

        try:
            if (meu_p0 == []):                          # sem p0
                if (meus_bounds == []): # sem bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo)
                else:                   # com bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo, bounds=meus_bounds)
            else:                                       # com p0
                if (meus_bounds == []): # sem bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo, p0 = meu_p0)
                else:                   # com bounds
                    parameters, pcov = curve_fit(modelo, p, v, method=metodo, p0 = meu_p0, bounds=meus_bounds)

            if ( df.iloc[caso_teste]["volume_esperado"] == 0 ):
                pass
            else:
                v_esperado = df.iloc[caso_teste]["volume_esperado"]
                erro = 100*(parameters[TLC_index]-v_esperado)/v_esperado
                erro_vec.append(erro)
                
            n_fitted = n_fitted + 1
            if ( (metodo=='lm') & (parameters[TLC_index] > 6000) ): # não fitou...
                n_fitted = n_fitted - 1

        except Exception as e:
            pass

    if ( len(erro_vec) > 0 ):
        erro_medio = np.mean(np.abs(erro_vec))
        erro_norm = np.linalg.norm(erro_vec)
    else:
        erro_medio = -1
        erro_norm = -1
        
    return erro_norm, erro_medio, n_fitted

# interpola vetores PV
# n_points = número de pontos intermediários
def interpola_PV(pressoes,volumes,n_points=0):
    if len(pressoes)<3:
        kind = "linear"
    elif len(pressoes)==3:
        kind = "quadratic"
    else:
        kind = "cubic"
    interp_pressures = np.linspace(pressoes[0], pressoes[-1], (len(pressoes)*(n_points+1))-n_points, endpoint=True)
    interp_func = interp1d(pressoes, volumes, kind=kind)
    interp_volumes = interp_func(interp_pressures)
    return interp_pressures, interp_volumes

# Classe usada para dados dos modelos usados na função testa_varios
class dados_modelos:
    model_function = ''
    TLC_index = ''
    p0 = ''
    bounds = ''
    
def testa_varios(dadosdf, modelos, metodos = ('lm','trf','dogbox'), vec_interp = [0, 1, 2, 10, 20]):
    df_lst = []
    for mod in modelos:
        print(f'Rodando {mod.model_function.__name__}')
        for n_points_interp in vec_interp:
            for metodo in metodos:
                if (metodo == 'lm'):
                    erro_norm, erro_medio, n_fitted = testa_modelo_loop(dadosdf, mod.model_function, metodo = metodo, meu_p0 = mod.p0,
                                                                 TLC_index=mod.TLC_index, n_points_interp=n_points_interp)
                else:
                    erro_norm, erro_medio, n_fitted = testa_modelo_loop(dadosdf, mod.model_function, metodo = metodo, meu_p0 = mod.p0,
                                                                 TLC_index=mod.TLC_index, meus_bounds=mod.bounds,
                                                                 n_points_interp=n_points_interp)
                caso = []
                caso.append(mod.model_function.__name__)
                caso.append(metodo)
                caso.append(n_points_interp)
                caso.append(erro_norm)
                caso.append(erro_medio)
                caso.append(n_fitted)
                casodf = pd.DataFrame(caso, index = ['Modelo', 'Método', 'N_points_interp', 'Norma erro', 'Erro médio', 'n_fitted']).T
                df_lst.append(casodf)
    dadosdf = pd.concat(df_lst, ignore_index=True)
    return dadosdf