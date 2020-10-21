import numpy as np

###########################
#Functions
class pigClass:
    
    def __init__(self, name:str):
        self.name = name
        self.manobra = {}
    def append(self,
               manobra:str, 
               parameters:list, 
               p_i:list, 
               p_e:list, 
               w_i:list, 
               w_e:list):
        
        data = {"parameters": parameters,
                "p_i": p_i, 
                "p_e": p_e,
                "w_i": w_i,
                "w_e": w_e}
        
        self.manobra[manobra] =  data
        
