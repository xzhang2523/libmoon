from typing import Any
import numpy as np 



class termination: 

    def __init__(self) -> None:
        pass

    def __call__(self, 
                 nfe: int=0, 
                 gen: int=0, 
                 ) -> bool:
        self.nfe += nfe 
        self.gen += gen 

    def setup(self, 
              nfe: int=0, 
              gen: int=0, 
              max_nfe: int=np.inf, 
              max_gen: int=np.inf, 
              ) -> None: 
        self.nfe = nfe 
        self.gen = gen 
        self.max_nfe = max_nfe 
        self.max_gen = max_gen 

    @property
    def has_next(self) -> bool: 
        return (self.nfe < self.max_nfe) and (self.gen < self.max_gen)
    
