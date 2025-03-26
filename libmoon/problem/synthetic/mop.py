import numpy as np
import torch


class BaseMOP():
    def __init__(self,
                 n_var: int,
                 n_obj: int,
                 lbound: np.ndarray=None,
                 ubound: np.ndarray=None,
                 n_cons: int=0,
                 ) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_cons = n_cons
        if type(lbound) != type(None):
            self.lbound = lbound
        if type(ubound) != type(None):
            self.ubound = ubound
    @property
    def get_number_variable(self) -> int:
        return self.n_var

    @property
    def get_number_objective(self) -> int:
        return self.n_obj

    @property
    def get_lower_bound(self) -> np.ndarray:
        return self.lbound

    @property
    def get_upper_bound(self) -> np.ndarray:
        return self.ubound

    @property
    def has_constraint(self) -> bool:
        return self.n_cons > 0

    def evaluate(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x: any) -> any:
        """
            Evaluate the objectives for x
            Parameters
            ----------
            x : any
                Tensor or ndarray
            Returns
            -------
            any
                Tensor or ndarray correspondingly
            Raises
            ------
            ValueError
                wrong type of x
        """
        if type(x) == torch.Tensor:
            return self._evaluate_torch(torch.atleast_2d(x))
        elif isinstance(x, np.ndarray):
            return self._evaluate_numpy(np.atleast_2d(x))
        else:
            raise ValueError("Input has to be in the form of Tensor or ndarray!")

    # def get_pf(self, n_points: int=100) -> np.ndarray:
    #     """
    #     Get Pareto front
    #     Parameters
    #     ----------
    #     num_points : int, optional
    #         _description_, by default 100
    #     Returns
    #     -------
    #     np.ndarray
    #         _description_
    #     """
    #     # TODO
    #     # if method=='uniform':
    #     if hasattr(self, "_get_pf"): return self._get_pf(n_points)
    #     else: raise NotImplementedError("Subclasses should implement this method.")



class mop_noCons(BaseMOP):

    def __init__(self, n_var: int, n_obj: int, lbound: np.ndarray, ubound: np.ndarray, n_cons: int = 0) -> None:
        super().__init__(n_var, n_obj, lbound, ubound, n_cons)