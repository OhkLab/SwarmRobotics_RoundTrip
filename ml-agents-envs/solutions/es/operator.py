from solutions.es.es import CMAES, SNES


class ESOperator():
    def __init__(self, x0, dim, popsize, sigma0, algo_number):
        self.x0 = x0
        self.dim = dim
        self.popsize = popsize
        self.sigma0 = sigma0
        self.algo_number = algo_number

    def get_solver(self):
        if self.algo_number == 0:
            solver = CMAES(
                num_params=self.dim,
                init_params=self.x0,
                sigma_init=self.sigma0,
                popsize=self.popsize,
                weight_decay=0.01
            )
        elif self.algo_number == 1:
            solver = SNES(
                x0=self.x0,
                popsize=self.popsize,
                sigma_init=self.sigma0
            )
        else:
            raise NotImplementedError()
        
        return solver