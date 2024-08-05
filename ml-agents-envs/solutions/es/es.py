import numpy as np


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


def compute_utilities(fitnesses):
    L = len(fitnesses)
    ranks = np.zeros_like(fitnesses)
    l = list(zip(fitnesses, range(L)))
    l.sort()
    for i, (_, j) in enumerate(l):
        ranks[j] = i
    # smooth reshaping
    utilities = np.array([max(0., x) for x in np.log(L / 2. + 1.0) - np.log(L - np.array(ranks))])
    # make the utilities sum to 1
    utilities /= sum(utilities)
    # baseline
    utilities -= 1. / L
    return utilities


# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


class CMAES:
    """CMA-ES wrapper."""

    def __init__(self, num_params,  # number of model parameters
                 init_params,
                 sigma_init=0.10,  # initial standard deviation
                 popsize=255,  # population size
                 weight_decay=0.01):  # weight decay coefficient

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None

        import cma
        self.es = cma.CMAEvolutionStrategy(init_params,
                                           self.sigma_init,
                                           {'popsize': self.popsize,
                                            })

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        """returns a list of parameters"""
        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = np.array(reward_table_result)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        self.es.tell(self.solutions, (-reward_table).tolist())  # convert minimizer to maximizer.

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return r[0], -r[1], -r[1], r[6]
    
    def get_sigma(self):
        return np.mean(self.es.result[6]), np.std(self.es.result[6])
    

class SNES():
    def __init__(self, x0, popsize, sigma_init=0.1):
        self.x0 = x0
        self.batchSize = popsize
        self.dim = len(x0)
        self.learningRate = 0.2 * (3  + np.log(self.dim)) / np.sqrt(self.dim)
        self.bestFound = None
        self.sigmas = np.ones(self.dim) * sigma_init
        self.center = x0.copy()

    def ask(self):
        self.samples = [np.random.randn(self.dim) for _ in range(self.batchSize)]
        asked = [(self.sigmas * s + self.center) for s in self.samples]
        self.asked = asked
        return asked

    def tell(self, fitnesses):
        samples = self.samples
        self.bestFound = self.asked[np.argmax(fitnesses)]

        # update center and variances
        utilities = compute_utilities(fitnesses)
        self.center += self.sigmas * np.dot(utilities, samples)
        covGradient = np.dot(utilities, [s ** 2 - 1 for s in samples])
        self.sigmas = self.sigmas * np.exp(0.5 * self.learningRate * covGradient)

    def best_param(self):
        return self.bestFound
    
    def get_sigma(self):
        return np.mean(self.sigmas), np.std(self.sigmas)