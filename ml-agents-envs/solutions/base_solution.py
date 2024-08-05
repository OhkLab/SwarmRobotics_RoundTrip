import abc


class BaseSolution(abc.ABC):
    @abc.abstractmethod
    def get_action(self, obs):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def set_params(self, params):
        raise NotImplementedError()

    def get_num_params(self):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()