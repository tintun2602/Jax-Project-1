from abc import abstractmethod


class BasePlant:
    def step(self, input):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        pass


    @abstractmethod
    def reset_to_initial_state(self):
        pass




