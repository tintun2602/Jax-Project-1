
class CournotPlant:
    """
    This plant (controlled system) comes from economics where two rival producers produces the same product (X)
    """

    def __init__(self, q_1, q_2):
        if self.valid(q_1):
            self.q_1 = q_1
        else:
            raise ValueError(f"q_1 is not valid")

        if self.valid(q_2):
            self.q_2 = q_2
        else:
            raise ValueError(f"q_2 is not valid")

        if self.valid(q_1) and self.valid(q_2):
            self.q = (self.q_1 + self.q_2)
        else:
            raise ValueError(f"Either q_1 or q_2 are not valid")

    def step(self, input):
        pass

    def valid(self, q_n):
        """
        Checks if  0 ≤ q_2 ≤ 1
        :param q_n: check if q_1 or q_2 are valid
        :return: boolean, true or false
        """
        return 0 <= q_n <= 1

    def price(self, q):
        #return (p_max - q)
        pass



if __name__ == "__main__":
    CP = CournotPlant(2,1)
    

