class Vector:
    def __init__(self, values):
        if isinstance(values, list) == True:
            self.data = values
            self.size = len(values)
        else:
            print("Vector init error")

    def vector_addition(self, vector):
        if self.size != vector.size:
            print("ERROR vector addition")
            return
        i = 0
        while i < self.size:
            self.values[i] = float(self.values[i]) + float(vector.values[i])
            i += 1
        return

    def scalar_addition(self, value):
        if isinstance(value, (int, float)) == False:
            print("ERROR vector addition")
            return
        i = 0
        while i < self.size:
            self.values[i] = float(self.values[i]) + value
            i += 1
        return

    def vector_substraction(self, vector):
        if self.size != vector.size:
            print("ERROR vector substraction")
            return
        i = 0
        while i < self.size:
            self.values[i] = float(self.values[i]) - float(vector.values[i])
            i += 1
        return

    def scalar_substraction(self, value):
        if isinstance(value, (int, float)) == False:
            print("ERROR vector substraction")
            return
        i = 0
        while i < self.size:
            self.values[i] = float(self.values[i]) - value
            i += 1
        return


    def scalar_division(self, value):
        if isinstance(value, (int, float)) == False:
            print("ERROR vector division")
            return
        i = 0
        while i < self.size:
            self.values[i] = float(self.values[i]) / value
            i += 1
        return

    def dot_product(self, values):
        if self.size != vector.size:
            print("ERROR vector substraction")
            return -1
        i = 0
        result = 0
        while i < self.size:
            result += (float(self.values[i]) * float(vector.values[i]))
            i += 1
        return result


    def scalar_multiplication(self, value):
        if isinstance(value, (int, float)) == False:
            print("ERROR vector multiplication")
            return
        i = 0
        while i < self.size:
            self.values[i] = float(self.values[i]) * value
            i += 1
        return
