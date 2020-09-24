from lib.vector import *

class Matrix:
    def __init__(self, matrix):
        if isinstance(matrix, list) == False:
            print("ERROR in matrix init")
            return 0
        for row in matrix:
            if len(row) != len(matrix[0]) or isinstance(row, list) == False:
                print("ERROR in matrix init")
                return 0
        self.data = matrix
        self.shape = (len(matrix), len(matrix[0]))

    def addition(self, matrix):
        if self.shape[0] != matrix.shape[0] or self.shape[1] != matrix.shape[1]:
            print("ERROR matrix addition")
            return
        i = 0
        l = 0
        while l < self.shape[0]:
            i = 0
            while i < self.shape[1]:
                self.data[l][i] = float(self.data[l][i]) + float(matrix.data[l][i])
                i += 1
            l += 1
        return

    def substraction(self, vector):
        if self.shape[0] != matrix.shape[0] or self.shape[1] != matrix.shape[1]:
            print("ERROR matrix substraction")
            return
        i = 0
        l = 0
        while l < self.shape[0]:
            i = 0
            while i < self.shape[1]:
                self.data[l][i] = float(self.data[l][i]) - float(matrix.data[l][i])
                i += 1
            l += 1
        return


    def scalar_division(self, value):
        if isinstance(value, (int, float)) == False:
            print("ERROR matrix division")
            return
        i = 0
        l = 0
        while l < self.shape[0]:
            i = 0
            while i < self.shape[1]:
                self.data[l][i] = float(self.data[l][i]) / value
                i += 1
            l += 1
        return

    def multiplication(self, matrix):
        if self.shape[1] != matrix.shape[0]:
            print("ERROR matrix multiplication")
            return -1
        i = 0
        l = 0
        new_vectors = []
        matrix = matrix.transpose()
        for vector in matrix:
            new_vectors.append(self.vector_multiplication(Vector(vector)))
        new_vectors = matrix(new_vectors)
        new_vectors = new_vector.transpose()
        return new_vectors


    def scalar_multiplication(self, value):
        if isinstance(value, (int, float)) == False:
            print("ERROR matrix multiplication")
            return
        i = 0
        l = 0
        while l < self.shape[0]:
            i = 0
            while i < self.shape[1]:
                self.data[l][i] = float(self.data[l][i]) * value
                i += 1
            l += 1
        return

    def vector_multiplication(self, vector):
        if self.shape[1] != vector.size:
            print("ERROR vector matrix multiplication")
            return
        l = 0
        new_vector = []
        while l < self.shape[0]:
            i = 0
            value = 0
            while i < self.shape[1]:
                value += (float(self.data[l][i]) * float(vector.data[i]))
                i += 1
            new_vector.append(value)
            l += 1
        return new_vector

    def transpose(self):
        i = 0
        l = 0
        column = []
        new = []
        while i < self.shape[0]:
            i = 0
            while l < self.shape[1]:
                column.append(self.data[l][i])
                l += 1
            new.append(column)
            i += 1
        return new
