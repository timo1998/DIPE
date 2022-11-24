import numpy as np 

class Matrix():
    
    #Constructor 
    def __init__(self, r, c):
        self.r = r 
        self.c = c
        self.data = [None] * r * c
        
    # Methods 
    def __add__(self, other):
        M = Matrix(self.r,self.c)
        for i in range(self.r):
            for j in range(self.c):
                M[i,j] = self[i, j] + other[i, j]
        return M

    def __sub__(self, other):
        M = Matrix(self.r, self.c)
        other = other * -1
        M = self + other
        return M
        
    def __mul__(self, other):
        if isinstance(other, Matrix):
            if other.c == 1:
                M = Matrix(self.r,1)
                for i in range(self.r):
                        M[i,0] = sum(self[i, k] * other[k,0] for k in range(other.r))
            else:
                M = Matrix(self.r,other.c)
                for i in range(self.r):
                    for j in range(other.c):
                        M[i,j] = sum(self[i, k] * other[k, j] for k in range(other.r))
        else:
            M = Matrix(self.r,self.c)
            for i in range(self.r):
                for j in range(self.c):
                    M[i,j] = self[i, j] * other
        return M

    def __repr__(self):
        str_ = '\n'.join(
               '\t'.join(
                f'{self[i,j]}'
                for j in range(self.c))
                for i in range(self.r))
        return str_
    
    def __setitem__(self, indices, value):
        i = indices[0]
        j = indices[1]
        self.data[i * self.c + j] = value

    def __getitem__(self, indices):
        i = indices[0]
        j = indices[1]
        return self.data[i * self.c + j] 
    
    @property  
    def sigmoid(self):
        M = Matrix(self.r,self.c)
        for i in range(self.r):
            for j in range(self.c):
                M[i,j] = 1/(1 + np.exp(-self[i,j]))
        return M
    
    @property 
    def T(self):
        M = Matrix(self.c,self.r)
        for i in range(self.r):
            for j in range(self.c):
                M[j,i] = self[i,j]
        return M

class Vector(Matrix):

    # Constructor
    def __init__(self, r, c):
        super().__init__(r, c)
        self.r = r
        self.c = 1
        self.data = [None] * r

    def __setitem__(self, indices, value):
        i = indices
        self.data[i * self.c] = value

    def __getitem__(self, indices):
        i = indices
        return self.data[i * self.c]

    def __repr__(self):
        str_ = '\n'.join(
                f'{self[i]}'
                for i in range(self.r))
        return str_
    

