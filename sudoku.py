import numpy as np
import cvxopt
import cvxopt.glpk # GNU Linear Programming Kit
from cvxopt import matrix


class Sudoku:


    def __init__(self, table, hex=False):
        
        self.table = table
        self.size = len(table)
        self.subsize = int(np.sqrt(self.size))

        self.constraint_matrix = np.zeros((4*self.size*self.size, self.size*self.size*self.size))
        self.constraint_c = 0

        self._init_numbers()

    
    def _init_numbers(self):
        for line in range(self.size):
            for column in range(self.size):
                number = self.table[line, column]
                if number>0:
                    new_row=np.zeros(self.size*self.size*self.size)
                    new_row[self.unravel(line, column, number-1)]=1
                    self.constraint_matrix = np.vstack((self.constraint_matrix, new_row))


    def unravel(self, i, j, k):
        """
        Associate a unique variable index given a 3-index (ijk) 
        """
        assert(i>=0 and i<self.size)
        assert(j>=0 and i<self.size)
        assert(k>=0 and i<self.size)
        
        return(k+ j*self.size+ i*self.size*self.size)


    def line_constraints(self):
        for number in range(self.size): 
            for column in range(self.size):
                for line in range(self.size): 
                    self.constraint_matrix[self.constraint_c, self.unravel(line, column, number)] = 1
                self.constraint_c +=1


    def column_constraints(self):
        for number in range(self.size): 
            for line in range(self.size):
                for column in range(self.size): 
                    self.constraint_matrix[self.constraint_c, self.unravel(line, column, number)] = 1
                self.constraint_c += 1


    def number_constraints(self):
        for line in range(self.size): 
            for column in range(self.size):
                for number in range(self.size): 
                    self.constraint_matrix[self.constraint_c, self.unravel(line, column, number)] = 1
                self.constraint_c += 1

    def subsquare_constraints(self):
        for number in range(self.size):
            for subsquare_x in range(self.subsize): 
                for subsquare_y in range(self.subsize):
                    for line in range(self.subsize*subsquare_x, self.subsize*subsquare_x+self.subsize): 
                        for column in range(self.subsize*subsquare_y, self.subsize*subsquare_y+self.subsize):
                            self.constraint_matrix[self.constraint_c, self.unravel(line, column, number)] = 1
                    self.constraint_c += 1


    def get_constraints(self):
        self.line_constraints()
        self.column_constraints()
        self.number_constraints()
        self.subsquare_constraints()
        

    def test_constraints(self):
        for constraint in range(self.constraint_c):
            if (np.sum(self.constraint_matrix[constraint,])!=self.size):
                print("Error on constraint: ", constraint)
                break
        print("All constraints OK!")
        return
    

    def solve(self):
        """
        minimize    c'*x
        subject to  G*x <= h
                    A*x = b
                    x[k] is integer for k in I
                    x[k] is binary for k in B
        """

        A = self.constraint_matrix

        b = matrix(np.ones(A.shape[0])) ## set partition
        c = matrix(np.zeros(A.shape[1])) ## zero cost

        G = matrix(np.zeros(A.shape))
        h = matrix(np.zeros(A.shape[0]))

        binary = np.array(range(A.shape[1]))
        I = set(binary)

        B = set(range(A.shape[1]))

        Aeq = matrix(A)
        (status, solution) = cvxopt.glpk.ilp(c=c, G=G, h=h, A=Aeq, b=b, B=set(range(A.shape[1])))

        self.status = status
        self.solution = solution

        print(f"Solver status: {status}")

    
    def print(self, hex_flag=False):
        
        sep = '+'
        for _ in range(self.subsize):
            sep += '-' * (2*self.subsize-1)
            sep += '+'

        for line in range(self.size):
            if line%3 == 0:
                print(sep)
            print("|", end='')
            for column in range(self.size):
                for number in range(self.size):
                    if self.solution[self.unravel(line, column, number)]==1:
                        if hex_flag: print(hex(number+1)[-1], end='')
                        else: print(number+1, end='')
                if column%3 ==2:
                    print("|", end='')
                else:
                    print(" ", end='')
            print("")
        print(sep)