import numpy as np
import math
from scipy.cluster.vq import kmeans2
from sklearn.cross_validation import train_test_split

class Solution() :
    def __init__(self, filename,ubit_name, person_number):
        self.filename = filename
        self.ubit_name = ubit_name
        self.person_number = person_number
        self.syn_input_data =[]
        self.syn_output_data =[]
        self.letor_input_data = []
        self.letor_output_data = []

    def read_data(self):
        self.syn_input_data = np.genfromtxt(filename, delimiter=',')
        self.syn_output_data = np.genfromtxt('/media/parik/New Volume/ML/Project2/output.csv', delimiter=',').reshape([-1, 1])
        self.letor_input_data = np.genfromtxt('/media/parik/New Volume/ML/Project2/Querylevelnorm_X.csv', delimiter=',')
        self.letor_output_data = np.genfromtxt('/media/parik/New Volume/ML/Project2/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])
        self.letor_input_train, self.letor_input_test, self.letor_output_train, self.letor_output_test = train_test_split(self.letor_input_data, self.letor_output_data, test_size=0.2)
        self.letor_input_test, self.letor_input_val, self.letor_output_test, self.letor_output_val = train_test_split(self.letor_input_test, self.letor_output_test, test_size=0.5)
        print(self.letor_output_train.shape)

ubit_name = 'pdeshmuk'
person_number = '50247649'
filename = '/media/parik/New Volume/ML/Project2/input.csv'
sol = Solution(filename,ubit_name, person_number)
sol.read_data()
