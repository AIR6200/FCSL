import numpy as np

class Topsis():
    def __init__(self, evaluation_matrix,minvalue=-1):
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")
        self.row_size = len(self.evaluation_matrix)
        self.column_size = len(self.evaluation_matrix[0])
        self.minvalue_index = minvalue
        self.normalized_matrix = np.array(evaluation_matrix, dtype="float")
        self.score = []

    def min_to_max(self):
        self.evaluation_matrix[:, self.minvalue_index] = np.max(self.evaluation_matrix[:, self.minvalue_index]) - self.evaluation_matrix[:, self.minvalue_index]


    def standard(self):
        n = self.column_size
        for j in range(0, n):
            self.normalized_matrix[:, j] = self.evaluation_matrix[:, j] / np.sqrt(sum(np.square(self.evaluation_matrix[:, j])))#


    def score_compute(self):
        z_max = np.amax(self.normalized_matrix, axis=0)
        z_min = np.amin(self.normalized_matrix, axis=0)
        tmpmaxdist = np.power(np.sum(np.power((z_max - self.normalized_matrix), 2), axis=1), 0.5)
        tmpmindist = np.power(np.sum(np.power((z_min - self.normalized_matrix), 2), axis=1), 0.5)
        score = tmpmindist / (tmpmindist + tmpmaxdist)
        self.score = score / np.sum(score)