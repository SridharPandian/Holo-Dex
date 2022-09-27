import torch

class KNearestNeighbors(object):
    def __init__(self, input_values, output_values):
        self.input_values = input_values.cpu()
        self.output_values = output_values.cpu()

    def _get_sorted_idxs(self, datapoint):
        l1_distances = self.input_values - datapoint
        l2_distances = torch.norm(l1_distances, dim = 1)

        sorted_idxs = torch.argsort(l2_distances)
        return sorted_idxs

    def get_nearest_neighbor(self, datapoint):
        sorted_idxs = self._get_sorted_idxs(datapoint)
        nn_idx = sorted_idxs[0]
        return self.output_values[nn_idx], nn_idx

    def get_k_nearest_neighbors(self, datapoint, k):
        if k == 1:
            return self.get_nearest_neighbor(datapoint)

        assert datapoint.shape == self.input_values[0].shape

        sorted_idxs = self._get_sorted_idxs(datapoint)
        k_nn_idxs = sorted_idxs[:k]
        return self.output_values[k_nn_idxs], k_nn_idxs