import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef class TreeNode:
    cdef int feature_index
    cdef double split_value
    cdef object left
    cdef object right
    cdef int is_terminal
    cdef double prediction

    def __init__(self):
        self.feature_index = -1
        self.split_value = 0.0
        self.left = None
        self.right = None
        self.is_terminal = 0
        self.prediction = 0.0

def bin_features(np.ndarray[np.float64_t, ndim=2] X, int n_bins):
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef np.ndarray[np.int32_t, ndim=2] binned_X = np.zeros_like(X, dtype=np.int32)
    cdef list bins = []
    cdef np.ndarray[np.float64_t, ndim=1] bin_edges
    cdef int feature_index, bin_index

    for feature_index in range(n_features):
        # Compute bin edges based on quantiles
        bin_edges = np.percentile(X[:, feature_index], np.linspace(0, 100, n_bins + 1))
        bins.append(bin_edges)
        binned_X[:, feature_index] = np.digitize(X[:, feature_index], bin_edges) - 1

    return binned_X, bins

def compute_histogram_gradients(np.ndarray[np.int32_t, ndim=2] X_binned, np.ndarray[np.float64_t, ndim=1] gradients, int n_bins):
    cdef int n_samples = X_binned.shape[0]
    cdef int n_features = X_binned.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] hist_grad = np.zeros((n_features, n_bins))

    cdef int feature_index, bin_index, i
    for feature_index in prange(n_features, nogil=True):
        for bin_index in range(n_bins):
            for i in range(n_samples):
                if X_binned[i, feature_index] == bin_index:
                    hist_grad[feature_index, bin_index] += gradients[i]
    return hist_grad

def compute_histogram_hessians(np.ndarray[np.int32_t, ndim=2] X_binned, np.ndarray[np.float64_t, ndim=1] hessians, int n_bins):
    cdef int n_samples = X_binned.shape[0]
    cdef int n_features = X_binned.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] hist_hess = np.zeros((n_features, n_bins))

    cdef int feature_index, bin_index, i
    for feature_index in prange(n_features, nogil=True):
        for bin_index in range(n_bins):
            for i in range(n_samples):
                if X_binned[i, feature_index] == bin_index:
                    hist_hess[feature_index, bin_index] += hessians[i]
    return hist_hess

def find_best_split_histogram(np.ndarray[np.float64_t, ndim=2] hist_grad, np.ndarray[np.float64_t, ndim=2] hist_hess, int n_bins, int min_samples_split, double lambda_):
    cdef double best_gain = -float('inf')
    cdef int best_feature_index = -1
    cdef int best_split_value = -1

    # Precompute total sums of gradients and hessians
    cdef np.ndarray[np.float64_t, ndim=1] total_grad_sum = np.sum(hist_grad, axis=1)
    cdef np.ndarray[np.float64_t, ndim=1] total_hess_sum = np.sum(hist_hess, axis=1)

    cdef int feature_index, bin_index
    cdef double left_grad_sum, left_hess_sum, right_grad_sum, right_hess_sum, gain, parent_gain

    for feature_index in range(hist_grad.shape[0]):
        left_grad_sum = 0
        left_hess_sum = 0

        parent_gain = (total_grad_sum[feature_index] ** 2) / (total_hess_sum[feature_index] + lambda_)

        for bin_index in range(0, n_bins):
            left_grad_sum += hist_grad[feature_index, bin_index]
            left_hess_sum += hist_hess[feature_index, bin_index]

            right_grad_sum = total_grad_sum[feature_index] - left_grad_sum
            right_hess_sum = total_hess_sum[feature_index] - left_hess_sum

            if left_hess_sum < min_samples_split or right_hess_sum < min_samples_split:
                continue

            gain = (left_grad_sum ** 2) / (left_hess_sum + lambda_) + (right_grad_sum ** 2) / (right_hess_sum + lambda_) - parent_gain

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_split_value = bin_index

    return best_feature_index, best_split_value

cdef class GBDT:
    cdef int n_estimators
    cdef int max_depth
    cdef int min_samples_split
    cdef double learning_rate
    cdef int n_bins
    cdef list trees
    cdef double initial_prediction

    def __init__(self, int n_estimators=100, int max_depth=3, int min_samples_split=2, double learning_rate=0.1, int n_bins=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.n_bins = n_bins
        self.trees = []
        self.initial_prediction = 0.0

    def fit(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y):
        self.initial_prediction = np.mean(y)
        cdef np.ndarray[np.float64_t, ndim=1] y_pred = np.empty_like(y)
        y_pred.fill(self.initial_prediction)
        X_binned, bins = bin_features(X, self.n_bins)
        cdef np.ndarray[np.int32_t, ndim=2] X_binned_c = np.asarray(X_binned, dtype=np.int32)
        cdef list bins_c = bins
        cdef np.ndarray[np.float64_t, ndim=1] hessians = np.ones_like(y)
        cdef np.ndarray[np.float64_t, ndim=2] hist_hess = compute_histogram_hessians(X_binned_c, hessians, self.n_bins)

        cdef int i
        cdef np.ndarray[np.float64_t, ndim=1] gradients
        cdef np.ndarray[np.float64_t, ndim=2] hist_grad
        cdef TreeNode tree

        for i in range(self.n_estimators):
            gradients = y - y_pred
            hist_grad = compute_histogram_gradients(X_binned_c, gradients, self.n_bins)
            tree = self.build_tree(X_binned_c, gradients, hessians, bins_c)
            y_pred += self.learning_rate * self.predict_tree(tree, X)
            self.trees.append(tree)

    cdef TreeNode build_tree(
        self, 
        np.ndarray[np.int32_t, ndim=2] X_binned, 
        np.ndarray[np.float64_t, ndim=1] gradients,
        np.ndarray[np.float64_t, ndim=1] hessians, 
        list bins, 
        int current_depth=0, 
        double lambda_=1.0
    ):
        cdef TreeNode node = TreeNode()
        cdef np.ndarray[np.float64_t, ndim=2] hist_grad = compute_histogram_gradients(X_binned, gradients, self.n_bins)
        cdef np.ndarray[np.float64_t, ndim=2] hist_hess = compute_histogram_hessians(X_binned, hessians, self.n_bins)
    
        if current_depth == self.max_depth or len(gradients) < self.min_samples_split:
            node.is_terminal = 1
            node.prediction = -np.sum(gradients) / (np.sum(hessians) + lambda_)
            return node
    
        cdef int feature_index, bin_index
        feature_index, bin_index = find_best_split_histogram(hist_grad, hist_hess, self.n_bins, self.min_samples_split, lambda_)

        if feature_index == -1:
            node.is_terminal = 1
            node.prediction = -np.sum(gradients) / (np.sum(hessians) + lambda_)
            return node
    
        node.feature_index = feature_index
        node.split_value = bins[feature_index][bin_index]

        cdef np.ndarray[np.uint8_t, ndim=1] left_indices = X_binned[:, feature_index] <= bin_index
        cdef np.ndarray[np.uint8_t, ndim=1] right_indices = X_binned[:, feature_index] > bin_index
    
        node.left = self.build_tree(
            X_binned[left_indices], 
            gradients[left_indices], 
            hessians[left_indices], 
            bins, 
            current_depth + 1, 
            lambda_
        )
        node.right = self.build_tree(
            X_binned[right_indices], 
            gradients[right_indices], 
            hessians[right_indices], 
            bins, 
            current_depth + 1, 
            lambda_
        )
    
        return node

    def predict_tree(self, TreeNode node, np.ndarray[np.float64_t, ndim=2] X):
        cdef int n_samples = X.shape[0]
        if node.is_terminal:
            return np.full(n_samples, node.prediction)
        
        cdef np.ndarray[np.uint8_t, ndim=1] left_indices = X[:, node.feature_index] <= node.split_value
        cdef np.ndarray[np.uint8_t, ndim=1] right_indices = X[:, node.feature_index] > node.split_value

        cdef np.ndarray[np.float64_t, ndim=1] y_pred = np.zeros(n_samples)
        y_pred[left_indices] = self.predict_tree(node.left, X[left_indices])
        y_pred[right_indices] = self.predict_tree(node.right, X[right_indices])
        return y_pred

    def predict(self, np.ndarray[np.float64_t, ndim=2] X):
        cdef int n_samples = X.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] y_pred = np.empty(n_samples, dtype=np.float64)
        y_pred.fill(self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * self.predict_tree(tree, X)
        return y_pred

    def print_tree(self, TreeNode node, int depth=0):
        indent = "  " * depth
        if node.is_terminal:
            print(f"{indent}Leaf: Predict {node.prediction}")
        else:
            print(f"{indent}Node: Feature {node.feature_index}, Split {node.split_value}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    def display_trees(self):
        for i, tree in enumerate(self.trees):
            print(f"Tree {i}:")
            self.print_tree(tree)