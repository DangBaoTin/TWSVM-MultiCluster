import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

class IFWLSTSVM:
    # def __init__(self, c1=1.0, c2=1.0, epsilon=1e-5, k=5, sigma=1.0):
    #     self.c1 = c1
    #     self.c2 = c2
    #     self.epsilon = epsilon
    #     self.k = k
    #     self.sigma = sigma
    #     self.w1, self.b1 = None, None
    #     self.w2, self.b2 = None, None
    #     self.X_train = None

    def __init__(self, kernel, gamma = 1, c1 = None, c2 = None, c3 = None, c4 = None):
        self.kernel = kernel
        
        self.gamma = float(gamma)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        if self.c1 is not None: self.c1 = float(self.c1)
        if self.c2 is not None: self.c2 = float(self.c2)
        if self.c3 is not None: self.c3 = float(self.c3)
        if self.c4 is not None: self.c4 = float(self.c4)
        # self.kf = {'linear':self.linear, 'polynomial':self.polynomial, 'rbf':self.rbf}
        self.k = None
        self.l = None

        # Savings results
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None

    # Helper function for the kernel, equivalent to kernelfun in MATLAB
    def rbf_kernel(self, X1, X2, sigma):
        """
        Computes the RBF kernel between two sets of data.
        """
        sq_dists = pdist(np.vstack([X1, X2]), 'sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        return np.exp(-mat_sq_dists[:X1.shape[0], X1.shape[0]:] / (2 * sigma**2))

    def _calculate_score_values(self, A):
        no_input, _ = A.shape

        # Split the data into two classes
        A1 = A[A[:, -1] == 1, :-1]
        B1 = A[A[:, -1] != 1, :-1]

        K1 = self.kernel_function(A1, A1, "rbf")
        K2 = self.kernel_function(B1, B1, "rbf")
        A_temp = A[:, :-1]
        K3 = self.kernel_function(A_temp, A_temp, "rbf")

        print(K1.shape)
        print(K2.shape)
        print(K3.shape)

        # Radius calculations
        radiusxp = np.sqrt(1 - 2 * np.mean(K1, axis=1) + np.mean(K1))
        radiusmaxxp = np.max(radiusxp)
        radiusxn = np.sqrt(1 - 2 * np.mean(K2, axis=1) + np.mean(K2))
        radiusmaxxn = np.max(radiusxn)

        # Determine alpha_d
        alpha_d = max(radiusmaxxn, radiusmaxxp)

        # Membership values
        mem1 = 1 - (radiusxp / (radiusmaxxp + 1e-4))
        mem2 = 1 - (radiusxn / (radiusmaxxn + 1e-4))

        # Distance-based outlier penalty
        DD = np.sqrt(2 * (1 - K3))
        ro = []
        for i in range(no_input):
            temp = DD[i, :]
            B1 = A[temp < alpha_d]
            x3 = B1.shape[0]
            count = np.sum(A[i, -1] * np.ones((x3,)) != B1[:, -1])
            ro.append(count / x3 if x3 > 0 else 0)
        ro = np.array(ro)

        # Combine class labels with penalties
        A2 = np.column_stack((A[:, -1], ro))
        ro2 = A2[A2[:, 0] == -1, 1]
        ro1 = A2[A2[:, 0] != -1, 1]

        # Adjust membership values
        print(mem1)
        print(1-mem1)
        print(ro1)
        v1 = (1 - mem1) * ro1
        v2 = (1 - mem2) * ro2

        # Final Membership Scores (S1 and S2)
        S1 = []
        for i in range(len(v1)):
            if v1[i] == 0:
                S1.append(mem1[i])
            elif mem1[i] <= v1[i]:
                S1.append(0)
            else:
                S1.append((1 - v1[i]) / (2 - mem1[i] - v1[i]))

        S2 = []
        for i in range(len(v2)):
            if v2[i] == 0:
                S2.append(mem2[i])
            elif mem2[i] <= v2[i]:
                S2.append(0)
            else:
                S2.append((1 - v2[i]) / (2 - mem2[i] - v2[i]))

        return np.array(S1), np.array(S2)

    def _calculate_interclass_weights(self, X, y):
        # Corresponds to linear_W_interclass_weights.m
        X1 = X[y == 1]
        X_neg = X[y == -1]

        # For class +1
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X1)
        distances, indices = nn.kneighbors(X1)
        # We take [:, 1:] because the first neighbor is the point itself
        k_distances = distances[:, 1:]
        weights = np.exp(-k_distances**2 / self.sigma**2)
        ro1 = np.sum(weights, axis=1)

        # For class -1
        nn_neg = NearestNeighbors(n_neighbors=self.k + 1).fit(X_neg)
        distances_neg, _ = nn_neg.kneighbors(X_neg)
        k_distances_neg = distances_neg[:, 1:]
        weights_neg = np.exp(-k_distances_neg**2 / self.sigma**2)
        ro2 = np.sum(weights_neg, axis=1)

        return ro1, ro2

    def fit(self, X, y):
        self.X_train = X
        A = X[y == 1]
        B = X[y == -1]
        e1 = np.ones((A.shape[0], 1))
        e2 = np.ones((B.shape[0], 1))

        # --- Weight and Score Calculation ---
        S1, S2 = self._calculate_score_values(X, y)
        ro1, ro2 = self._calculate_interclass_weights(X, y)

        # --- Kernel-based Solver ---
        # Corresponds to the 'else' block in IFW_LSTSVM.m
        KA = self.rbf_kernel(A, self.X_train, self.sigma)
        KB = self.rbf_kernel(B, self.X_train, self.sigma)

        KA_b = np.hstack([KA, e1])
        KB_b = np.hstack([KB, e2])

        # --- Solve for first hyperplane (w1, b1) ---
        # Equation (24)
        P = np.diag(ro1) @ KA_b
        Q = np.diag(S2) @ KB_b
        H1 = P.T @ P
        Q1 = Q.T @ Q
        
        # We solve the linear system: (H1 + c1*Q1 + epsilon*I) * u1 = -Q.T * S2 * e2
        # Which is equivalent to the matrix inversion in the code
        term1 = H1 + self.c1 * Q1 + self.epsilon * np.identity(Q1.shape[0])
        term2 = -Q.T @ (np.diag(S2) @ e2)
        kerH1 = np.linalg.solve(term1, term2)
        
        self.w1 = kerH1[:-1]
        self.b1 = kerH1[-1]

        # --- Solve for second hyperplane (w2, b2) ---
        # Equation (25)
        P = np.diag(S1) @ KA_b
        Q = np.diag(ro2) @ KB_b
        H1 = P.T @ P
        Q1 = Q.T @ Q

        # We solve the linear system: (Q1 + c2*H1 + epsilon*I) * u2 = P.T * S1 * e1
        term1_h2 = Q1 + self.c2 * H1 + self.epsilon * np.identity(H1.shape[0])
        term2_h2 = P.T @ (np.diag(S1) @ e1)
        kerH2 = np.linalg.solve(term1_h2, term2_h2)

        self.w2 = kerH2[:-1]
        self.b2 = kerH2[-1]

    def predict(self, X_test):
        # Corresponds to the prediction logic in IFW_LSTSVM.m
        K_test = self.rbf_kernel(X_test, self.X_train, self.sigma)
        
        # Calculate normalization terms (denominator)
        K_train = self.rbf_kernel(self.X_train, self.X_train, self.sigma)
        w11 = np.sqrt(self.w1.T @ K_train @ self.w1)
        w22 = np.sqrt(self.w2.T @ K_train @ self.w2)

        # Calculate distances to hyperplanes
        y1 = (K_test @ self.w1 + self.b1) / w11
        y2 = (K_test @ self.w2 + self.b2) / w22

        # Classify based on the smaller distance
        return np.sign(np.abs(y2) - np.abs(y1))