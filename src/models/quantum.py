"""
quantum.py -- Quantum Kernel Recommendation Models using PennyLane.

Implements QuantumKernelModel:
    - Encodes user and item latent feature representations into quantum Hilbert space.
    - Evaluates quantum state overlap / fidelity kernel: K(u, i) = |<psi(u)|W(theta)|psi(i)>|^2
    - Supports both:
        * trainable=False (untrained baseline kernel with fixed embedding)
        * trainable=True  (variational quantum kernel aligned via gradient optimization)
    - Defaults match MovieLens 100K: user_col='user_id', item_col='item_id', rating_col='rating'
    - Fully compatible with the project benchmark harness and metrics.

Usage:
    from src.models.quantum import QuantumKernelModel
    model = QuantumKernelModel(n_qubits=4, trainable=False)
    model.fit(train_df)
    preds = model.predict(test_df)
"""

import time
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pennylane as qml

# Lightweight Prediction container compatible with Surprise evaluation interface
Prediction = namedtuple("Prediction", ["uid", "iid", "r_ui", "est", "details"])


class QuantumKernelModel:
    """
    Hybrid Quantum-Classical Collaborative Filtering Model using Quantum Kernels.

    Parameters:
        n_qubits: int, number of qubits (default 4)
        trainable: bool, whether the quantum kernel circuit has trainable ansatz parameters
        user_col: str, column name for user IDs (default 'user_id')
        item_col: str, column name for item IDs (default 'item_id')
        rating_col: str, column name for ratings (default 'rating')
        n_epochs: int, training epochs for variational parameters if trainable=True
        lr: float, learning rate for quantum parameter optimization
        batch_size: int, mini-batch size for parameter training
        seed: int, random seed for reproducibility
    """

    def __init__(
        self,
        n_qubits: int = 4,
        trainable: bool = False,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        n_epochs: int = 15,
        lr: float = 0.05,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.trainable = trainable
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

        self.fit_time = 0.0
        self.predict_time = 0.0

        # Learned mappings & weights
        self.user_features = {}
        self.item_features = {}
        self.default_user_feat = None
        self.default_item_feat = None
        self.global_mean = 3.5
        self.weights = None
        self.scale = 1.0
        self.bias = 0.0

        # Build PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._build_circuit()

    def get_name(self) -> str:
        status = "Trained" if self.trainable else "Untrained"
        return f"QuantumKernel-{self.n_qubits}Q ({status})"

    def _build_circuit(self):
        """Build the quantum kernel fidelity circuit."""
        n_q = self.n_qubits

        def _entangle(wires):
            for i in range(len(wires)):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])

        def _feature_map(features):
            for i in range(n_q):
                qml.RY(features[i], wires=i)
            if n_q > 1:
                _entangle(range(n_q))

        def _feature_map_adjoint(features):
            if n_q > 1:
                for i in reversed(range(n_q)):
                    qml.CNOT(wires=[i, (i + 1) % n_q])
            for i in reversed(range(n_q)):
                qml.RY(-features[i], wires=i)

        if not self.trainable:
            @qml.qnode(self.dev, interface="autograd")
            def _kernel_qnode(x_user, x_item):
                _feature_map(x_user)
                _feature_map_adjoint(x_item)
                return qml.probs(wires=range(n_q))

            self.kernel_circuit = lambda u, v, w=None: _kernel_qnode(u, v)[0]

        else:
            @qml.qnode(self.dev, interface="autograd")
            def _trainable_qnode(x_user, x_item, weights):
                _feature_map(x_user)
                # Variational ansatz layer
                for i in range(n_q):
                    qml.RY(weights[0, i], wires=i)
                    qml.RZ(weights[1, i], wires=i)
                if n_q > 1:
                    _entangle(range(n_q))
                _feature_map_adjoint(x_item)
                return qml.probs(wires=range(n_q))

            self.kernel_circuit = lambda u, v, w: _trainable_qnode(u, v, w)[0]

    def _extract_latent_features(self, train_df: pd.DataFrame):
        """Decompose user-item matrix into n_qubits normalized angle features."""
        users = sorted(train_df[self.user_col].unique())
        items = sorted(train_df[self.item_col].unique())
        user_to_idx = {u: i for i, u in enumerate(users)}
        item_to_idx = {it: j for j, it in enumerate(items)}

        # Build sparse matrix
        rows = [user_to_idx[u] for u in train_df[self.user_col]]
        cols = [item_to_idx[it] for it in train_df[self.item_col]]
        data = train_df[self.rating_col].to_numpy(dtype=np.float32)

        mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        # TruncatedSVD for latent representations
        svd = TruncatedSVD(n_components=self.n_qubits, random_state=self.seed)
        u_emb = svd.fit_transform(mat)
        v_emb = svd.components_.T

        # Normalize to [0, pi] for angle embedding
        def _scale_to_pi(arr):
            mins = arr.min(axis=0, keepdims=True)
            maxs = arr.max(axis=0, keepdims=True)
            ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
            return np.pi * (arr - mins) / ranges

        u_scaled = _scale_to_pi(u_emb)
        v_scaled = _scale_to_pi(v_emb)

        self.user_features = {u: u_scaled[i] for u, i in user_to_idx.items()}
        self.item_features = {it: v_scaled[j] for it, j in item_to_idx.items()}
        self.default_user_feat = np.mean(u_scaled, axis=0)
        self.default_item_feat = np.mean(v_scaled, axis=0)

    def fit(self, train_df: pd.DataFrame):
        """Fit the quantum model on the training dataframe."""
        start_time = time.perf_counter()
        self.global_mean = float(train_df[self.rating_col].mean())

        # Step 1: Classical factor extraction
        self._extract_latent_features(train_df)

        # Step 2: Initialize variational weights if trainable
        rng = np.random.RandomState(self.seed)
        if self.trainable:
            # 2 layers x n_qubits
            self.weights = qml.numpy.array(
                rng.uniform(0, 2 * np.pi, size=(2, self.n_qubits)),
                requires_grad=True,
            )

            # Sample mini-batch to train variational parameters
            sample_df = train_df.sample(
                n=min(len(train_df), 300), random_state=self.seed
            ).reset_index(drop=True)

            u_feats = np.array([self.user_features.get(u, self.default_user_feat) for u in sample_df[self.user_col]])
            v_feats = np.array([self.item_features.get(it, self.default_item_feat) for it in sample_df[self.item_col]])
            y_targets = sample_df[self.rating_col].to_numpy(dtype=np.float32)

            opt = qml.AdamOptimizer(stepsize=self.lr)

            # Train variational circuit
            print(f"  [{self.get_name()}] Training variational parameters ({self.n_epochs} epochs)...")
            batch_sz = min(self.batch_size, len(sample_df))
            for epoch in range(self.n_epochs):
                idx = rng.choice(len(sample_df), size=batch_sz, replace=False)
                bx_u = u_feats[idx]
                bx_v = v_feats[idx]
                by = y_targets[idx]

                def cost_fn(w):
                    preds = []
                    for i in range(len(bx_u)):
                        k_val = self.kernel_circuit(bx_u[i], bx_v[i], w)
                        preds.append(1.0 + 4.0 * k_val)
                    preds = qml.numpy.array(preds)
                    return qml.numpy.mean((preds - by) ** 2)

                self.weights, loss = opt.step_and_cost(cost_fn, self.weights)

        # Step 3: Calibrate kernel fidelity to rating range [1, 5]
        calib_df = train_df.sample(n=min(len(train_df), 200), random_state=self.seed)
        k_vals = []
        y_vals = []
        for _, row in calib_df.iterrows():
            u_f = self.user_features.get(row[self.user_col], self.default_user_feat)
            v_f = self.item_features.get(row[self.item_col], self.default_item_feat)
            k_val = float(self.kernel_circuit(u_f, v_f, self.weights))
            k_vals.append(k_val)
            y_vals.append(row[self.rating_col])

        k_arr = np.array(k_vals)
        y_arr = np.array(y_vals)
        var_k = np.var(k_arr)
        if var_k > 1e-6:
            self.scale = float(np.cov(k_arr, y_arr)[0, 1] / var_k)
            self.bias = float(np.mean(y_arr) - self.scale * np.mean(k_arr))
        else:
            self.scale = 2.0
            self.bias = self.global_mean - self.scale * np.mean(k_arr)

        self.fit_time = time.perf_counter() - start_time
        print(f"  [{self.get_name()}] Fitted in {self.fit_time:.2f}s")
        return self

    def predict(self, test_df: pd.DataFrame) -> list:
        """
        Generate predictions for test_df.
        Costs exactly ~1 quantum circuit run per test row.
        Returns list of Prediction objects matching the benchmark evaluation interface.
        """
        start_time = time.perf_counter()
        predictions = []

        u_col = self.user_col
        it_col = self.item_col
        r_col = self.rating_col

        total_rows = len(test_df)
        print(f"  [{self.get_name()}] Computing quantum kernel predictions for {total_rows:,} rows...")

        for idx, row in test_df.iterrows():
            uid = row[u_col]
            iid = row[it_col]
            true_r = float(row[r_col]) if r_col in row else float("nan")

            u_f = self.user_features.get(uid, self.default_user_feat)
            v_f = self.item_features.get(iid, self.default_item_feat)

            # Single circuit evaluation per row
            k_val = float(self.kernel_circuit(u_f, v_f, self.weights))

            # Calibrate to rating scale and clip to [1.0, 5.0]
            est_r = self.scale * k_val + self.bias
            est_r = float(np.clip(est_r, 1.0, 5.0))

            predictions.append(
                Prediction(uid=uid, iid=iid, r_ui=true_r, est=est_r, details={"kernel_val": k_val})
            )

        self.predict_time = time.perf_counter() - start_time
        print(f"  [{self.get_name()}] Evaluated {total_rows:,} circuits in {self.predict_time:.2f}s")
        return predictions
