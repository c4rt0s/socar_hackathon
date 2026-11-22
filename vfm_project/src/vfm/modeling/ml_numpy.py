"""Small, deterministic NumPy-only regressors for a virtual flow meter."""

from __future__ import annotations

from typing import Any

import numpy as np


class _BaseRegressor:
    """Base mixin providing validation and column-median NaN imputation."""

    _n_features: int
    _medians: np.ndarray

    def _validate_fit_inputs(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array.")

        n_samples, n_features = X_arr.shape
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1 or y_arr.shape[0] != n_samples:
            raise ValueError("y must be a 1D array with length matching X.shape[0].")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y must contain only finite values.")

        if sample_weight is None:
            w = np.ones(n_samples, dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            if w.ndim != 1 or w.shape[0] != n_samples:
                raise ValueError("sample_weight must be 1D with length matching X.shape[0].")
            if not np.all(np.isfinite(w)) or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
                raise ValueError("sample_weight must be finite, non-negative, and have positive sum.")

        medians = np.zeros(n_features, dtype=np.float64)
        for j in range(n_features):
            col = X_arr[:, j]
            valid = col[~np.isnan(col)]
            if valid.size > 0:
                medians[j] = float(np.median(valid))

        self._n_features = n_features
        self._medians = medians
        return self._impute(X_arr), y_arr, w

    def _impute(self, X_arr: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(X_arr)
        if not np.any(nan_mask):
            return X_arr
        out = X_arr.copy()
        _, cols = np.where(nan_mask)
        out[nan_mask] = self._medians[cols]
        return out

    def _validate_predict_X(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_n_features"):
            raise ValueError("Model must be fitted before calling predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] != self._n_features:
            raise ValueError(
                f"X must be a non-empty 2D array with {self._n_features} features."
            )
        return self._impute(X_arr)


class RidgeRegressor(_BaseRegressor):
    """Closed-form weighted ridge regression on optionally standardised features."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        standardize: bool = True,
    ) -> None:
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.standardize = bool(standardize)
        self.coef_: np.ndarray = np.empty(0, dtype=np.float64)
        self.intercept_: float = 0.0

    def get_params(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "standardize": self.standardize,
        }

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RidgeRegressor:
        """Fit weighted ridge regression in closed form."""
        X_clean, y_arr, w = self._validate_fit_inputs(X, y, sample_weight)
        w_sum = float(np.sum(w))
        n_features = X_clean.shape[1]

        if self.fit_intercept:
            x_mean = np.sum(w[:, None] * X_clean, axis=0) / w_sum
            y_mean = float(np.sum(w * y_arr) / w_sum)
            X_work = X_clean - x_mean
            y_work = y_arr - y_mean
        else:
            x_mean = np.zeros(n_features, dtype=np.float64)
            y_mean = 0.0
            X_work = X_clean
            y_work = y_arr

        if self.standardize:
            if self.fit_intercept:
                var = np.sum(w[:, None] * (X_work * X_work), axis=0) / w_sum
            else:
                xc = X_clean - (np.sum(w[:, None] * X_clean, axis=0) / w_sum)
                var = np.sum(w[:, None] * (xc * xc), axis=0) / w_sum
            scale = np.sqrt(np.maximum(var, 0.0))
            scale = np.where(scale < 1e-12, 1.0, scale)
            X_scaled = X_work / scale
        else:
            scale = np.ones(n_features, dtype=np.float64)
            X_scaled = X_work

        Xw = X_scaled * w[:, None]
        gram = Xw.T @ X_scaled
        gram[np.diag_indices(n_features)] += self.alpha
        rhs = Xw.T @ y_work

        try:
            beta_scaled = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta_scaled, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)

        self.coef_ = beta_scaled / scale
        self.intercept_ = float(y_mean - np.dot(x_mean, self.coef_)) if self.fit_intercept else 0.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for X."""
        X_clean = self._validate_predict_X(X)
        return X_clean @ self.coef_ + self.intercept_


def _resolve_max_features(max_features: int | float | str | None, n_features: int) -> int:
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if isinstance(max_features, bool):
        raise ValueError("max_features cannot be a boolean.")
    if isinstance(max_features, int):
        if max_features <= 0:
            raise ValueError("max_features int must be positive.")
        return min(max_features, n_features)
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError("max_features float must be in (0, 1].")
        return max(1, min(n_features, int(np.ceil(max_features * n_features))))
    raise ValueError(f"Unsupported max_features: {max_features!r}")


class DecisionTreeRegressor(_BaseRegressor):
    """Exact greedy CART regressor on weighted squared error stored in flat arrays."""

    def __init__(
        self,
        max_depth: int | None = 3,
        min_samples_leaf: int = 3,
        max_features: int | float | str | None = None,
        random_state: int | None = 0,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.feature_: np.ndarray = np.empty(0, dtype=np.int64)
        self.threshold_: np.ndarray = np.empty(0, dtype=np.float64)
        self.left_: np.ndarray = np.empty(0, dtype=np.int64)
        self.right_: np.ndarray = np.empty(0, dtype=np.int64)
        self.value_: np.ndarray = np.empty(0, dtype=np.float64)
        self._depth: int = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> DecisionTreeRegressor:
        """Fit CART regression tree using vectorised cumulative-sum split search."""
        X_clean, y_arr, w = self._validate_fit_inputs(X, y, sample_weight)
        n_samples, n_features = X_clean.shape
        n_cand = _resolve_max_features(self.max_features, n_features)
        rng = np.random.default_rng(self.random_state)

        features: list[int] = []
        thresholds: list[float] = []
        lefts: list[int] = []
        rights: list[int] = []
        values: list[float] = []
        max_depth_seen = 0

        def build(idx: np.ndarray, depth: int) -> int:
            nonlocal max_depth_seen
            if depth > max_depth_seen:
                max_depth_seen = depth

            node_id = len(values)
            w_node = w[idx]
            y_node = y_arr[idx]
            w_sum = float(np.sum(w_node))
            y_mean = float(np.sum(w_node * y_node) / w_sum)

            values.append(y_mean)
            features.append(-1)
            thresholds.append(0.0)
            lefts.append(-1)
            rights.append(-1)

            n_node = idx.size
            if (self.max_depth is not None and depth >= self.max_depth) or n_node < 2 * self.min_samples_leaf:
                return node_id

            y_centered = y_node - y_mean
            node_sse = float(np.sum(w_node * (y_centered * y_centered)))
            if node_sse <= 0.0:
                return node_id

            cand = (
                rng.choice(n_features, size=n_cand, replace=False)
                if n_cand < n_features
                else np.arange(n_features)
            )

            lo = self.min_samples_leaf - 1
            hi = n_node - self.min_samples_leaf
            best_gain = 1e-15 * node_sse
            best_feat = -1
            best_thresh = 0.0

            wy_centered = w_node * y_centered
            for f in cand:
                x_f = X_clean[idx, f]
                order = np.argsort(x_f, kind="mergesort")
                x_sorted = x_f[order]
                if x_sorted[0] == x_sorted[-1]:
                    continue

                valid_mask = x_sorted[lo:hi] != x_sorted[lo + 1 : hi + 1]
                if not np.any(valid_mask):
                    continue

                w_cum = np.cumsum(w_node[order])
                wy_cum = np.cumsum(wy_centered[order])

                valid_pos = np.flatnonzero(valid_mask)
                w_left = w_cum[lo:hi][valid_pos]
                w_right = w_sum - w_left
                pos_w = (w_left > 0.0) & (w_right > 0.0)
                if not np.any(pos_w):
                    continue

                valid_pos = valid_pos[pos_w]
                w_left = w_left[pos_w]
                w_right = w_right[pos_w]
                s_left = wy_cum[lo:hi][valid_pos]

                gains = (s_left * s_left) * (w_sum / (w_left * w_right))
                k_best = int(np.argmax(gains))
                gain = float(gains[k_best])
                if gain > best_gain:
                    pos = lo + int(valid_pos[k_best])
                    a = float(x_sorted[pos])
                    b = float(x_sorted[pos + 1])
                    thresh = a + 0.5 * (b - a)
                    if thresh >= b:
                        thresh = a
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = thresh

            if best_feat < 0:
                return node_id

            left_mask = X_clean[idx, best_feat] <= best_thresh
            left_idx = idx[left_mask]
            right_idx = idx[~left_mask]
            if left_idx.size == 0 or right_idx.size == 0:
                return node_id

            features[node_id] = best_feat
            thresholds[node_id] = best_thresh
            lefts[node_id] = build(left_idx, depth + 1)
            rights[node_id] = build(right_idx, depth + 1)
            return node_id

        build(np.arange(n_samples, dtype=np.int64), 0)
        self.feature_ = np.asarray(features, dtype=np.int64)
        self.threshold_ = np.asarray(thresholds, dtype=np.float64)
        self.left_ = np.asarray(lefts, dtype=np.int64)
        self.right_ = np.asarray(rights, dtype=np.int64)
        self.value_ = np.asarray(values, dtype=np.float64)
        self._depth = max_depth_seen
        return self

    def _predict_clean(self, X_clean: np.ndarray) -> np.ndarray:
        node_idx = np.zeros(X_clean.shape[0], dtype=np.int64)
        for _ in range(self._depth + 1):
            feat = self.feature_[node_idx]
            internal = feat >= 0
            if not np.any(internal):
                break
            idx_int = np.flatnonzero(internal)
            cur_nodes = node_idx[idx_int]
            go_left = X_clean[idx_int, feat[idx_int]] <= self.threshold_[cur_nodes]
            node_idx[idx_int] = np.where(go_left, self.left_[cur_nodes], self.right_[cur_nodes])
        return self.value_[node_idx]

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using depth-wise vectorised tree traversal."""
        X_clean = self._validate_predict_X(X)
        return self._predict_clean(X_clean)


class GradientBoostingRegressor(_BaseRegressor):
    """Gradient boosted regression trees with squared error loss."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        min_samples_leaf: int = 3,
        subsample: float = 1.0,
        random_state: int | None = 0,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1].")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = random_state
        self.init_: float = 0.0
        self.estimators_: list[DecisionTreeRegressor] = []

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "subsample": self.subsample,
            "random_state": self.random_state,
        }

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> GradientBoostingRegressor:
        """Fit additive stage-wise regression trees on pseudo-residuals."""
        X_clean, y_arr, w = self._validate_fit_inputs(X, y, sample_weight)
        n_samples = X_clean.shape[0]
        self.init_ = float(np.sum(w * y_arr) / np.sum(w))
        pred = np.full(n_samples, self.init_, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        n_sub = max(1, int(np.floor(self.subsample * n_samples)))
        use_sub = self.subsample < 1.0 and n_sub < n_samples
        self.estimators_ = []

        for _ in range(self.n_estimators):
            residuals = y_arr - pred
            seed = int(rng.integers(0, 2**31 - 1))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                random_state=seed,
            )
            if use_sub:
                sub_idx = rng.choice(n_samples, size=n_sub, replace=False)
                tree.fit(X_clean[sub_idx], residuals[sub_idx], sample_weight=w[sub_idx])
            else:
                tree.fit(X_clean, residuals, sample_weight=w)
            pred += self.learning_rate * tree._predict_clean(X_clean)
            self.estimators_.append(tree)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values by summing boosted tree stages."""
        X_clean = self._validate_predict_X(X)
        out = np.full(X_clean.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            out += self.learning_rate * tree._predict_clean(X_clean)
        return out


class RandomForestRegressor(_BaseRegressor):
    """Bootstrap-aggregated ensemble of random regression trees."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_leaf: int = 2,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = 0,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressor] = []

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RandomForestRegressor:
        """Fit random forest of regression trees."""
        X_clean, y_arr, w = self._validate_fit_inputs(X, y, sample_weight)
        n_samples = X_clean.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31 - 1))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            if self.bootstrap:
                boot_idx = rng.integers(0, n_samples, size=n_samples)
                tree.fit(X_clean[boot_idx], y_arr[boot_idx], sample_weight=w[boot_idx])
            else:
                tree.fit(X_clean, y_arr, sample_weight=w)
            self.estimators_.append(tree)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values by averaging predictions across all trees."""
        X_clean = self._validate_predict_X(X)
        out = np.zeros(X_clean.shape[0], dtype=np.float64)
        for tree in self.estimators_:
            out += tree._predict_clean(X_clean)
        out /= float(len(self.estimators_))
        return out


class KNNRegressor(_BaseRegressor):
    """Standardised Euclidean k-nearest neighbours regressor with chunked prediction."""

    _CHUNK_SIZE = 20_000

    def __init__(self, k: int = 5, weights: str = "distance") -> None:
        if k < 1:
            raise ValueError("k must be >= 1.")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        self.k = int(k)
        self.weights = weights
        self._mean: np.ndarray = np.empty(0, dtype=np.float64)
        self._std: np.ndarray = np.empty(0, dtype=np.float64)
        self._X_train: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._y_train: np.ndarray = np.empty(0, dtype=np.float64)
        self._w_train: np.ndarray = np.empty(0, dtype=np.float64)
        self._has_sample_weight: bool = False

    def get_params(self) -> dict[str, Any]:
        return {"k": self.k, "weights": self.weights}

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KNNRegressor:
        """Store standardised training features and targets."""
        X_clean, y_arr, w = self._validate_fit_inputs(X, y, sample_weight)
        mean = np.mean(X_clean, axis=0)
        std = np.std(X_clean, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        self._mean = mean
        self._std = std
        self._X_train = (X_clean - mean) / std
        self._y_train = y_arr.copy()
        self._w_train = w.copy()
        self._has_sample_weight = sample_weight is not None
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict in chunks of 20,000 rows to bound peak memory."""
        X_clean = self._validate_predict_X(X)
        n_pred = X_clean.shape[0]
        n_train = self._X_train.shape[0]
        k_eff = min(self.k, n_train)

        t_sq = np.sum(self._X_train * self._X_train, axis=1)[None, :]
        x_train_t = self._X_train.T
        out = np.empty(n_pred, dtype=np.float64)

        for start in range(0, n_pred, self._CHUNK_SIZE):
            end = min(start + self._CHUNK_SIZE, n_pred)
            x_chunk = (X_clean[start:end] - self._mean) / self._std
            x_sq = np.sum(x_chunk * x_chunk, axis=1, keepdims=True)
            d2 = np.maximum(x_sq + t_sq - 2.0 * (x_chunk @ x_train_t), 0.0)
            dist = np.sqrt(d2)

            if k_eff == n_train:
                nn_idx = np.broadcast_to(np.arange(n_train, dtype=np.int64), (end - start, n_train))
                nn_dist = dist
            else:
                nn_idx = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
                nn_dist = np.take_along_axis(dist, nn_idx, axis=1)

            nn_y = self._y_train[nn_idx]
            if self.weights == "uniform":
                if self._has_sample_weight:
                    w = self._w_train[nn_idx]
                    w_sum = np.sum(w, axis=1)
                    out[start:end] = np.where(
                        w_sum > 0.0,
                        np.sum(w * nn_y, axis=1) / np.where(w_sum > 0.0, w_sum, 1.0),
                        np.mean(nn_y, axis=1),
                    )
                else:
                    out[start:end] = np.mean(nn_y, axis=1)
            else:
                w = 1.0 / (nn_dist + 1e-9)
                if self._has_sample_weight:
                    w = w * self._w_train[nn_idx]
                w_sum = np.sum(w, axis=1)
                out[start:end] = np.where(
                    w_sum > 0.0,
                    np.sum(w * nn_y, axis=1) / np.where(w_sum > 0.0, w_sum, 1.0),
                    np.mean(nn_y, axis=1),
                )
        return out


_MODEL_REGISTRY: dict[str, type[_BaseRegressor]] = {
    "ridge": RidgeRegressor,
    "tree": DecisionTreeRegressor,
    "gbm": GradientBoostingRegressor,
    "rf": RandomForestRegressor,
    "knn": KNNRegressor,
}


def make_model(name: str, **kwargs: Any) -> _BaseRegressor:
    """Instantiate a regressor by short name ('ridge', 'tree', 'gbm', 'rf', 'knn')."""
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Valid model names are: {valid}.")
    return cls(**kwargs)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. Ridge exact linear recovery
    X_lin = rng.normal(size=(80, 4))
    y_lin = 3.0 * X_lin[:, 0] - 2.0 * X_lin[:, 1] + 5.0
    ridge = make_model("ridge", alpha=1e-9).fit(X_lin, y_lin)
    assert np.allclose(ridge.coef_, [3.0, -2.0, 0.0, 0.0], atol=1e-6)
    assert abs(ridge.intercept_ - 5.0) < 1e-6
    assert np.max(np.abs(ridge.predict(X_lin) - y_lin)) < 1e-6

    # 2. GBM and RF reduce MSE well below target variance on noisy sin(x0)
    X_sin = rng.uniform(-np.pi, np.pi, size=(150, 4))
    y_sin = np.sin(X_sin[:, 0]) + 0.1 * rng.normal(size=150)
    var_y = float(np.var(y_sin))
    gbm = make_model("gbm", random_state=0).fit(X_sin, y_sin)
    rf = make_model("rf", random_state=0).fit(X_sin, y_sin)
    assert float(np.mean((gbm.predict(X_sin) - y_sin) ** 2)) < 0.2 * var_y
    assert float(np.mean((rf.predict(X_sin) - y_sin) ** 2)) < 0.2 * var_y

    # 3. DecisionTree depth-1 split on a step function
    X_step = np.column_stack([np.linspace(-2.0, 2.0, 20), np.zeros(20)])
    y_step = np.where(X_step[:, 0] <= 0.0, -4.0, 6.0)
    tree = make_model("tree", max_depth=1, min_samples_leaf=3).fit(X_step, y_step)
    assert tree.feature_[0] == 0 and abs(tree.threshold_[0]) < 0.25
    assert np.allclose(tree.predict(X_step), y_step)

    # 4. KNN k=1 exact training target recovery
    knn = make_model("knn", k=1, weights="distance").fit(X_sin, y_sin)
    assert np.allclose(knn.predict(X_sin), y_sin, atol=1e-12)

    # 5 & 6. NaN handling at predict time and 400,000-row vectorised prediction for all models
    X_big = rng.normal(size=(400_000, 4))
    X_big[:100, 0] = np.nan
    X_big[100:200, 2] = np.nan
    for name in ("ridge", "tree", "gbm", "rf", "knn"):
        model = make_model(name).fit(X_sin, y_sin)
        preds = model.predict(X_big)
        assert preds.shape == (400_000,) and np.all(np.isfinite(preds))
        assert isinstance(model.get_params(), dict)

    print("ml_numpy self-checks passed")
