import numpy as np

class SoftmaxRegression:
    """
    Softmax Regression - Academic Version
    ---------------------------------------------------
    - Mini-batch SGD
    - Softmax + Cross-Entropy (Sparse)
    - Correct L2 Regularization
    - Xavier Initialization
    - Z-score Normalization
    - Fully consistent with mathematical formulation
    - Numerical Gradient Checking
    """

    def __init__(
        self,
        learning_rate=0.1,
        epochs=100,
        batch_size=128,
        reg=1e-4,
        normalize=True,
        random_state=None,
        verbose=True
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        self.normalize = normalize
        self.random_state = random_state
        self.verbose = verbose

        self.W = None
        self.b = None
        self.scaler_mean = None
        self.scaler_std = None
        self.history = {"loss": []}
        self.n_samples = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    # =========================
    # Initialization
    # =========================
    def _initialize_weights(self, n_features, n_classes):
        std = np.sqrt(1.0 / n_features)
        self.W = np.random.randn(n_features, n_classes) * std
        self.b = np.zeros((1, n_classes))

    # =========================
    # Core Math
    # =========================
    def _softmax(self, z):
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        oh = np.zeros((y.shape[0], n_classes))
        oh[np.arange(y.shape[0]), y] = 1.0
        return oh

    def _forward(self, X):
        return np.dot(X, self.W) + self.b

    def _loss_and_gradients(self, X, y):
        m = X.shape[0]
        scores = self._forward(X)
        probs = self._softmax(scores)

        correct_logprobs = -np.log(probs[np.arange(m), y] + 1e-12)
        data_loss = np.mean(correct_logprobs)

        reg_loss = (self.reg / (2 * self.n_samples)) * np.sum(self.W ** 2)
        loss = data_loss + reg_loss

        Y = self._one_hot(y, probs.shape[1])
        diff = probs - Y

        dW = (np.dot(X.T, diff) / m) + (self.reg / self.n_samples) * self.W
        db = np.sum(diff, axis=0, keepdims=True) / m

        return loss, dW, db

    # =========================
    # Training
    # =========================
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if self.normalize:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0)
            self.scaler_std[self.scaler_std == 0] = 1.0
            X = (X - self.scaler_mean) / self.scaler_std

        self.n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1

        self._initialize_weights(n_features, n_classes)

        num_batches = int(np.ceil(self.n_samples / self.batch_size))

        for epoch in range(self.epochs):
            idx = np.random.permutation(self.n_samples)
            X_shuf, y_shuf = X[idx], y[idx]

            epoch_loss = 0.0

            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, self.n_samples)

                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                loss, dW, db = self._loss_and_gradients(X_batch, y_batch)

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

                epoch_loss += loss * (end - start)

            epoch_loss /= self.n_samples
            self.history["loss"].append(epoch_loss)

            if self.verbose and ((epoch == 0) or ((epoch + 1) % 10 == 0)):
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss:.6f}")

        return self

    # =========================
    # Prediction
    # =========================
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.normalize:
            X = (X - self.scaler_mean) / self.scaler_std
        scores = self._forward(X)
        return self._softmax(scores)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # =========================
    # Numerical Gradient Check
    # =========================
    def gradient_check(self, X, y, eps=1e-5, tol=1e-6):
        X = X[:5]
        y = y[:5]

        if self.normalize:
            mu = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            X = (X - mu) / std

        self.n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1

        self._initialize_weights(n_features, n_classes)

        _, dW_anal, db_anal = self._loss_and_gradients(X, y)

        num_dW = np.zeros_like(self.W)
        for i in range(n_features):
            for j in range(n_classes):
                old = self.W[i, j]

                self.W[i, j] = old + eps
                loss1, _, _ = self._loss_and_gradients(X, y)

                self.W[i, j] = old - eps
                loss2, _, _ = self._loss_and_gradients(X, y)

                num_dW[i, j] = (loss1 - loss2) / (2 * eps)
                self.W[i, j] = old

        num_db = np.zeros_like(self.b)
        for j in range(n_classes):
            old = self.b[0, j]

            self.b[0, j] = old + eps
            loss1, _, _ = self._loss_and_gradients(X, y)

            self.b[0, j] = old - eps
            loss2, _, _ = self._loss_and_gradients(X, y)

            num_db[0, j] = (loss1 - loss2) / (2 * eps)
            self.b[0, j] = old

        def rel_error(a, b):
            return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b))))

        err_W = rel_error(dW_anal, num_dW)
        err_b = rel_error(db_anal, num_db)

        print(f"Gradient check | rel_err(W)={err_W:.2e}, rel_err(b)={err_b:.2e}")
        return err_W < tol and err_b < tol
