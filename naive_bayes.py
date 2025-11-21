import numpy as np
import random
import csv
from typing import List, Tuple

# NOTE: If there are parameters that are not used, keep them in the function signature for consistency.

text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?",
    "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)",
]
categorical_cols = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?"
]

def extract_likert_scale(response):
    response = str(response).strip()
    if '—' in response:
        try:
            return int(response.split('—')[0].strip())
        except Exception:
            return 3
    else:
        try:
            return int(response)
        except Exception:
            return 3

# --------------------------------------------------
# Naive Bayes (multiclass Bernoulli)
# --------------------------------------------------

import numpy as np

class NaiveBayesModel:
    def __init__(self, smoothing: float = 0, method: str = 'mle'):
        """ Initialize the Naive Bayes model.
        Args:
            smoothing: Smoothing parameter (float).
            method: 'mle' for Maximum Likelihood Estimation, 'map' for Maximum A Posteriori.
        """
        self.smoothing = smoothing
        self.method = method.lower()
        if self.method not in ['mle', 'map']:
            raise ValueError("method must be 'mle' or 'map'")
        self.pi = None
        self.theta_text = None
        self._log_theta_text = None
        self._log_1m_theta_text = None
        self.theta_quant = None
        self._log_theta_quant = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.spec = None

    @staticmethod
    def make_bow_with_spec(data, spec=None):
        """Convert raw CSV data into bag-of-words and quantitative feature matrices.
        Args:
            data: List of rows, where the first row is the header.
            spec: Optional specification dictionary for feature extraction.
        Returns:
            X_text: np.ndarray of shape (N, D_text) with bag-of-words features.
            X_quant: np.ndarray of shape (N, Q) with quantitative features.
            t: np.ndarray of shape (N,) with class labels as integers.
            spec: Specification dictionary used for feature extraction.
        """
        if not data or len(data) < 2:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64), spec if spec else {}
        header = data[0]
        rows = data[1:]
        name_to_idx = {col_name: idx for idx, col_name in enumerate(header)}
        if spec is None:
            text_indices = [name_to_idx[c] for c in text_cols if c in name_to_idx]
            likert_indices = [name_to_idx[c] for c in categorical_cols if c in name_to_idx]
            def tokenize(s):
                return str(s).lower().split()
            per_col_vocab = {}
            per_col_vocab_sizes = {}
            for col_idx in text_indices:
                vocab_set = set()
                for row in rows:
                    if col_idx < len(row):
                        for w in tokenize(row[col_idx]):
                            vocab_set.add(w)
                sorted_vocab = sorted(vocab_set)
                per_col_vocab[col_idx] = {w: i for i, w in enumerate(sorted_vocab)}
                per_col_vocab_sizes[col_idx] = len(sorted_vocab)
            feature_offsets = {}
            offset = 0
            for col_idx in text_indices:
                Vj = per_col_vocab_sizes[col_idx]
                feature_offsets[col_idx] = (offset, offset + Vj)
                offset += Vj
            D_text = offset
            Q = len(likert_indices)
            label_map = {'ChatGPT': 0, 'Claude': 1, 'Gemini': 2}
            spec = {
                'text_indices': text_indices,
                'likert_indices': likert_indices,
                'per_col_vocab': per_col_vocab,
                'feature_offsets': feature_offsets,
                'label_map': label_map,
                'D_text': D_text,
                'Q': Q,
            }
        else:
            text_indices = spec['text_indices']
            likert_indices = spec['likert_indices']
            per_col_vocab = spec['per_col_vocab']
            feature_offsets = spec['feature_offsets']
            label_map = spec['label_map']
            D_text = spec['D_text']
            Q = spec['Q']
            def tokenize(s):
                return str(s).lower().split()
        N = len(rows)
        X_text = np.zeros((N, D_text), dtype=np.float32)
        X_quant = np.zeros((N, Q), dtype=np.int64)
        t = np.zeros((N,), dtype=np.int64)
        for i, row in enumerate(rows):
            for col_idx in text_indices:
                if col_idx < len(row):
                    words = set(tokenize(row[col_idx]))
                    start, end = feature_offsets[col_idx]
                    lookup = per_col_vocab[col_idx]
                    for w in words:
                        if w in lookup:
                            X_text[i, start + lookup[w]] = 1.0
            for qpos, col_idx in enumerate(likert_indices):
                if col_idx < len(row):
                    val = extract_likert_scale(row[col_idx])
                    if 1 <= val <= 5:
                        X_quant[i, qpos] = int(val)
            if 'label' in name_to_idx:
                lbl = str(row[name_to_idx['label']]).strip()
                t[i] = spec['label_map'].get(lbl, 0)
            else:
                assigned = False
                for cell in row:
                    key = str(cell).strip()
                    if key in spec['label_map']:
                        t[i] = spec['label_map'][key]
                        assigned = True
                        break
                if not assigned:
                    t[i] = 0
        return X_text, X_quant, t, spec

    def naive_bayes_mle(self, X_text: np.ndarray, X_quant: np.ndarray, t: np.ndarray, num_classes: int = 3):
        """Maximum Likelihood Estimation for Naive Bayes with text and quantitative features.
        Args:
            X_text: np.ndarray of shape (N, V_text) with binary bag-of-words features.
            X_quant: np.ndarray of shape (N, Q) with quantitative features (Likert scale 1-5).
            t: np.ndarray of shape (N,) with class labels as integers.
            num_classes: Number of classes (C).
        Returns:
            pi: np.ndarray of shape (C,) with class prior probabilities.
            theta_text: np.ndarray of shape (V_text, C) with Bernoulli parameters for text features.
            theta_quant: np.ndarray of shape (Q, 5, C) with Multinomial parameters for quantitative features.
        """
        N, V_text = X_text.shape
        _, Q = X_quant.shape
        C = num_classes
        alpha = float(self.smoothing)

        class_counts = np.bincount(t, minlength=C).astype(np.float64)
        pi = class_counts / max(1, N)

        Y = np.eye(C, dtype=np.float64)[t]

        theta_text = np.zeros((V_text, C), dtype=np.float64)
        if V_text > 0:
            counts_text = X_text.T @ Y
            denom_text = class_counts + 2.0 * alpha
            theta_text = (counts_text + alpha) / denom_text
            theta_text = np.clip(theta_text, 1e-12, 1.0 - 1e-12)

        theta_quant = np.zeros((Q, 5, C), dtype=np.float64)
        if Q > 0:
            for v in range(1, 6):
                mask_v = (X_quant == v).astype(np.float64)
                counts_v = mask_v.T @ Y
                denom_q = class_counts + 5.0 * alpha
                theta_quant[:, v-1, :] = (counts_v + alpha) / denom_q
            theta_quant = np.clip(theta_quant, 1e-12, 1.0 - 1e-12)

        return pi, theta_text, theta_quant

    def naive_bayes_map(self, X_text: np.ndarray, X_quant: np.ndarray, t: np.ndarray, num_classes: int = 3):
        N_samples = X_text.shape[0]
        V_text = X_text.shape[1]
        Q = X_quant.shape[1]
        C = num_classes

        class_counts = np.bincount(t, minlength=C).astype(np.float64)

        # Priors for Beta-Binomial (for text features) - typically Beta(a, b)
        # Using a=2.0, b=2.0 as in the original `naive_bayes_map` for text features.
        a_map_text = 2.0
        b_map_text = 2.0

        # Pi (class priors) using a symmetric Dirichlet(1.0) prior, as it's common for MAP.
        # This effectively adds 1 pseudo-count to each class.
        pi = (class_counts + 1.0) / (N_samples + C * 1.0)

        theta_text = np.zeros((V_text, C), dtype=np.float64)
        if V_text > 0:
            Y = np.eye(C, dtype=np.float64)[t]
            counts_text_1 = X_text.T @ Y  # Counts where word is present for each class
            # MAP estimate for Bernoulli parameter with Beta(a,b) prior is (counts_1 + a - 1) / (N_c + a + b - 2)
            theta_text = (counts_text_1 + (a_map_text - 1)) / (class_counts + (a_map_text + b_map_text - 2))
            theta_text = np.clip(theta_text, 1e-12, 1.0 - 1e-12)

        theta_quant = np.zeros((Q, 5, C), dtype=np.float64)
        if Q > 0:
            # Use self.smoothing as the Dirichlet prior parameter (alpha) for each category within each quantitative feature.
            # If smoothing is 0 (or not positive), use 1.0 for Laplace smoothing as a default for MAP.
            alpha_dirichlet = self.smoothing if self.smoothing > 0 else 1.0

            for q_idx in range(Q):
                for v in range(1, 6):  # Likert scale values 1 to 5
                    # Count samples where X_quant[:, q_idx] == v AND t == c
                    mask_q_v = (X_quant[:, q_idx] == v)
                    class_specific_counts = np.bincount(t[mask_q_v], minlength=C).astype(np.float64)

                    # MAP estimate for Multinomial parameter with Dirichlet(alpha) prior
                    # (counts_v_c + alpha_dirichlet) / (N_c + K * alpha_dirichlet) where K=5 categories
                    theta_quant[q_idx, v-1, :] = (class_specific_counts + alpha_dirichlet) / (class_counts + 5 * alpha_dirichlet)

            theta_quant = np.clip(theta_quant, 1e-12, 1.0 - 1e-12)

        return pi, theta_text, theta_quant

    def train(self, train_X, train_t, learning_rate=None, batch_size=None, n_epochs: int = 1):
        """Train the Naive Bayes model using the specified method (MLE or MAP).
        Args:
            train_X: np.ndarray or tuple of np.ndarrays. If tuple, should be (X_text, X_quant).
            train_t: np.ndarray of shape (N,) with class labels as integers.
            learning_rate: Not used in Naive Bayes, included for compatibility.
            batch_size: Not used in Naive Bayes, included for compatibility.
            n_epochs: Not used in Naive Bayes, included for compatibility.
        """
        num_classes = int(np.max(train_t)) + 1
        if isinstance(train_X, (list, tuple)) and len(train_X) == 2:
            X_text, X_quant = train_X
        else:
            X_text = train_X
            X_quant = np.zeros((X_text.shape[0], 0), dtype=np.int64) # Handle text-only input

        if self.method == 'mle':
            pi_est, theta_text_est, theta_quant_est = self.naive_bayes_mle(X_text, X_quant, train_t, num_classes=num_classes)
        elif self.method == 'map':
            pi_est, theta_text_est, theta_quant_est = self.naive_bayes_map(X_text, X_quant, train_t, num_classes=num_classes)

        self.pi = pi_est.astype(np.float64)
        self.theta_text = theta_text_est.astype(np.float64)
        self.theta_quant = theta_quant_est.astype(np.float64)

        self._log_theta_text = np.log(self.theta_text)
        self._log_1m_theta_text = np.log(1.0 - self.theta_text)
        if self.theta_quant is not None and self.theta_quant.size > 0: # Check if theta_quant is not empty
            self._log_theta_quant = np.log(self.theta_quant)
        else:
            self._log_theta_quant = None # Set to None if no quant features

        history = {'method': self.method, 'pi': pi_est, 'theta_text_shape': theta_text_est.shape, 'theta_quant_shape': theta_quant_est.shape}
        return history

    def predict(self, X: np.ndarray):
        """Predict class labels for the given input data.
        Args:
            X: np.ndarray or tuple of np.ndarrays. If tuple, should be (X_text, X_quant).
        Returns:
            np.ndarray of shape (N,) with predicted class labels.
        """
        if isinstance(X, (list, tuple)) and len(X) == 2:
            X_text, X_quant = X
        else:
            X_text = X
            X_quant = np.zeros((X_text.shape[0], 0), dtype=np.int64)

        if self.pi is None:
            raise ValueError("Model is not trained. Call train() first.")
        C = self.pi.shape[0]
        log_pi = np.log(self.pi + 1e-12)

        # Initialize log_post with log_pi, replicating for each sample
        log_post = np.zeros((X_text.shape[0], C), dtype=np.float64) + log_pi

        if X_text.shape[1] > 0 and self._log_theta_text is not None:
            term_pos = X_text @ self._log_theta_text
            term_neg = (1.0 - X_text) @ self._log_1m_theta_text
            log_post += term_pos + term_neg

        if X_quant.shape[1] > 0 and self._log_theta_quant is not None:
            # Ensure self._log_theta_quant is compatible (Q, 5, C)
            if self._log_theta_quant.shape[0] != X_quant.shape[1]:
                raise ValueError("Mismatch between number of quantitative features and trained model parameters.")

            for q in range(X_quant.shape[1]):
                vals = X_quant[:, q]
                logp_q = self._log_theta_quant[q] # Shape (5, C)
                # Map Likert values (1-5) to array indices (0-4)
                idx = (vals - 1).clip(0, logp_q.shape[0]-1)
                # Select probabilities for the observed values for each sample and class
                logp_pick = logp_q[idx[:, np.newaxis], np.arange(C)] # Shape (N_samples, C)
                log_post += logp_pick

        return np.argmax(log_post, axis=1)

    def predict_proba(self, X: np.ndarray):
        """Predict class probabilities for the given input data.
        Args:
            X: np.ndarray or tuple of np.ndarrays. If tuple, should be (X_text, X_quant).
        Returns:
            np.ndarray of shape (N, C) with predicted class probabilities.
        """
        if isinstance(X, (list, tuple)) and len(X) == 2:
            X_text, X_quant = X
        else:
            X_text = X
            X_quant = np.zeros((X_text.shape[0], 0), dtype=np.int64)

        if self.pi is None:
            raise ValueError("Model is not trained. Call train() first.")
        C = self.pi.shape[0]
        log_pi = np.log(self.pi + 1e-12)

        log_post = np.zeros((X_text.shape[0], C), dtype=np.float64) + log_pi

        if X_text.shape[1] > 0 and self._log_theta_text is not None:
            term_pos = X_text @ self._log_theta_text
            term_neg = (1.0 - X_text) @ self._log_1m_theta_text
            log_post += term_pos + term_neg

        if X_quant.shape[1] > 0 and self._log_theta_quant is not None:
            if self._log_theta_quant.shape[0] != X_quant.shape[1]:
                raise ValueError("Mismatch between number of quantitative features and trained model parameters.")

            for q in range(X_quant.shape[1]):
                vals = X_quant[:, q]
                logp_q = self._log_theta_quant[q]
                idx = (vals - 1).clip(0, logp_q.shape[0]-1)
                logp_pick = logp_q[idx[:, np.newaxis], np.arange(C)]
                log_post += logp_pick

        log_post -= np.max(log_post, axis=1, keepdims=True)
        probs = np.exp(log_post)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def save_model(self, file_path):
        """Save the trained model parameters to a file.
        Args:
            file_path: Path to the file where the model will be saved.
        """
        if self.pi is None:
            raise ValueError("Nothing to save: train the model first.")
        save_dict = {'pi': self.pi, 'smoothing': np.array([self.smoothing], dtype=np.float64)}
        if hasattr(self, 'theta_text') and self.theta_text is not None:
            save_dict['theta_text'] = self.theta_text
        if hasattr(self, 'theta_quant') and self.theta_quant is not None:
            save_dict['theta_quant'] = self.theta_quant
        if hasattr(self, 'spec') and self.spec is not None:
            save_dict['spec'] = np.array([self.spec], dtype=object)
        np.savez_compressed(file_path, **save_dict)

    def load_model(self, file_path):
        """Load model parameters from a file.
        Args:
            file_path: Path to the file from which the model will be loaded.
        """
        data = np.load(file_path, allow_pickle=True)
        self.pi = data['pi']
        self.smoothing = float(data['smoothing'][0]) if 'smoothing' in data else 1.0
        if 'theta_text' in data:
            self.theta_text = data['theta_text']
            self._log_theta_text = np.log(self.theta_text)
            self._log_1m_theta_text = np.log(1.0 - self.theta_text)
        else:
            self.theta_text = None
            self._log_theta_text = None
            self._log_1m_theta_text = None
        if 'theta_quant' in data:
            self.theta_quant = data['theta_quant']
            self._log_theta_quant = np.log(self.theta_quant)
        else:
            self.theta_quant = None
            self._log_theta_quant = None
        if 'spec' in data:
            try:
                self.spec = data['spec'].tolist()[0]
            except Exception:
                self.spec = None
        else:
            self.spec = None

if __name__ == '__main__':
    random.seed(67)
    # Load dataset
    with open("training_data_clean.csv", newline="", encoding="utf-8") as f:
        data = list(csv.reader(f))

    if not data or len(data) < 2:
        raise SystemExit("Dataset is empty or missing header.")

    header = data[0]
    rows = data[1:]
    random.shuffle(rows)
    split = int(0.8 * len(rows)) if len(rows) > 1 else len(rows)

    train_data = [header] + rows[:split]
    valid_data = [header] + rows[split:]

    # Build features with a persistent spec to ensure consistent dimensions
    X_train_text, X_train_quant, t_train, spec = NaiveBayesModel.make_bow_with_spec(train_data, spec=None)
    X_valid_text, X_valid_quant, t_valid, _ = NaiveBayesModel.make_bow_with_spec(valid_data, spec=spec)

    # Train and evaluate with both methods
    print("Training with MLE...")
    model_mle = NaiveBayesModel(smoothing=0.5, method='mle')
    model_mle.spec = spec
    model_mle.train((X_train_text, X_train_quant), t_train, n_epochs=1)

    if X_valid_text.shape[0] > 0:
        y_pred_mle = model_mle.predict((X_valid_text, X_valid_quant))
        acc_mle = float((y_pred_mle == t_valid).mean())
        print(f"MLE valid accuracy: {acc_mle:.4f} ({y_pred_mle.size} samples)")
    
    print("\nTraining with MAP...")
    model_map = NaiveBayesModel(smoothing=0.2, method='map')
    model_map.spec = spec
    model_map.train((X_train_text, X_train_quant), t_train, n_epochs=1)
    
    if X_valid_text.shape[0] > 0:
        y_pred_map = model_map.predict((X_valid_text, X_valid_quant))
        acc_map = float((y_pred_map == t_valid).mean())
        print(f"MAP valid accuracy: {acc_map:.4f} ({y_pred_map.size} samples)")
    
    if X_valid_text.shape[0] == 0:
        print("No validation samples; training completed.")
    