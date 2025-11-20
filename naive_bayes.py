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

# --------------------------------------------------
# Naive Bayes (multiclass Bernoulli)
# --------------------------------------------------

class NaiveBayesModel:
    """
    A template class for machine learning models.
    """
    # TODO: Put any needed helper functions here
    # e.g., debugging utilities, model-specific methods, forward pass, back propagation etc.

    def __init__(self, smoothing: float = 0, method: str = 'mle'): 
        """
        Bernoulli Naive Bayes (multiclass) with additive smoothing (Laplace).
        
        Args:
            smoothing: Additive smoothing parameter (α). Used for MLE smoothing.
            method: Estimation method - 'mle' or 'map'. 
                    'mle' uses maximum likelihood with Laplace smoothing.
                    'map' uses maximum a posteriori with Beta(2,2) priors.
        """
        self.smoothing = smoothing  # Laplace smoothing alpha
        self.method = method.lower()
        if self.method not in ['mle', 'map']:
            raise ValueError("method must be 'mle' or 'map'")
        self.pi = None          # [C]
        self.theta = None       # [V, C]  P(x_j=1 | class c)
        self._log_theta = None      # [V, C]
        self._log_1m_theta = None   # [V, C]
        self.label_to_index = {}
        self.index_to_label = {}
    

    @staticmethod
    def build_vocab(dataset: List[Tuple[str, str]]) -> List[str]:
        """
        Build vocabulary from a (text,label) dataset.

        Args:
            dataset: List of (text, label)

        Returns:
            vocab: Sorted list of unique tokens
        """
        vocab_set = set()
        for text, _ in dataset:
            for w in text.lower().split():
                vocab_set.add(w)
        return sorted(vocab_set)

    @staticmethod
    def extract_likert_scale(response):
        """
        Extract Likert scale value from response string.

        Args:
            response: String response from CSV

        Returns:
            likert_value: Integer Likert scale value (1-5)
        """
        response = str(response).strip()
        if '—' in response:
            likert_value = int(response.split('—')[0].strip())
        else:
            try:
                likert_value = int(response)
            except:
                likert_value = 3  # Default to neutral if parsing fails
        return likert_value

    @staticmethod
    def make_bow_with_spec(data, spec=None):
        """
        Build features using a persistent per-column vocabulary spec.

        This function returns separate matrices for text (binary bag-of-words)
        and Likert (ordinal integer) features to allow the model to treat them
        appropriately.

        Returns:
            X_text: [N, D_text] binary text features
            X_quant: [N, Q] integer Likert values (1..5)
            t: [N] labels
            spec: metadata needed to reproduce the mapping
        """
        # No data
        if not data or len(data) < 2:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64), spec if spec else {}

        # Extract header and rows
        header = data[0]
        rows = data[1:]

        # Map column names to indices
        name_to_idx = {col_name: idx for idx, col_name in enumerate(header)}

        if spec is None:
            # Resolve text and likert column indices
            text_indices = [name_to_idx[c] for c in text_cols if c in name_to_idx]
            likert_indices = [name_to_idx[c] for c in categorical_cols if c in name_to_idx]

            def tokenize(s):
                return str(s).lower().split()

            # Build per-column vocabularies (for text columns)
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

            # Compute feature offsets for text features only
            feature_offsets = {}
            offset = 0
            for col_idx in text_indices:
                Vj = per_col_vocab_sizes[col_idx]
                feature_offsets[col_idx] = (offset, offset + Vj)
                offset += Vj

            # Total text dimension and number of Likert questions
            D_text = offset
            Q = len(likert_indices)

            # Label mapping
            label_map = {'ChatGPT': 0, 'Claude': 1, 'Gemini': 2}

            # Build the vocabulary specification
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
            # Reuse existing spec
            text_indices = spec['text_indices']
            likert_indices = spec['likert_indices']
            per_col_vocab = spec['per_col_vocab']
            feature_offsets = spec['feature_offsets']
            label_map = spec['label_map']
            D_text = spec['D_text']
            Q = spec['Q']

            def tokenize(s):
                return str(s).lower().split()

        # Core construction: text binary features + likert integer features
        N = len(rows)
        X_text = np.zeros((N, D_text), dtype=np.float32)
        X_quant = np.zeros((N, Q), dtype=np.int64)
        t = np.zeros((N,), dtype=np.int64)

        for i, row in enumerate(rows):
            # Text
            for col_idx in text_indices:
                if col_idx < len(row):
                    words = set(tokenize(row[col_idx]))
                    start, end = feature_offsets[col_idx]
                    lookup = per_col_vocab[col_idx]
                    for w in words:
                        if w in lookup:
                            X_text[i, start + lookup[w]] = 1.0

            # Likert (ordinal) - keep as integer values in X_quant
            for qpos, col_idx in enumerate(likert_indices):
                if col_idx < len(row):
                    val = NaiveBayesModel.extract_likert_scale(row[col_idx])
                    if 1 <= val <= 5:
                        X_quant[i, qpos] = int(val)

            # Label
            if 'label' in name_to_idx:
                lbl = str(row[name_to_idx['label']]).strip()
                t[i] = label_map.get(lbl, 0)
            else:
                assigned = False
                for cell in row:
                    key = str(cell).strip()
                    if key in label_map:
                        t[i] = label_map[key]
                        assigned = True
                        break
                if not assigned:
                    t[i] = 0
        return X_text, X_quant, t, spec

    def naive_bayes_mle(self, X_text: np.ndarray, X_quant: np.ndarray, t: np.ndarray, num_classes: int = 3):
        """
        Vectorized MLE for multiclass Naive Bayes with mixed feature types.

        - `X_text`: binary matrix [N, V_text] for text Bernoulli features
        - `X_quant`: integer matrix [N, Q] with Likert values in {1..5}
        - `t`: labels [N]

        Returns and stores:
            self.pi, self.theta_text [V_text, C], self.theta_quant [Q,5,C]
        """
        N, V_text = X_text.shape
        _, Q = X_quant.shape
        C = num_classes

        # smoothing/pseudo-count: mle uses self.smoothing, map uses alpha=1.0
        alpha = float(self.smoothing) if self.method == 'mle' else 1.0

        # Class counts and priors
        class_counts = np.bincount(t, minlength=C).astype(np.float64)
        pi = class_counts / max(1, N)

        # One-hot label matrix
        Y = np.eye(C, dtype=np.float64)[t]

        # Text (Bernoulli) parameters
        counts_text = X_text.T @ Y  # [V_text, C]
        denom_text = class_counts + 2.0 * alpha  # Bernoulli has two outcomes
        theta_text = (counts_text + alpha) / denom_text
        theta_text = np.clip(theta_text, 1e-12, 1.0 - 1e-12)

        # Likert categorical parameters: for each question q and value v in 1..5
        theta_quant = np.zeros((Q, 5, C), dtype=np.float64)
        for v in range(1, 6):
            mask_v = (X_quant == v).astype(np.float64)  # [N, Q]
            counts_v = mask_v.T @ Y  # [Q, C]
            denom_q = class_counts + 5.0 * alpha
            theta_quant[:, v-1, :] = (counts_v + alpha) / denom_q
        theta_quant = np.clip(theta_quant, 1e-12, 1.0 - 1e-12)

        # Store
        self.pi = pi.astype(np.float64)
        self.theta_text = theta_text.astype(np.float64)
        self.theta_quant = theta_quant.astype(np.float64)
        self._log_theta_text = np.log(self.theta_text)
        self._log_1m_theta_text = np.log(1.0 - self.theta_text)
        self._log_theta_quant = np.log(self.theta_quant)

        return self.pi, self.theta_text, self.theta_quant

    @staticmethod
    def naive_bayes_map(X, t, num_classes: int = 3):
        """
        Compute the parameters $\\pi$ and $\\theta_{jc}$ that maximizes the posterior
        of the provided data (X, t). We will use the beta distribution with
        $a=2$ and $b=2$ for all of our parameters.

        **Your solution should be vectorized, and contain no loops**

        Parameters:
            `X` - a matrix of bag-of-word features of shape [N, V],
                where N is the number of data points and V is the vocabulary size.
                X[i,j] should be either 0 or 1. Produced by the make_bow() function.
            `t` - a vector of class labels of shape [N], with values in [0..C-1].

        Returns:
            `pi` - a vector; the MAP estimate of the parameter $pi_c = p(c)$
            `theta` - a matrix of shape [V, C], where `theta[j, c]` corresponds to
                    the MAP estimate of the parameter $theta_{jc} = p(x_j = 1 | c)$
        """
        N, V = X.shape
        C = num_classes
        a, b = 2.0, 2.0

        # Class prior with Beta prior equivalent to Dirichlet(1,1,...)?
        class_counts = np.bincount(t, minlength=C).astype(np.float64)

        # MAP estimate for pi
        pi = (class_counts + (a - 1)) / (N + C * (a + b - 2))

        # Feature likelihoods with Beta(a,b)
        Y = np.eye(C, dtype=np.float64)[t]
        counts_1 = X.T @ Y  # [V, C]

        # MAP estimate for theta
        theta = (counts_1 + (a - 1)) / (class_counts + (a + b - 2))

        # Numerical safety
        theta = np.clip(theta, 1e-9, 1 - 1e-9)

        return pi, theta
        
    def train(self, train_X, train_t, learning_rate=None, batch_size=None, n_epochs: int = 1):
        """
        Train the model on the provided training data.
        
        Args:
            train_X (np.array): Training data of shape (N, num_features)
                where N is the number of data points
            train_t (np.array): Training targets of shape (N, num_classes)
            learning_rate (float or callable): Learning rate as a schedule function
                that takes current epoch and total epochs as input
            batch_size (int): Number of samples per gradient update
            n_epochs (int): Number of training epochs
            
        Returns:
            dict: Training history (for NB this is a single closed-form fit)
        """
        # Closed-form fit; ignore optimizer params
        num_classes = int(np.max(train_t)) + 1

        # Accept either a single matrix X or a tuple (X_text, X_quant)
        if isinstance(train_X, (list, tuple)) and len(train_X) == 2:
            X_text, X_quant = train_X
        else:
            # If a single matrix is given, assume all features are text Bernoulli
            X_text = train_X
            X_quant = np.zeros((X_text.shape[0], 0), dtype=np.int64)

        if self.method == 'mle':
            pi_est, theta_text_est, theta_quant_est = self.naive_bayes_mle(X_text, X_quant, train_t, num_classes=num_classes)
            # store shapes for history
            theta_est = (theta_text_est.shape, theta_quant_est.shape)
        else:  # 'map'
            # For MAP we currently treat as bernoulli on combined binary features
            pi_est, theta_est = self.naive_bayes_map(X_text, train_t, num_classes=num_classes)
            # Store the parameters in instance variables for MAP as well
            self.pi = pi_est.astype(np.float64)
            self.theta = theta_est.astype(np.float64)
            self._log_theta = np.log(self.theta)
            self._log_1m_theta = np.log(1.0 - self.theta)

        history = {
            'method': self.method,
            'pi': pi_est,
            'theta_shape': theta_est,
        }
        return history
        
    def predict(self, X: np.ndarray):
        """
        Make predictions on given input data.
        
        Args:
            X (np.array): Input data of shape (N, num_features)
            
        Returns:
            np.array: Predicted class indices of shape (N,)
        """
        # Support mixed inputs: either single X matrix, or tuple (X_text, X_quant)
        if isinstance(X, (list, tuple)) and len(X) == 2:
            X_text, X_quant = X
        else:
            X_text = X
            X_quant = np.zeros((X_text.shape[0], 0), dtype=np.int64)

        if self.pi is None:
            raise ValueError("Model is not trained. Call train() first.")

        C = self.pi.shape[0]
        log_pi = np.log(self.pi + 1e-12)  # [C]

        # Text contribution (Bernoulli)
        if hasattr(self, '_log_theta_text') and self._log_theta_text is not None and X_text.shape[1] > 0:
            term_pos = X_text @ self._log_theta_text  # [N, C]
            term_neg = (1.0 - X_text) @ self._log_1m_theta_text
            log_post = term_pos + term_neg + log_pi
        elif hasattr(self, '_log_theta') and self._log_theta is not None:
            term_pos = X_text @ self._log_theta
            term_neg = (1.0 - X_text) @ self._log_1m_theta
            log_post = term_pos + term_neg + log_pi
        else:
            # No text features
            log_post = np.zeros((X_text.shape[0], C), dtype=np.float64) + log_pi

        # Quant (Likert) contribution: add log prob for observed value per question
        if X_quant is not None and X_quant.shape[1] > 0:
            if not hasattr(self, '_log_theta_quant'):
                raise ValueError("Model does not have quant parameters. Train the model with quant features.")
            # For each sample i and question q, add log P(value | class)
            N = X_text.shape[0]
            Q = X_quant.shape[1]
            # Accumulate per-class log-probs
            for q in range(Q):
                vals = X_quant[:, q]  # [N], values in 1..5
                # gather log probs for this question: [5, C]
                logp_q = self._log_theta_quant[q]  # [5, C]
                # For each sample pick the row corresponding to vals-1
                # Build index array
                idx = (vals - 1).clip(0, logp_q.shape[0]-1)
                logp_pick = logp_q[idx]  # [N, C]
                log_post += logp_pick

        return np.argmax(log_post, axis=1)

    def predict_proba(self, X: np.ndarray):
        """
        Return class probabilities for each sample.
        """
        # Support mixed inputs: either single X matrix, or tuple (X_text, X_quant)
        if isinstance(X, (list, tuple)) and len(X) == 2:
            X_text, X_quant = X
        else:
            X_text = X
            X_quant = np.zeros((X_text.shape[0], 0), dtype=np.int64)

        if self.pi is None:
            raise ValueError("Model is not trained. Call train() first.")

        C = self.pi.shape[0]
        log_pi = np.log(self.pi + 1e-12)

        # Text contribution
        if hasattr(self, '_log_theta_text') and self._log_theta_text is not None and X_text.shape[1] > 0:
            term_pos = X_text @ self._log_theta_text
            term_neg = (1.0 - X_text) @ self._log_1m_theta_text
            log_post = term_pos + term_neg + log_pi
        elif hasattr(self, '_log_theta') and self._log_theta is not None:
            term_pos = X_text @ self._log_theta
            term_neg = (1.0 - X_text) @ self._log_1m_theta
            log_post = term_pos + term_neg + log_pi
        else:
            log_post = np.zeros((X_text.shape[0], C), dtype=np.float64) + log_pi

        # Quant contribution
        if X_quant is not None and X_quant.shape[1] > 0:
            if not hasattr(self, '_log_theta_quant'):
                raise ValueError("Model does not have quant parameters. Train the model with quant features.")
            N = X_text.shape[0]
            Q = X_quant.shape[1]
            for q in range(Q):
                vals = X_quant[:, q]
                logp_q = self._log_theta_quant[q]
                idx = (vals - 1).clip(0, logp_q.shape[0]-1)
                logp_pick = logp_q[idx]
                log_post += logp_pick

        # stabilize and convert to probabilities
        log_post -= np.max(log_post, axis=1, keepdims=True)
        probs = np.exp(log_post)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
    
    def save_model(self, file_path):
        """
        Save the model parameters to a file.
        
        Args:
            file_path (str): Path to the file where model parameters will be saved.
        """
        if self.pi is None:
            raise ValueError("Nothing to save: train the model first.")
        save_dict = {
            'pi': self.pi,
            'smoothing': np.array([self.smoothing], dtype=np.float64),
        }
        if hasattr(self, 'theta_text') and self.theta_text is not None:
            save_dict['theta_text'] = self.theta_text
        if hasattr(self, 'theta_quant') and self.theta_quant is not None:
            save_dict['theta_quant'] = self.theta_quant
        if hasattr(self, 'spec') and self.spec is not None:
            save_dict['spec'] = np.array([self.spec], dtype=object)
        np.savez_compressed(file_path, **save_dict)
    
    def load_model(self, file_path):
        """
        Load model parameters from a file.
        
        Args:
            file_path (str): Path to the file from which model parameters will be loaded.
        """
        data = np.load(file_path, allow_pickle=True)
        self.pi = data['pi']
        self.smoothing = float(data['smoothing'][0]) if 'smoothing' in data else 1.0
        # theta_text
        if 'theta_text' in data:
            self.theta_text = data['theta_text']
            self._log_theta_text = np.log(self.theta_text)
            self._log_1m_theta_text = np.log(1.0 - self.theta_text)
        else:
            self.theta_text = None
            self._log_theta_text = None
            self._log_1m_theta_text = None
        # theta_quant
        if 'theta_quant' in data:
            self.theta_quant = data['theta_quant']
            self._log_theta_quant = np.log(self.theta_quant)
        else:
            self.theta_quant = None
            self._log_theta_quant = None
        # spec
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
    print("Training with MLE (Laplace smoothing)...")
    model_mle = NaiveBayesModel(smoothing=0.0, method='mle')
    model_mle.spec = spec
    model_mle.train((X_train_text, X_train_quant), t_train, n_epochs=1)

    if X_valid_text.shape[0] > 0:
        y_pred_mle = model_mle.predict((X_valid_text, X_valid_quant))
        acc_mle = float((y_pred_mle == t_valid).mean())
        print(f"MLE valid accuracy: {acc_mle:.4f} ({y_pred_mle.size} samples)")
    
    print("\nTraining with MAP (Beta(2,2) priors)...")
    model_map = NaiveBayesModel(smoothing=1.0, method='map')
    model_map.spec = spec
    # MAP currently only implemented for Bernoulli text features; pass text matrix
    model_map.train(X_train_text, t_train, n_epochs=1)
    
    if X_valid_text.shape[0] > 0:
        y_pred_map = model_map.predict(X_valid_text)
        acc_map = float((y_pred_map == t_valid).mean())
        print(f"MAP valid accuracy: {acc_map:.4f} ({y_pred_map.size} samples)")
    
    if X_valid_text.shape[0] == 0:
        print("No validation samples; training completed.")
    