import numpy as np
import random
import csv
from typing import List, Tuple

# NOTE: If there are parameters that are not used, keep them in the function signature for consistency.

text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
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
    # Hello
    # TODO: Put any needed helper functions here
    # e.g., debugging utilities, model-specific methods, forward pass, back propagation etc.

    def __init__(self, smoothing: float = 1.0, method: str = 'mle'): # TODO: Consider adding hyperparameters/branch out to diff versions of Naive Bayes
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

        If spec is None: build vocabularies from the provided data (treat as training set)
        and return (X, t, spec).
        If spec is provided: reuse vocab/offsets to produce feature matrix with identical
        dimensionality/order. Unknown words are ignored.

        Returns:
            X: [N, D] feature matrix
            t: [N] labels
            spec: dict containing vocabulary/offset metadata for reuse
        """
        # No data
        if not data or len(data) < 2:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), spec if spec else {}

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

            # Compute feature offsets and total dimension D
            feature_offsets = {}
            offset = 0
            for col_idx in text_indices:
                Vj = per_col_vocab_sizes[col_idx]
                feature_offsets[col_idx] = (offset, offset + Vj)
                offset += Vj
            for col_idx in likert_indices:
                feature_offsets[col_idx] = (offset, offset + 5)
                offset += 5

            # Total dimension
            D = offset

            # Label mapping
            label_map = {'ChatGPT': 0, 'Claude': 1, 'Gemini': 2}

            # Build the vocabulary specification
            spec = {
                'text_indices': text_indices,
                'likert_indices': likert_indices,
                'per_col_vocab': per_col_vocab,
                'feature_offsets': feature_offsets,
                'label_map': label_map,
                'D': D,
            }
        else:
            # Reuse existing spec
            text_indices = spec['text_indices']
            likert_indices = spec['likert_indices']
            per_col_vocab = spec['per_col_vocab']
            feature_offsets = spec['feature_offsets']
            label_map = spec['label_map']
            D = spec['D']

            def tokenize(s):
                return str(s).lower().split()

        # Core bag-of-words construction
        N = len(rows)
        X = np.zeros((N, D), dtype=np.float32)
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
                            X[i, start + lookup[w]] = 1.0
            # Likert
            for col_idx in likert_indices:
                if col_idx < len(row):
                    val = NaiveBayesModel.extract_likert_scale(row[col_idx])
                    if 1 <= val <= 5:
                        start, end = feature_offsets[col_idx]
                        X[i, start + (val - 1)] = 1.0
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
        return X, t, spec

    def naive_bayes_mle(self, X: np.ndarray, t: np.ndarray, num_classes: int = 3):
        """
        Vectorized MLE for multiclass Bernoulli Naive Bayes with Laplace smoothing.

        Args:
            X: [N, V] binary feature matrix (multi-hot across text+Likert one-hots)
            t: [N] integer labels in [0..C-1]
            num_classes: C

        Sets:
            self.pi  [C]
            self.theta [V, C]
        """
        # N = number of samples, V = vocabulary size
        N, V = X.shape
        C = num_classes
        alpha = self.smoothing

        # Class counts and priors (MLE)
        class_counts = np.bincount(t, minlength=C).astype(np.float64)  # [C]
        pi = class_counts / max(1, N)

        # Compute feature counts per class: for each class c, sum X over samples in class
        # Build an indicator matrix for classes: [N, C]
        Y = np.eye(C, dtype=np.float32)[t]  # one-hot labels
        # counts_1: [V, C] = X^T @ Y
        counts_1 = X.T @ Y

        # For Bernoulli NB, denominator per class is number of docs in class
        denom = class_counts + 2 * alpha  # [C]
        theta = (counts_1 + alpha) / denom  # broadcasting over columns

        # Numerical safety
        theta = np.clip(theta, 1e-9, 1 - 1e-9)

        self.pi = pi.astype(np.float64)
        self.theta = theta.astype(np.float64)
        self._log_theta = np.log(self.theta)
        self._log_1m_theta = np.log(1.0 - self.theta)
        return self.pi, self.theta


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
        pi = (class_counts + (a - 1)) / (N + C * (a + b - 2))

        # Feature likelihoods with Beta(a,b)
        Y = np.eye(C, dtype=np.float64)[t]
        counts_1 = X.T @ Y  # [V, C]
        denom = class_counts + (a + b - 2)  # [C]
        theta = (counts_1 + (a - 1)) / denom
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
        
        if self.method == 'mle':
            pi_est, theta_est = self.naive_bayes_mle(train_X, train_t, num_classes=num_classes)
        else:  # 'map'
            pi_est, theta_est = self.naive_bayes_map(train_X, train_t, num_classes=num_classes)
            # Store the parameters in instance variables for MAP as well
            self.pi = pi_est.astype(np.float64)
            self.theta = theta_est.astype(np.float64)
            self._log_theta = np.log(self.theta)
            self._log_1m_theta = np.log(1.0 - self.theta)
        
        history = {
            'method': self.method,
            'pi': pi_est,
            'theta_shape': theta_est.shape,
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
        if self.pi is None or self.theta is None:
            raise ValueError("Model is not trained. Call train() first.")

        # Compute log P(c) + sum_j [ x_j log theta_jc + (1-x_j) log (1-theta_jc) ]
        log_pi = np.log(self.pi + 1e-12)  # [C]
        # [N, V] @ [V, C] -> [N, C]
        term_pos = X @ self._log_theta
        term_neg = (1.0 - X) @ self._log_1m_theta
        log_post = term_pos + term_neg + log_pi
        return np.argmax(log_post, axis=1)

    def predict_proba(self, X: np.ndarray):
        """
        Return class probabilities for each sample.
        """
        if self.pi is None or self.theta is None:
            raise ValueError("Model is not trained. Call train() first.")
        log_pi = np.log(self.pi + 1e-12)
        term_pos = X @ self._log_theta
        term_neg = (1.0 - X) @ self._log_1m_theta
        log_post = term_pos + term_neg + log_pi
        # stabilize
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
        if self.pi is None or self.theta is None:
            raise ValueError("Nothing to save: train the model first.")
        np.savez_compressed(
            file_path,
            pi=self.pi,
            theta=self.theta,
            smoothing=np.array([self.smoothing], dtype=np.float64),
        )
    
    def load_model(self, file_path):
        """
        Load model parameters from a file.
        
        Args:
            file_path (str): Path to the file from which model parameters will be loaded.
        """
        data = np.load(file_path, allow_pickle=False)
        self.pi = data['pi']
        self.theta = data['theta']
        self.smoothing = float(data['smoothing'][0]) if 'smoothing' in data else 1.0
        self._log_theta = np.log(self.theta)
        self._log_1m_theta = np.log(1.0 - self.theta)
    

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
    X_train, t_train, spec = NaiveBayesModel.make_bow_with_spec(train_data, spec=None)
    X_valid, t_valid, _ = NaiveBayesModel.make_bow_with_spec(valid_data, spec=spec)

    # Train and evaluate with both methods
    print("Training with MLE (Laplace smoothing)...")
    model_mle = NaiveBayesModel(smoothing=0.0, method='mle')
    model_mle.train(X_train, t_train, n_epochs=1)

    if X_valid.shape[0] > 0:
        y_pred_mle = model_mle.predict(X_valid)
        acc_mle = float((y_pred_mle == t_valid).mean())
        print(f"MLE valid accuracy: {acc_mle:.4f} ({y_pred_mle.size} samples)")
    
    print("\nTraining with MAP (Beta(2,2) priors)...")
    model_map = NaiveBayesModel(smoothing=1.0, method='map')
    model_map.train(X_train, t_train, n_epochs=1)
    
    if X_valid.shape[0] > 0:
        y_pred_map = model_map.predict(X_valid)
        acc_map = float((y_pred_map == t_valid).mean())
        print(f"MAP valid accuracy: {acc_map:.4f} ({y_pred_map.size} samples)")
    
    if X_valid.shape[0] == 0:
        print("No validation samples; training completed.")
    