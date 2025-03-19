"""
Quantum Machine Learning System: Variational Quantum Classifier with Quantum Feature Maps
Author: [Your Name]
Date: March 19, 2025

This module implements a scalable Quantum Machine Learning System (QMLS) using PennyLane
with NumPy interface that demonstrates potential quantum advantage through expressivity
of quantum feature maps and variational quantum circuits for classification tasks.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp  # PennyLane's NumPy interface
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import time

class QuantumMLSystem:
    """
    Quantum Machine Learning System using Variational Quantum Circuits.
    
    This class implements a quantum machine learning model that leverages quantum feature maps
    for enhancing expressivity and potential quantum advantage.
    """
    
    def __init__(self, n_qubits, n_layers, feature_map_type='ZZ', ansatz_type='strongly_entangling', 
                 optimizer='adam', device='default.qubit', shots=None):
        """
        Initialize the Quantum ML System.
        
        Args:
            n_qubits (int): Number of qubits to use
            n_layers (int): Number of layers in the variational circuit
            feature_map_type (str): Type of feature map ('ZZ', 'amplitude', 'angle')
            ansatz_type (str): Type of variational ansatz ('strongly_entangling', 'basic', 'custom')
            optimizer (str): Optimization algorithm to use ('adam', 'gradient_descent')
            device (str): Quantum device to use
            shots (int): Number of shots for simulation (None for exact)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self.optimizer_name = optimizer
        self.weights = None
        self.weight_shapes = None
        self.feature_dim = None
        self.trained = False
        self.training_history = []
        self.scaler = StandardScaler()
        
        # Set up the quantum circuit with numpy interface
        self.qnode = qml.QNode(self._circuit, self.device, interface="numpy")
        
    def _zz_feature_map(self, x):
        """ZZ Feature map: maps classical data to quantum states with entanglement."""
        # First order expansion
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i % len(x)], wires=i)
            
        # Second order expansion with ZZ entanglement
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(np.pi * x[i % len(x)] * x[j % len(x)], wires=j)
                qml.CNOT(wires=[i, j])
    
    def _amplitude_embedding(self, x):
        """Amplitude embedding: encodes data in the amplitudes of the quantum state."""
        # Normalize the input vector for valid quantum state
        x_normalized = x / np.linalg.norm(x)
        # Pad with zeros if needed
        padding = 2**self.n_qubits - len(x_normalized)
        if padding > 0:
            x_padded = np.pad(x_normalized, (0, padding))
            x_padded = x_padded / np.linalg.norm(x_padded)  # Renormalize
        else:
            x_padded = x_normalized[:2**self.n_qubits]
            x_padded = x_padded / np.linalg.norm(x_padded)  # Renormalize
            
        qml.AmplitudeEmbedding(features=x_padded, wires=range(self.n_qubits), normalize=True)
    
    def _angle_embedding(self, x):
        """Angle embedding: encodes data as rotation angles."""
        qml.AngleEmbedding(features=x, wires=range(self.n_qubits), rotation='X')
        qml.AngleEmbedding(features=x, wires=range(self.n_qubits), rotation='Y')
        
    def _strongly_entangling_ansatz(self, weights):
        """Strongly entangling ansatz: creates deep parameterized circuit."""
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
    
    def _basic_ansatz(self, weights):
        """Basic ansatz with less entanglement."""
        qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
        
    def _custom_ansatz(self, weights):
        """Custom ansatz with controlled rotation gates for increased expressivity."""
        for l in range(self.n_layers):
            # Single qubit rotations
            for i in range(self.n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
            
            # Entangling layer
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i+1) % self.n_qubits])
                
            # Controlled rotations to increase expressivity
            for i in range(self.n_qubits):
                target = (i+1) % self.n_qubits
                qml.CRZ(weights[l, i, 3], wires=[i, target])
        
    def _feature_map(self, x):
        """Apply the selected feature map."""
        if self.feature_map_type == 'ZZ':
            self._zz_feature_map(x)
        elif self.feature_map_type == 'amplitude':
            self._amplitude_embedding(x)
        elif self.feature_map_type == 'angle':
            self._angle_embedding(x)
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")
    
    def _ansatz(self, weights):
        """Apply the selected variational ansatz."""
        if self.ansatz_type == 'strongly_entangling':
            self._strongly_entangling_ansatz(weights)
        elif self.ansatz_type == 'basic':
            self._basic_ansatz(weights)
        elif self.ansatz_type == 'custom':
            self._custom_ansatz(weights)
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
    
    def _circuit(self, x, weights):
        """
        Quantum circuit for classification.

        Args:
            x: Input features
            weights: Trainable circuit parameters

        Returns:
            Expectation value for classification
        """
        # Apply feature map
        self._feature_map(x)
        
        # Apply variational ansatz
        self._ansatz(weights)
        
        # Measure expectation value of Z on the first qubit
        return qml.expval(qml.PauliZ(0))
    
    def _cost_function(self, weights, X, y):
        """
        Cost function for training.
        
        Args:
            weights: Circuit parameters
            X: Input features
            y: Target labels
            
        Returns:
            Mean squared error loss
        """
        predictions = np.array([self.qnode(x, weights) for x in X])
        
        # Convert from [-1,1] to [0,1] for binary classification
        predictions = (predictions + 1) / 2
        
        # Binary cross-entropy loss
        epsilon = 1e-12  # To avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss
    
    def _gradient(self, weights, X, y, eps=0.01):
        """
        Compute gradient using finite difference method.
        
        Args:
            weights: Circuit parameters
            X: Input features
            y: Target labels
            eps: Small perturbation for finite difference
            
        Returns:
            Gradient of the cost function
        """
        grad = np.zeros_like(weights)
        f0 = self._cost_function(weights, X, y)
        
        # Compute gradient for each parameter
        it = np.nditer(weights, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            weights_plus = weights.copy()
            weights_plus[idx] += eps
            f_plus = self._cost_function(weights_plus, X, y)
            
            # Finite difference
            grad[idx] = (f_plus - f0) / eps
            it.iternext()
            
        return grad
    
    def fit(self, X, y, batch_size=32, epochs=100, verbose=1):
        """
        Train the quantum machine learning model.
        
        Args:
            X: Training features
            y: Training labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Scale the input features
        X_scaled = self.scaler.fit_transform(X)
        
        # Save the feature dimension
        self.feature_dim = X.shape[1]
        
        # Determine weight shapes based on the ansatz type
        if self.ansatz_type == 'strongly_entangling':
            weight_shape = (self.n_layers, self.n_qubits, 3)
        elif self.ansatz_type == 'basic':
            weight_shape = (self.n_layers, self.n_qubits)
        elif self.ansatz_type == 'custom':
            weight_shape = (self.n_layers, self.n_qubits, 4)
            
        # Initialize weights
        self.weights = np.random.uniform(0, 2*np.pi, size=weight_shape)
        
        # Set up the optimizer
        if self.optimizer_name == 'adam':
            optimizer = qml.AdamOptimizer(stepsize=0.01)
        else:
            optimizer = qml.GradientDescentOptimizer(stepsize=0.01)
            
        # Track training history
        self.training_history = []
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Split data into batches
            permutation = np.random.permutation(len(X_scaled))
            X_shuffled = X_scaled[permutation]
            y_shuffled = y[permutation]
            
            epoch_loss = 0
            num_batches = int(np.ceil(len(X_scaled) / batch_size))
            
            for i in range(num_batches):
                batch_indices = slice(i * batch_size, min((i + 1) * batch_size, len(X_scaled)))
                X_batch = X_shuffled[batch_indices]
                y_batch = y_shuffled[batch_indices]
                
                # Compute cost and update parameters
                def cost_fn(weights):
                    return self._cost_function(weights, X_batch, y_batch)
                
                # Update weights using the optimizer
                self.weights = optimizer.step(cost_fn, self.weights)
                
                batch_loss = cost_fn(self.weights)
                epoch_loss += batch_loss
            
            epoch_loss /= num_batches
            epoch_time = time.time() - start_time
            
            self.training_history.append(epoch_loss)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Time: {epoch_time:.2f}s")
                
        self.trained = True
        return self.training_history
    
    def predict(self, X, return_proba=False):
        """
        Make predictions using the trained quantum model.
        
        Args:
            X: Input features
            return_proba: Whether to return probabilities instead of class labels
            
        Returns:
            Predictions or probabilities
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")
            
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = np.array([self.qnode(x, self.weights) for x in X_scaled])
        
        # Convert from [-1,1] to [0,1] for binary classification probabilities
        probabilities = (predictions + 1) / 2
        
        if return_proba:
            return probabilities
        else:
            return (probabilities > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict(X, return_proba=True)
        
        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        
        return {
            "accuracy": acc,
            "confusion_matrix": conf_matrix,
            "predictions": y_pred,
            "probabilities": y_proba
        }
    
    def plot_training_history(self):
        """Plot the training loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        
    def compare_with_classical(self, X_train, y_train, X_test, y_test, classical_models=None):
        """
        Compare the quantum model with classical models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            classical_models: List of classical models to compare with
            
        Returns:
            Dictionary of comparison results
        """
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        
        if classical_models is None:
            classical_models = {
                'SVM': SVC(probability=True),
                'Random Forest': RandomForestClassifier(),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
            }
            
        results = {
            'Quantum Model': self.evaluate(X_test, y_test)['accuracy']
        }
        
        # Scale the data for classical models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate classical models
        for name, model in classical_models.items():
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            start_time = time.time()
            accuracy = model.score(X_test_scaled, y_test)
            inference_time = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'inference_time': inference_time
            }
            
        return results

    def circuit_complexity_analysis(self):
        """
        Analyze the complexity of the quantum circuit.
        
        Returns:
            Dictionary with circuit complexity metrics
        """
        # Generate a random input
        x = np.random.normal(0, 1, size=self.feature_dim)
        x = self.scaler.transform([x])[0]
        
        # Generate random weights
        if self.ansatz_type == 'strongly_entangling':
            random_weights = np.random.uniform(0, 2*np.pi, size=(self.n_layers, self.n_qubits, 3))
        elif self.ansatz_type == 'basic':
            random_weights = np.random.uniform(0, 2*np.pi, size=(self.n_layers, self.n_qubits))
        elif self.ansatz_type == 'custom':
            random_weights = np.random.uniform(0, 2*np.pi, size=(self.n_layers, self.n_qubits, 4))
        
        # Create a tape to analyze the circuit
        with qml.tape.QuantumTape() as tape:
            self._feature_map(x)
            self._ansatz(random_weights)
            qml.expval(qml.PauliZ(0))
            
        # Count gates by type
        op_dict = {}
        for op in tape.operations:
            op_name = op.name
            if op_name in op_dict:
                op_dict[op_name] += 1
            else:
                op_dict[op_name] = 1
                
        # Calculate circuit depth and width
        depth = len(tape.operations)
        width = self.n_qubits
        
        # Calculate number of parameters
        if self.ansatz_type == 'strongly_entangling':
            num_params = self.n_layers * self.n_qubits * 3
        elif self.ansatz_type == 'basic':
            num_params = self.n_layers * self.n_qubits
        elif self.ansatz_type == 'custom':
            num_params = self.n_layers * self.n_qubits * 4
            
        return {
            'depth': depth,
            'width': width,
            'num_parameters': num_params,
            'gate_counts': op_dict
        }
    
    def barren_plateau_analysis(self, num_samples=10):
        """
        Analyze the model for barren plateaus by measuring gradient variance.
        
        Args:
            num_samples: Number of random parameter vectors to sample
            
        Returns:
            Dictionary with gradient statistics
        """
        # Generate random data
        X = np.random.normal(0, 1, size=(10, self.feature_dim))
        X_scaled = self.scaler.transform(X)
        y = np.random.randint(0, 2, size=10)
        
        gradient_norms = []
        
        # Sample random parameter vectors and compute gradient norms
        for _ in range(num_samples):
            if self.ansatz_type == 'strongly_entangling':
                random_weights = np.random.uniform(0, 2*np.pi, size=(self.n_layers, self.n_qubits, 3))
            elif self.ansatz_type == 'basic':
                random_weights = np.random.uniform(0, 2*np.pi, size=(self.n_layers, self.n_qubits))
            elif self.ansatz_type == 'custom':
                random_weights = np.random.uniform(0, 2*np.pi, size=(self.n_layers, self.n_qubits, 4))
            
            # Compute gradient using finite difference
            grad = self._gradient(random_weights, X_scaled, y)
            gradient_norm = np.linalg.norm(grad)
            gradient_norms.append(gradient_norm)
            
        return {
            'mean_gradient_norm': np.mean(gradient_norms),
            'std_gradient_norm': np.std(gradient_norms),
            'min_gradient_norm': np.min(gradient_norms),
            'max_gradient_norm': np.max(gradient_norms)
        }
    
    def save_model(self, filename):
        """
        Save the model parameters to a file.
        
        Args:
            filename: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")
            
        model_data = {
            'weights': self.weights,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'feature_map_type': self.feature_map_type,
            'ansatz_type': self.ansatz_type,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'feature_dim': self.feature_dim
        }
        
        np.save(filename, model_data)
        
    @classmethod
    def load_model(cls, filename, device='default.qubit', shots=None):
        """
        Load a saved model.
        
        Args:
            filename: Path to the saved model
            device: Quantum device to use
            shots: Number of shots for simulation
            
        Returns:
            Loaded model
        """
        model_data = np.load(filename, allow_pickle=True).item()
        
        # Create a new model with the saved parameters
        model = cls(
            n_qubits=model_data['n_qubits'],
            n_layers=model_data['n_layers'],
            feature_map_type=model_data['feature_map_type'],
            ansatz_type=model_data['ansatz_type'],
            device=device,
            shots=shots
        )
        
        # Restore the weights
        model.weights = model_data['weights']
        
        # Restore the scaler
        model.scaler = StandardScaler()
        model.scaler.mean_ = model_data['scaler_mean']
        model.scaler.scale_ = model_data['scaler_scale']
        
        # Restore other attributes
        model.feature_dim = model_data['feature_dim']
        model.trained = True
        
        return model


# Example usage
def run_quantum_ml_experiment():
    """Run a complete quantum ML experiment with comparison to classical ML."""
    # Generate synthetic dataset
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=0, 
                              random_state=42, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train quantum model
    print("Training Quantum ML model...")
    qml_model = QuantumMLSystem(n_qubits=4, n_layers=2, feature_map_type='ZZ', 
                               ansatz_type='strongly_entangling')
    qml_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
    
    # Evaluate quantum model
    print("\nEvaluating Quantum ML model...")
    qml_results = qml_model.evaluate(X_test, y_test)
    print(f"Quantum ML Accuracy: {qml_results['accuracy']:.4f}")
    
    # Compare with classical models
    print("\nComparing with classical models...")
    comparison = qml_model.compare_with_classical(X_train, y_train, X_test, y_test)
    
    for model_name, results in comparison.items():
        if model_name == 'Quantum Model':
            print(f"{model_name} Accuracy: {results:.4f}")
        else:
            print(f"{model_name} Accuracy: {results['accuracy']:.4f}")
    
    # Circuit complexity analysis
    print("\nAnalyzing circuit complexity...")
    complexity = qml_model.circuit_complexity_analysis()
    print(f"Circuit depth: {complexity['depth']}")
    print(f"Number of parameters: {complexity['num_parameters']}")
    print("Gate counts:")
    for gate, count in complexity['gate_counts'].items():
        print(f"  {gate}: {count}")
    
    # Check for barren plateaus
    print("\nAnalyzing gradient statistics for barren plateaus...")
    gradient_stats = qml_model.barren_plateau_analysis(num_samples=5)
    print(f"Mean gradient norm: {gradient_stats['mean_gradient_norm']:.6f}")
    print(f"Std of gradient norm: {gradient_stats['std_gradient_norm']:.6f}")
    
    # Visualize training history
    qml_model.plot_training_history()
    
    return qml_model


if __name__ == "__main__":
    model = run_quantum_ml_experiment()
