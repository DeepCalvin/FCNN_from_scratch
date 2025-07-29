import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_derivative(Z):
    return np.where(Z<0, 0, 1)

def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def Sigmoid_derivative(Z):
    s = Sigmoid(Z)
    return s*(1-s)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]


activations = {
    'relu': (ReLU, ReLU_derivative),
    'sigmoid': (Sigmoid, Sigmoid_derivative)
}

class Dense:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.randn(output_size, input_size)
        self.B = np.random.randn(output_size, 1)
        self.activation_func, self.activation_func_derivative = activations[activation]

    def forward(self, X): # X.shape = (samples, input_size)
        self.Z = np.dot(X, self.W.T) + self.B.T # self.Z.shape = (samples, output_size)
        self.A = self.activation_func(self.Z) # self.A.shape = (samples, output_size)
        return self.A
    
    def backward(self, dLdA, X): # dLdA.shape = (samples, output_size), self.A.shape = (samples, input_size)
        dAdZ = self.activation_func_derivative(self.Z) # self.dAdZ.shape = (samples, output_size)
        dLdZ = dLdA * dAdZ # dLdZ.shape = (samples, output_size)

        self.dLdW = np.dot(dLdZ.T, X) / X.shape[0] # dLdW.shape = (output_size, input_size), intuitively, as dLdW is the direction and magnitude of move W, we have that gradient for each neuron, and we update each w_i in that neuron, we divide by the number of samples so learning is stabilized
        self.dLdB = np.sum(dLdZ, axis=0).reshape(-1, 1) / X.shape[0] # dLdB.shape = (output_size, 1), intuitively, similar to dLdB, it's the learning of how to move B, and we have n neurons each with only 1 B term, we also divide by X.shape[0] for stability

        # Both dLdW and dLdB are matching the shape of W and B in the first place, and this is because we can now gradient descent easily with the same shape later, very convinient for sure!
        
        # Updating dLdA, so usually in the back prop, dLdZ = dLdA * dAdZ, where dLdA is the gradient from layer ahead of us, and dAdZ is the current layer's gradient
        # From dLdZ, we calculated dLdW and dLdB by multiplying them by dZdW (X) and dZdB (1)
        # Now we pass this information into the layer behind us, so we  calculate dLdZ * dZdA, which is dLdA_prev, exactly what we received at the beggining
        # dLdZ is the gradient we already have, and dZdA is just the weight:

        dLdA_prev = np.dot(dLdZ, self.W) # dLdA_prev.shape = (samples, output_size), exactly like how we received it at the beggining

        # To intuitively link back to why dLdA_prev is in the shape of (samples, output_size), it's because saw how much OUR layer's output affect the loss, so it's saying how does loss change per sample for each output of our neuron

        return dLdA_prev
    
    # Gradient descent update W and B
    def update(self, lr):
        self.W -= lr * self.dLdW
        self.B -= lr * self.dLdB



class FCNN:
    def __init__(self, layer_size: list, activationFuncs: list): # layer_size is a list of input size for each layer
        self.layers = []

        for i in range(len(layer_size) -1):
            self.layers.append(Dense(layer_size[i], layer_size[i+1], activationFuncs[i]))

    def forward(self, X):
        self.inputs = [] # This contains a list of inputs to the next layer so that during back prop, it can be reused
        self.inputs.append(X)
        out = X

        for layer in self.layers:
            out = layer.forward(out)
            self.inputs.append(out)

        return out # Return final prediction to compute loss
    
    def backward(self, dLdA, lr):
        for i in range(len(self.layers)-1, -1, -1): # Rervse through the layers list
            dLdA = self.layers[i].backward(dLdA, self.inputs[i])

            print(f"Layer {i} gradients:")
            print(f"dLdW (shape {self.layers[i].dLdW.shape}):\n{self.layers[i].dLdW}")
            print(f"dLdB (shape {self.layers[i].dLdB.shape}):\n{self.layers[i].dLdB}")
            print()

            self.layers[i].update(lr)

# Train
def fit(epochs, lr, X_train, y_train, batch_size=32):
    mse_list = np.array([])
    
    for epoch in range(epochs):
        
        # Shuffle data every epoch to reduce chance of overfitting
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Enter batch loop
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_pred = model.forward(X_batch)
            loss = mse_loss(y_batch, y_pred)
            dLdA = mse_loss_derivative(y_batch, y_pred)
            model.backward(dLdA, lr)

        mse_list = np.append(mse_list, loss)

        print(f"Epoch {epoch}: MSE = {loss:.4f}")

    print(np.argmin(mse_list))


# Using fake dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load iris
iris = load_iris()
X = iris.data # (150, 4)
y = iris.target # 0 = setosa, 1 = versicolor, 2 = virginica

# Binary classification Setosa vs Not Setosa
y_binary = (y == 0).astype(np.float32).reshape(-1, 1) # 1 = Setosa, 0 = others

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)


model = FCNN([4, 128, 128, 1], ['relu', 'relu', 'sigmoid'])

fit(
    epochs=200,
    lr=0.1,
    X_train=X_train,
    y_train=y_train,
    batch_size=16
)

