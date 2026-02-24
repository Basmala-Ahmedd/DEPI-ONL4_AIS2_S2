import numpy as np
import matplotlib.pyplot as plt

class LinearModel_Basmala:
    def __init__(self,alpha=0.01,num_iters=1000,degree=1,regularization=None,lambda_=0.1):
        self.alpha =alpha
        self.num_iters = num_iters
        self.degree = degree
        self.regularization = regularization
        self.lambda_ = lambda_

        
        self.theta = None
        self.sse_history = []
        
    def _polynomial_features(self, X):
        X = X.reshape(-1,1)
        X_poly = np.ones((len(X),1))
    
        for d in range(1, self.degree+1):
            X_poly = np.hstack((X_poly, X**d))
            
        return X_poly
        
    def fit(self,X,y):
        self.sse_history = []
        X_poly = self._polynomial_features(X)
        m, n = X_poly.shape
        
        self.theta = np.zeros(n)
        
        for i in range(self.num_iters):
            y_hat = X_poly @ self.theta
            error = y_hat - y
            
            gradients = (2/m) * (X_poly.T @ error)
            
            if self.regularization == "ridge":
                gradients[1:] += 2 * self.lambda_ * self.theta[1:]
                
            elif self.regularization == "lasso":
                gradients[1:] += self.lambda_ * np.sign(self.theta[1:])
            
            self.theta -= self.alpha * gradients
            
            sse = np.sum(error ** 2)
            self.sse_history.append(sse)
            
            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}, SSE = {sse}")
    
    
    def predict(self,X):
        X_poly = self._polynomial_features(X)
        return X_poly @ self.theta

    def plot_training(self, X, y):
        
        plt.figure(figsize=(12,4))
        
        # SSE
        plt.subplot(1,2,1)
        plt.plot(self.sse_history)
        plt.title("SSE Over Iterations")
        
        # Fit Line
        plt.subplot(1,2,2)
        plt.scatter(X, y, color="red", label="Data Point")

        X_sorted = np.sort(X)
        y_pred = self.predict(X_sorted)
        
        if self.regularization == "ridge":
            model_name = "Ridge Regression"
        elif self.regularization == "lasso":
            model_name = "Lasso Regression"
        else:
            model_name = "Linear Regression"

        if self.degree > 1:
            model_name += f" (Degree {self.degree})"

        plt.plot(X_sorted, y_pred, color="green", label=model_name)

        plt.title(model_name)
        plt.legend()

# Bonus
    
    def mse(self, X, y):
        y_hat= self.predict(X)
        return np.mean((y - y_hat)**2)
