import numpy as np
import matplotlib.pyplot as plt

class LinearRegression_Basmala:
    def __init__(self,alpha,num_iters):
        self.alpha =alpha
        self.num_iters = num_iters
        self.theta_0 = 0
        self.theta_1 = 0
        self.sse_history = []
        
    def fit(self,X,y):
        for i in range(self.num_iters):
            y_hat = self.theta_1 * X + self.theta_0
            
            D_theta_0 = (2/len(X)) * np.sum(y_hat - y)
            D_theta_1 = (2/len(X)) * np.sum((y_hat - y) * X)
            
            self.theta_0 -= self.alpha * D_theta_0
            self.theta_1 -= self.alpha * D_theta_1
            sse = (np.sum((y_hat - y) ** 2))
            self.sse_history.append(sse)

            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}, SSE = {sse}")
    
    def predict(self,X):
        y_hat = self.theta_1 * X + self.theta_0
        return y_hat 
    
    def plot_training(self, X, y):
        
        plt.figure(figsize=(12,5))
        
        # SSE 
        plt.subplot(1,2,1)
        plt.plot(range(self.num_iters),self.sse_history,label='SSE',color="green")
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.title("SEE Over Iteraion")
        plt.legend()

        
        # Regression Line
        plt.subplot(1,2,2)
        plt.scatter(X, y, color="red", label="Data Point")
        plt.plot(X, self.theta_1 * X + self.theta_0, color ="green",label ="Regression Line")
        plt.xlabel("House size (m²)")
        plt.ylabel("Price (thousand $)")
        plt.title("Linear Regression Fit Line")
        plt.show()
        
    # Bonus
    
    def mse(self, X, y):
        y_hat = self.predict(X)
        return np.mean((y - y_hat)**2)
