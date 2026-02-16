import tkinter as tk
import numpy as np

class SalaryPredictionApp:
    def __init__(self,root):
        self.root = root
        self.root.title("Depi R4 - Machine Learning Diploma")
        self.root.geometry("500x400")
    def create_weidget(self):
        header = tk.label(self.root,text="DEPI",bg="blue",fg="white")
        
        
        
        
        if __name__ == "__main__":
            root = tk.Tk()
            app = SalaryPredictionApp(root)
            root.mainloop()