import torch
import torch.nn as nn

class Rosenbrock(nn.Module):
    def __init__(self, a, b):
        super(Rosenbrock, self).__init__()
        # Initializing the Rosenbrock function
        self.a = a
        self.b = b
        # Optimization parameters are randomly initialized and
        # defined to be a nn.Parameter object.
        self.x = torch.nn.Parameter(torch.Tensor([-1.0]))
        self.y = torch.nn.Parameter(torch.Tensor([2.0]))
    
    def forward(self, x):
        # Here is the function that is being optimized
        return (self.x - self.a) ** 2 + self.b * (self.y - self.x ** 2) ** 2
