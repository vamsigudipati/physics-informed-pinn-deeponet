import torch
import torch.nn as nn

# Collocation points
x = torch.linspace(0, 3.14, 100).reshape(-1, 1)
x.requires_grad_(True)

# Sparse observations
x_data = torch.tensor([[0.5],
                       [1.0],
                       [1.5],
                       [2.0]])

y_data = torch.sin(2 * x_data)   # synthetic measurements

# MLP
net = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

# Unknown parameter λ
lam = nn.Parameter(torch.tensor([1.0]))

opt = torch.optim.Adam(
    list(net.parameters()) + [lam],
    lr=2e-3
)

# Training
for i in range(10000):

    # Prediction
    y = net(x)

    # First derivative
    dy = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True)[0]

    # Second derivative
    d2y = torch.autograd.grad(
        dy, x,
        grad_outputs=torch.ones_like(dy),
        create_graph=True)[0]

    # Physics
    ode_loss = ((d2y + lam * y) ** 2).mean()

    # Data mismatch
    y_pred_data = net(x_data)
    data_loss = ((y_pred_data - y_data) ** 2).mean()
   
    loss = ode_loss + data_loss
    opt.zero_grad()
    loss.backward()
    opt.step()

print("True λ      = 4")
print("Estimated λ =", lam.item())