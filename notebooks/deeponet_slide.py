import torch
import torch.nn as nn

# Operator to learn:  u'' = a  on [0,1],  u(0)=u(1)=0
m = 100
x = torch.linspace(0, 1, m)
k = torch.arange(1, 9).float()                 # 8 sine modes
S = torch.sin(torch.pi * x[:, None] * k)       # (m, 8) sines at sensors

# Random forcings a and their EXACT solutions u
def sample(n):
    c = torch.randn(n, 8) / k
    a = c @ S.T                                # a = sum c_k sin(k pi x)
    u = -(c / (torch.pi * k) ** 2) @ S.T       # exact: u'' = a
    return a, u

# Branch = which function ;  Trunk = where to evaluate
branch = nn.Sequential(nn.Linear(m, 64), nn.Tanh(), nn.Linear(64, 40))
trunk  = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 40))
b0  = nn.Parameter(torch.zeros(1))
opt = torch.optim.Adam([*branch.parameters(), *trunk.parameters(), b0], lr=1e-3)

y = x[:, None]                                 # query grid (m,1)
a_tr, u_tr = sample(2000)

# Training
for step in range(6000):
    i     = torch.randint(0, 2000, (64,))
    coef  = branch(a_tr[i])                    # (64,40) coefficients b_k
    basis = trunk(y)                           # (m,40)  basis t_k(y)
    pred  = coef @ basis.T + b0                # G(a)(y) = sum_k b_k t_k
    loss  = ((pred - u_tr[i]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# Test on an unseen forcing a(x)
a_te, u_te = sample(1)
with torch.no_grad():
    u_pred = branch(a_te) @ trunk(y).T + b0
j = m // 2                                     # midpoint x = 0.5
print("exact    u(0.5) =", u_te[0, j].item())
print("DeepONet u(0.5) =", u_pred[0, j].item())
