from micrograd.engine import Value
from micrograd.nn import *

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

for k in range(1000):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)

print(ypred)

# loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
# print(loss)
# print(ypred)
# loss.backward()

# for p in n.parameters():
#     p.data += -0.01 * p.grad

# ypred = [n(x) for x in xs]

# loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
# print(loss)
# print(ypred)