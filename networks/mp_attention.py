
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shift(x, direction, amount):
    if direction == 'left':
        ret = F.pad(x, (amount,0,0,0,0,0,0,0))[:,:,:,:-amount]
    elif direction == 'right':
        ret = F.pad(x, (0,amount,0,0,0,0,0,0))[:,:,:,amount:]
    elif direction == 'top':
        ret = F.pad(x, (0,0,amount,0,0,0,0,0))[:,:,:-amount,:]
    elif direction == 'bottom':
        ret = F.pad(x, (0,0,0,amount,0,0,0,0))[:,:,amount:,:]
    else:
        raise
    return ret


class ManifoldPropagation(nn.Module):
    def __init__(self, ic, k_hop=5):
        super().__init__()
        self.k = nn.Conv2d(ic, ic, kernel_size=1, padding=0)
        self.q = nn.Conv2d(ic, ic, kernel_size=1, padding=0)
        self.v = nn.Conv2d(ic, ic, kernel_size=1, padding=0)
        self.k_hop = k_hop
        self.normalize = nn.Softmax(dim=1)
        self.aggregate = nn.Conv2d(ic, ic, kernel_size=1)

    def forward(self, x):

        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        batch_size, channel, h, w = x.shape
        xl = shift(k, 'left', 1)
        xr = shift(k, 'right', 1)
        xt = shift(k, 'top', 1)
        xb = shift(k, 'bottom', 1)

#         l = torch.exp((x * xl).sum(1, keepdim=True))
#         r = torch.exp((x * xr).sum(1, keepdim=True))
#         t = torch.exp((x * xt).sum(1, keepdim=True))
#         b = torch.exp((x * xb).sum(1, keepdim=True))
#         m = torch.exp(torch.ones_like(l).to(device))
#         # softmax
#         z = l + r + t + b + m
#         l = l / z
#         r = r / z
#         t = t / z
#         m = m / z
#         b = b / z
#         print(z[0,0,0])

        l = (q * xl).sum(1, keepdim=True)
        r = (q * xr).sum(1, keepdim=True)
        t = (q * xt).sum(1, keepdim=True)
        b = (q * xb).sum(1, keepdim=True)
        m = torch.ones_like(l).to(device)

#         l = l.detach()
#         r = l.detach()
#         t = l.detach()
#         b = l.detach()

        A = self.normalize(torch.cat((l,r,t,b,m), dim=1))
        l = A[:,0:1]
        r = A[:,1:2]
        t = A[:,2:3]
        b = A[:,3:4]
        m = A[:,4:5]

        #print(l[0,0,0])

        for _ in range(self.k_hop):
            v = self.propagation(v, l, r, t, b, m)
        v = self.aggregate(v)
        return v


    def propagation(self, x, l, r, t, b, m):

        p = l * shift(x, 'right', 1) + \
            r * shift(x, 'left', 1) + \
            t * shift(x, 'bottom', 1) + \
            b * shift(x, 'top', 1) + \
            m * x
        return p
