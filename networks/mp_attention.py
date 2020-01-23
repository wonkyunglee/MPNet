
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
    def __init__(self, ic, k_hop=3, stride=1):
        super().__init__()
        self.k = nn.Conv2d(ic, ic, kernel_size=1, padding=0)
        self.q = nn.Conv2d(ic, ic, kernel_size=1, padding=0)
        self.v = nn.Conv2d(ic, ic, kernel_size=1, padding=0)
        self.k_hop = k_hop
        self.stride = stride
        self.normalize = nn.Softmax(dim=1)
        self.aggregate = nn.Conv2d(ic, ic, kernel_size=1)

    def forward(self, x):

        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        batch_size, channel, h, w = x.shape
        xl = shift(k, 'left', self.stride)
        xr = shift(k, 'right', self.stride)
        xt = shift(k, 'top', self.stride)
        xb = shift(k, 'bottom', self.stride)

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

        p = l * shift(x, 'right', self.stride) + \
            r * shift(x, 'left', self.stride) + \
            t * shift(x, 'bottom', self.stride) + \
            b * shift(x, 'top', self.stride) + \
            m * x
        return p
