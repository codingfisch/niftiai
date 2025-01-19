import torch
import torch.nn as nn
NORM_CLS = nn.BatchNorm3d
ACT_MODULE = nn.ReLU(inplace=True)


class ResNet3d(nn.Sequential):
    def __init__(self, shape, c_in=1, n_out=1, c_start=16, n_blocks=(3,), kernel=3, pool1_kernel=2,
                 p_dropout=.5, cat_pool=False, bias=False, norm_cls=NORM_CLS, act_module=ACT_MODULE):
        conv1 = nn.Conv3d(c_in, c_start, kernel, stride=2, bias=bias)
        pool1 = nn.MaxPool3d(kernel_size=pool1_kernel, stride=2, padding=1)
        norm1 = norm_cls(c_start)
        res_blocks = []
        for i, nb in enumerate(n_blocks):
            stride = 1 if i in [0, len(n_blocks) - 1] else [2 if e > max(shape) // 2 else 1 for e in shape]
            c_in = 2 ** i * c_start
            block1 = ResBlock(c_in, c_in * 2, stride=stride, bias=bias, norm_cls=norm_cls, act_module=act_module)
            blocks = nb * [ResBlock(c_in * 2, c_in * 2, stride=1, bias=bias, norm_cls=norm_cls, act_module=act_module)]
            res_blocks.append(nn.Sequential(block1, *blocks))
        res_blocks = nn.Sequential(*res_blocks)
        final_pool = AdaptiveConcatPool3d() if cat_pool else nn.AdaptiveAvgPool3d(1)
        n_flatten = 2 ** len(n_blocks) * c_start * 2 if cat_pool else 2 ** len(n_blocks) * c_start
        head = ClassifierHead(nodes=[n_flatten, n_out], drops=[p_dropout], act_module=act_module)
        super().__init__(conv1, pool1, nn.ReLU(), norm1, res_blocks, final_pool, nn.Flatten(), head)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, bias=False, norm_cls=NORM_CLS, act_module=ACT_MODULE):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, stride=stride, bias=bias),
                                   act_module, norm_cls(c_out),
                                   nn.Conv3d(c_out, c_out, 3, padding=1, bias=bias), norm_cls(c_out))
        if stride > 1 or (c_in != c_out):
            self.down = nn.Sequential(nn.Conv3d(c_in, c_out, 1, stride=stride, bias=bias), norm_cls(c_out))
        else:
            self.down = None
        self.act = act_module

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.act(self.convs(x) + res)


class ClassifierHead(nn.Sequential):
    def __init__(self, nodes, drops, act_module=ACT_MODULE):
        modules = []
        activs = [act_module] * (len(nodes) - 2) + [None]
        for n_in, n_out, p, actn in zip(nodes[:-1], nodes[1:], drops, activs):
                modules.append(nn.BatchNorm1d(n_in))
                if p != 0:
                    modules.append(nn.Dropout(p))
                modules.append(nn.Linear(n_in, n_out))
                if actn is not None:
                    modules.append(actn)
        super().__init__(*modules)


class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool3d(sz), nn.AdaptiveMaxPool3d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)
