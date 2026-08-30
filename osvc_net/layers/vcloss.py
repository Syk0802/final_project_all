import torch
import torch.nn as nn
import torch.nn.functional as F


def build_vcloss(cfg):
    vc_type = getattr(cfg.SOLVER, 'VC_TYPE', 'l2')
    num = getattr(cfg.SOLVER, 'VC_NUM', 2)
    in_dim = getattr(cfg.SOLVER, 'VC_IN_DIM', 2048)
    temperature = getattr(cfg.SOLVER, 'VC_TEMPERATURE', 0.07)
    eps = getattr(cfg.SOLVER, 'VC_EPS', 1e-4)

    if vc_type == 'l2':
        return VCLoss_L2(num=num, in_dim=in_dim)
    elif vc_type == 'ntxent':
        return VCLoss_NTXent(num=num, in_dim=in_dim, temperature=temperature)
    elif vc_type == 'cosine':
        return VCLoss_Cosine(num=num, in_dim=in_dim, eps=eps)
    elif vc_type == 'kl':
        return VCLoss_KL(num=num, in_dim=in_dim)
    else:
        raise ValueError(f"Unknown VC_TYPE: {vc_type}")


class VCLoss_L2(nn.Module):
    """原始版本: L2 距离最小化"""

    def __init__(self, num=2, in_dim=2048):
        super(VCLoss_L2, self).__init__()
        self.num = num
        self.fc1 = nn.Sequential(nn.Linear(in_dim, 256, bias=False))
        self.fc2 = nn.Sequential(nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 256), nn.BatchNorm1d(256))

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.normalize(x)
        loss = 0
        num = int(x.size(0) / self.num)
        for i in range(self.num):
            for j in range(self.num):
                if i < j:
                    loss += ((x[i * num:(i + 1) * num, :] - x[j * num:(j + 1) * num, :]).norm(dim=1, keepdim=True)).mean()
        return loss


class VCLoss_NTXent(nn.Module):
    """对比学习风格: NT-Xent / InfoNCE loss"""

    def __init__(self, num=2, in_dim=2048, temperature=0.07):
        super(VCLoss_NTXent, self).__init__()
        self.num = num
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128)
        )

    def forward(self, x):
        z = self.projector(x)
        z = F.normalize(z, dim=1)

        N = int(z.size(0) / self.num)

        loss = 0.0
        n_pairs = 0
        for i in range(self.num):
            for j in range(i + 1, self.num):
                z_i = z[i * N:(i + 1) * N]
                z_j = z[j * N:(j + 1) * N]

                logits = torch.mm(z_i, z_j.t()) / self.temperature
                labels = torch.arange(N, device=z.device)

                loss += F.cross_entropy(logits, labels)
                loss += F.cross_entropy(logits.t(), labels)
                n_pairs += 1

        return loss / n_pairs


class VCLoss_Cosine(nn.Module):
    """余弦一致性 + 方差正则: 防止特征塌缩"""

    def __init__(self, num=2, in_dim=2048, eps=1e-4):
        super(VCLoss_Cosine, self).__init__()
        self.num = num
        self.eps = eps
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128)
        )

    def forward(self, x):
        z = self.projector(x)
        z = F.normalize(z, dim=1)

        num = int(z.size(0) / self.num)

        loss_consistency = 0
        for i in range(self.num):
            for j in range(i + 1, self.num):
                sim = (z[i * num:(i + 1) * num] * z[j * num:(j + 1) * num]).sum(dim=1)
                loss_consistency += (1 - sim).mean()

        std_z = torch.sqrt(z.var(dim=0) + self.eps)
        loss_var = F.relu(1.0 - std_z).mean()

        return loss_consistency + 0.1 * loss_var


class VCLoss_KL(nn.Module):
    """对称 KL 散度: 在 softmax 分布上的对齐"""

    def __init__(self, num=2, in_dim=2048):
        super(VCLoss_KL, self).__init__()
        self.num = num
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=False)
        )

    def forward(self, x):
        z = self.projector(x)
        z = F.normalize(z, dim=1)
        num = int(z.size(0) / self.num)

        loss = 0
        n_pairs = 0
        for i in range(self.num):
            for j in range(i + 1, self.num):
                z_i = z[i * num:(i + 1) * num]
                z_j = z[j * num:(j + 1) * num]

                log_sm_i = F.log_softmax(torch.mm(z_i, z_j.t()), dim=1)
                log_sm_j = F.log_softmax(torch.mm(z_j, z_i.t()), dim=1)
                sm_j = F.softmax(torch.mm(z_j, z_i.t()), dim=1)
                sm_i = F.softmax(torch.mm(z_i, z_j.t()), dim=1)

                loss += (sm_j * (log_sm_j - log_sm_i.t())).sum(dim=1).mean()
                loss += (sm_i * (log_sm_i - log_sm_j.t())).sum(dim=1).mean()
                n_pairs += 1

        return loss / n_pairs


class VCLoss(VCLoss_L2):
    """保持向后兼容的别名"""
    pass
