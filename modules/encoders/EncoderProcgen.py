import numpy as np
import torch
import torch.nn as nn

from modules import init_orthogonal


class ST_DIM_CNN(nn.Module):

    def __init__(self, input_shape, feature_dim):
        super().__init__()
        self.feature_size = feature_dim
        self.hidden_size = self.feature_size

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 64 * (self.input_width // 8) * (self.input_height // 8)
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        # gain = nn.init.calculate_gain('relu')
        gain = 0.5
        init_orthogonal(self.main[0], gain)
        init_orthogonal(self.main[2], gain)
        init_orthogonal(self.main[4], gain)
        init_orthogonal(self.main[6], gain)
        init_orthogonal(self.main[9], gain)

        self.local_layer_depth = self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        out = self.main[6:](f5)

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }
        return out


class ST_DIMEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(ST_DIMEncoderProcgen, self).__init__()

        self.config = config
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.encoder = ST_DIM_CNN(input_shape, feature_dim)
        self.classifier1 = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(self.encoder.local_layer_depth, self.encoder.local_layer_depth)

    def forward(self, state, fmaps=False):
        return self.encoder(state, fmaps)

    def loss_function_crossentropy(self, states, next_states):
        f_t_maps, f_t_prev_maps = self.encoder(next_states, fmaps=True), self.encoder(states, fmaps=True)

        # Loss 1: Global at time t, f5 patches at time t-1
        f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
        sy = f_t_prev.size(1)
        sx = f_t_prev.size(2)
        N = f_t.size(0)

        positive = []
        for y in range(sy):
            for x in range(sx):
                positive.append(f_t_prev[:, y, x, :].T)

        predictions = self.classifier1(f_t)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.config.device).unsqueeze(0).repeat(logits.shape[0], 1)
        loss1 = nn.functional.cross_entropy(logits, target, reduction='mean')
        norm_loss1 = torch.norm(logits, p=2, dim=[1, 2]).mean()

        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        f_t = f_t_maps['f5']
        predictions = []
        positive = []
        for y in range(sy):
            for x in range(sx):
                predictions.append(self.classifier2(f_t[:, y, x, :]))
                positive.append(f_t_prev[:, y, x, :].T)

        predictions = torch.stack(predictions)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.config.device).unsqueeze(0).repeat(logits.shape[0], 1)
        loss2 = nn.functional.cross_entropy(logits, target, reduction='mean')
        norm_loss2 = torch.norm(logits, p=2, dim=[1, 2]).mean()

        loss = loss1 + loss2
        norm_loss = norm_loss1 + norm_loss2

        return loss, norm_loss

    def loss_function_cdist(self, states, next_states):
        f_t_maps, f_t_prev_maps = self.encoder(next_states, fmaps=True), self.encoder(states, fmaps=True)

        # Loss 1: Global at time t, f5 patches at time t-1
        f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
        sy = f_t_prev.size(1)
        sx = f_t_prev.size(2)

        N = f_t.size(0)
        target = torch.ones((N, N), device=self.config.device) - torch.eye(N, N, device=self.config.device)
        loss1 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier1(f_t) + 1e-8
                positive = f_t_prev[:, y, x, :] + 1e-8
                logits = torch.cdist(predictions, positive, p=0.5)
                step_loss = nn.functional.mse_loss(logits, target)
                loss1 += step_loss

        loss1 = loss1 / (sx * sy)

        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        f_t = f_t_maps['f5']
        loss2 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier2(f_t[:, y, x, :]) + 1e-8
                positive = f_t_prev[:, y, x, :] + 1e-8
                logits = torch.cdist(predictions, positive, p=0.5)
                step_loss = nn.functional.mse_loss(logits, target)
                loss2 += step_loss

        loss2 = loss2 / (sx * sy)

        loss = loss1 + loss2

        return loss


class SNDVEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(SNDVEncoderProcgen, self).__init__()

        self.config = config
        fc_size = (input_shape[1] // 8) * (input_shape[2] // 8)

        self.layers = [
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),

            nn.Flatten(),

            nn.Linear(64 * fc_size, feature_dim)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2.0 ** 0.5)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.encoder = nn.Sequential(*self.layers)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states_a, states_b, target):
        xa = states_a.clone()
        xb = states_b.clone()

        # normalise states
        # if normalise is not None:
        #     xa = normalise(xa)
        #     xb = normalise(xb)

        # states augmentation
        xa = self.augment(xa)
        xb = self.augment(xb)

        # obtain features from model
        za = self(xa)
        zb = self(xb)

        # predict close distance for similar, far distance for different states
        predicted = ((za - zb) ** 2).mean(dim=1)

        # similarity MSE loss
        loss_sim = ((target - predicted) ** 2).mean()

        # L2 magnitude regularisation
        magnitude = (za ** 2).mean() + (zb ** 2).mean()

        # care only when magnitude above 200
        loss_magnitude = torch.relu(magnitude - 200.0)

        loss = loss_sim + loss_magnitude

        return loss

    def augment(self, x):
        x = self.aug_random_apply(x, 0.5, self.aug_mask_tiles)
        x = self.aug_random_apply(x, 0.5, self.aug_noise)

        return x.detach()

    @staticmethod
    def aug_random_apply(x, p, aug_func):
        mask = (torch.rand(x.shape[0]) < p)
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        mask = mask.float().to(x.device)
        y = (1.0 - mask) * x + mask * aug_func(x)

        return y

    @staticmethod
    def aug_mask_tiles(x, p=0.1):

        if x.shape[2] == 96:
            tile_sizes = [1, 2, 4, 8, 12, 16]
        else:
            tile_sizes = [1, 2, 4, 8, 16]

        tile_size = tile_sizes[np.random.randint(len(tile_sizes))]

        size_h = x.shape[2] // tile_size
        size_w = x.shape[3] // tile_size

        mask = (torch.rand((x.shape[0], 1, size_h, size_w)) < (1.0 - p))

        mask = torch.kron(mask, torch.ones(tile_size, tile_size))

        return x * mask.float().to(x.device)

    # uniform aditional noise
    @staticmethod
    def aug_noise(x, k=0.2):
        pointwise_noise = k * (2.0 * torch.rand(x.shape, device=x.device) - 1.0)
        return x + pointwise_noise


class BarlowTwinsEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(BarlowTwinsEncoderProcgen, self).__init__()

        self.config = config
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim

        self.encoder = ST_DIM_CNN(input_shape, feature_dim)
        self.lam = 5e-3

        self.lam_mask = torch.maximum(torch.ones(self.feature_dim, self.feature_dim, device=self.config.device) * self.lam, torch.eye(self.feature_dim, self.feature_dim, device=self.config.device))

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        n = states.shape[0]
        d = self.feature_dim
        y_a = self.augment(states)
        y_b = self.augment(states)
        z_a = self.encoder(y_a)
        z_b = self.encoder(y_b)

        # z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        # z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)

        c = torch.matmul(z_a.t(), z_b) / n
        c_diff = (c - torch.eye(d, d, device=self.config.device)).pow(2) * self.lam_mask
        loss = c_diff.sum()

        return loss

    def augment(self, x):
        # ref = transforms.ToPILImage()(x[0])
        # ref.show()
        # transforms_train = torchvision.transforms.Compose([
        #     transforms.RandomResizedCrop(96, scale=(0.66, 1.0))])
        # transforms_train = transforms.RandomErasing(p=1)
        # print(x.max())
        ax = x + torch.randn_like(x) * 0.1
        ax = nn.functional.upsample(nn.functional.avg_pool2d(ax, kernel_size=2), scale_factor=2, mode='bilinear')
        # print(ax.max())

        # aug = transforms.ToPILImage()(ax[0])
        # aug.show()

        return ax


class VICRegEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(VICRegEncoderProcgen, self).__init__()

        self.config = config
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim

        self.encoder = ST_DIM_CNN(input_shape, feature_dim)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        n = states.shape[0]
        d = self.feature_dim
        # y_a = self.augment(states)
        # y_b = self.augment(states)
        z_a = self.encoder(states)
        z_b = self.encoder(next_states)

        inv_loss = nn.functional.mse_loss(z_a, z_b)

        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        var_loss = torch.mean(nn.functional.relu(1 - std_z_a)) + torch.mean(nn.functional.relu(1 - std_z_b))

        z_a = (z_a - z_a.mean(dim=0))
        z_b = (z_b - z_b.mean(dim=0))

        cov_z_a = torch.matmul(z_a.t(), z_a) / (n - 1)
        cov_z_b = torch.matmul(z_b.t(), z_b) / (n - 1)

        cov_loss = cov_z_a.masked_select(~torch.eye(self.feature_dim, dtype=torch.bool, device=self.config.device)).pow_(2).sum() / self.feature_dim + \
                   cov_z_b.masked_select(~torch.eye(self.feature_dim, dtype=torch.bool, device=self.config.device)).pow_(2).sum() / self.feature_dim

        la = 1.
        mu = 1.
        nu = 1. / 25

        return la * inv_loss + mu * var_loss + nu * cov_loss

    def augment(self, x):
        # ref = transforms.ToPILImage()(x[0])
        # ref.show()
        # transforms_train = torchvision.transforms.Compose([
        #     transforms.RandomResizedCrop(96, scale=(0.66, 1.0))])
        # transforms_train = transforms.RandomErasing(p=1)
        # print(x.max())
        ax = x + torch.randn_like(x) * 0.1
        ax = nn.functional.upsample(nn.functional.avg_pool2d(ax, kernel_size=2), scale_factor=2, mode='bilinear')
        # print(ax.max())

        # aug = transforms.ToPILImage()(ax[0])
        # aug.show()

        return ax
