import torch
import torch.nn as nn
import numpy as np

from analytic.ResultCollector import ResultCollector
from modules import init_orthogonal
from modules.encoders.EncoderAtari import ST_DIMEncoderAtari


class SPModelAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, action_dim, config):
        super(SPModelAtari, self).__init__()

        self.input_shape = input_shape
        self.encoder = ST_DIMEncoderAtari(input_shape, feature_dim, config)

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        init_orthogonal(self.forward_model[0], np.sqrt(2))
        init_orthogonal(self.forward_model[2], np.sqrt(2))
        init_orthogonal(self.forward_model[4], np.sqrt(2))

    def forward(self, state, action):
        encoded_state = self.encoder(state)
        predicted_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            predicted_state = self(state, action)
            target = self.encoder(next_state)
            error = torch.mean(torch.pow(predicted_state.view(predicted_state.shape[0], -1) - target.view(next_state.shape[0], -1), 2), dim=1).unsqueeze(1)

        return error

    def loss_function(self, state, action, next_state):
        loss_target, loss_target_norm = self.encoder.loss_function_crossentropy(state, next_state)
        loss_target_norm *= 1e-4

        predicted_state = self(state, action)
        # detached version
        # target = self.encoder(next_state).detach()

        # not detached
        target = self.encoder(next_state)
        fwd_loss = nn.functional.mse_loss(predicted_state, target)

        loss = loss_target + loss_target_norm + fwd_loss
        ResultCollector().update(loss_prediction=loss.unsqueeze(-1).detach().cpu(),
                                 loss_target=loss_target.unsqueeze(-1).detach().cpu(),
                                 loss_target_norm=loss_target_norm.unsqueeze(-1).detach().cpu(),
                                 loss_fwd=fwd_loss.unsqueeze(-1).detach().cpu())
        return loss


class ICMModelAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, action_dim, config):
        super(ICMModelAtari, self).__init__()

        # calar that weighs the inverse model loss against the forward model loss
        self.scaling_factor = 0.2

        # encoder
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 128 * (self.input_width // 8) * (self.input_height // 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        init_orthogonal(self.encoder[0], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[2], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[4], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[6], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[9], nn.init.calculate_gain('relu'))

        # dopredny model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        init_orthogonal(self.forward_model[0], np.sqrt(2))
        init_orthogonal(self.forward_model[2], np.sqrt(2))
        init_orthogonal(self.forward_model[4], np.sqrt(2))

        # inverzny model
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim)
        )

        init_orthogonal(self.inverse_model[0], np.sqrt(2))
        init_orthogonal(self.inverse_model[2], np.sqrt(2))
        init_orthogonal(self.inverse_model[4], np.sqrt(2))

    #
    # def forward(self, state, action, next_state):
    #     encoded_state = self.encoder(state)
    #     encoded_next_state = self.encoder(next_state)
    #     predicted_action = self.inverse_model(torch.cat((encoded_state, encoded_next_state), dim=1))
    #     predicted_next_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
    #     return predicted_next_state, predicted_action

    def error(self, state, action, next_state):
        with torch.no_grad():
            encoded_state = self.encoder(state)
            encoded_next_state = self.encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
            error = torch.mean(torch.pow(predicted_next_state.view(predicted_next_state.shape[0], -1) - encoded_next_state.view(encoded_next_state.shape[0], -1), 2), dim=1).unsqueeze(1)
        return error

    def loss_function(self, state, action, next_state):
        encoded_state = self.encoder(state)
        encoded_next_state = self.encoder(next_state)
        predicted_action = self.inverse_model(torch.cat((encoded_state, encoded_next_state), dim=1))
        # loss na predikovanu akciu
        inverse_loss = nn.functional.mse_loss(predicted_action, action.detach())

        predicted_next_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
        # loss na predikovany dalsi stav
        forward_loss = nn.functional.mse_loss(predicted_next_state, encoded_next_state.detach())

        loss = (1 - self.scaling_factor) * inverse_loss + self.scaling_factor * forward_loss

        ResultCollector().update(loss=loss.unsqueeze(-1).detach().cpu(),
                                 inverse_loss=inverse_loss.unsqueeze(-1).detach().cpu(),
                                 forward_loss=forward_loss.unsqueeze(-1).detach().cpu())

        return loss
