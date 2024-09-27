# ライブラリのインポート
from solutions.trainer import BaseTorchSolution

import time
import cv2
import numpy as np
import torch
import torch.nn as nn


class CNN(BaseTorchSolution):
    def __init__(self, device, file_name, num_robots, act_dim, feat_dim):
        super(CNN, self).__init__(device)
        self.file_name = file_name
        self.num_robots = num_robots
        self.act_dim = act_dim
        self.feat_dim = feat_dim
        self.prev_xs = [torch.zeros((self.feat_dim, self.feat_dim, 3)) for _ in range(self.num_robots)]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3 * 2,
                out_channels=16,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=act_dim),
            nn.Tanh(),
        )
        self.modules_to_learn.append(self.cnn)

    def _get_action(self, robot_index, obs):
        x = torch.tensor(obs)
        x = torch.div(x, 255)
        self.x_arg = torch.cat((x, self.prev_xs[robot_index]), dim=2).unsqueeze(0)
        self.prev_xs[robot_index] = x
        self.x_arg = self.x_arg.permute(0, 3, 1, 2)
        output = self.cnn(self.x_arg)
        action = output.squeeze(0).cpu().numpy()
        return action

    def reset(self):
        self.prev_xs = [torch.zeros((self.feat_dim, self.feat_dim, 3)) for _ in range(self.num_robots)]
    
    @staticmethod
    def to_heatmap(x):
        x_scaled = (x * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(x_scaled, cv2.COLORMAP_JET)
        return heatmap

    def show_gui(self, img, target: int):
        # Show Grad-CAM
        feat = self.cnn[:8](self.x_arg)
        feat = feat.clone().detach().requires_grad_(True)
        output = self.cnn[8:10](feat)
        output[0][target].backward()
        alpha = torch.mean(feat.grad.view(16, 6 * 6), dim=1)
        feat = feat.squeeze(0)
        L = torch.sum(feat * alpha.view(-1, 1, 1), 0).cpu().detach().numpy()
        L_min = np.min(L)
        L_max = np.max(L - L_min)
        L = (L - L_min) / L_max if L_max != 0 else L
        L = self.to_heatmap(cv2.resize(L, (400, 400)))
        img = cv2.resize(img, (400, 400))[:, :, ::-1]
        blended = img * 0.5 + L * 0.5
        cv2.imshow("render", blended.astype(np.uint8))
        cv2.waitKey(1)
