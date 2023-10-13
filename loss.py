import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import torch.distributed as dist


class SelfDistillationAndMSELoss(nn.Module):
    def __init__(self, out_dim, teacher_temp, nepochs, warmup_teacher_temp_epochs, warmup_teacher_temp, p=0.5,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.p = p
        print('SelfDistillationAndMSELoss: ', p)
        self.mse = nn.MSELoss()
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, input, target, emb_i, emb_j, epoch):
        loss_mse = self.mse(input, target)

        student_out = emb_i / self.student_temp
        temp = self.teacher_temp_schedule[epoch]

        teacher_output = F.softmax((emb_j - self.center) / temp, dim=-1)
        loss_selfdis = torch.sum(-teacher_output * F.log_softmax(student_out, dim=-1), dim=-1).mean()

        loss = self.p * loss_selfdis + (1 - self.p) * loss_mse

        self.update_center(teacher_output)
        return loss, loss_selfdis, loss_mse

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
