import torch
import torch.nn as nn

class Orth_Loss(nn.Module):
    def __init__(self):
        super(Orth_Loss,self).__init__()

    def forward(self,p_img,p_txt,w_shr):
        # p_img = torch.transpose(p_img)
        # p_txt = torch.transpose(p_txt)
        w_shr = torch.transpose(w_shr,0,1)
        loss = torch.norm(torch.mm(p_img,w_shr))+torch.norm(torch.mm(p_txt,w_shr))
        return loss
