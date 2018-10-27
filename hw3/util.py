import torch

def log_sum_exp(score):
    maxscore = torch.max(score, -1)[0] # [C]
    return maxscore + torch.log(torch.sum(torch.exp(score - maxscore.unsqueeze(-1)), -1))