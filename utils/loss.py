import torch
import torch.nn as nn
import torch.nn.functional as F

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, args, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.args = args

    def forward(self, x):
        loss_sum = 0
        loss_list = []
        for i, loss in enumerate(x):
            if self.args.auto_weight:
                loss = 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            else:
                loss = 0.5 * loss
            loss_sum += loss
            loss_list.append(loss)
        return loss_sum, loss_list

def CosineLoss(output, target, num_classes=2, ce_weight=0., label_smoothing=0.):
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    eps = label_smoothing / num_classes
    negative = eps
    positive = (1 - label_smoothing) + eps

    # output = F.softmax(torch.Tensor(output), dim=-1)
    y=torch.Tensor([1]).to(output.device)
    
    true_dist = torch.zeros_like(torch.Tensor(output))
    true_dist.fill_(negative)
    true_dist.scatter_(1, torch.Tensor(target).type(torch.int64).to(output.device).data.unsqueeze(1), positive)

    return  F.cosine_embedding_loss(output, true_dist, y, reduction='mean') # + ce_weight * ce(output, target)


class MultiTaskLoss(nn.Module):
    """Computes and combines the losses for the two tasks: CLS and PRED
    """
    
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.cls_enabled = args.cls_enabled
        self.pred_enabled = args.pred_enabled
        self.cosine_loss = CosineLoss
        
    def cls_loss(self, sample, prob):
        y = sample['y']
        loss = self.cosine_loss(prob, y)
        
        return loss
        
    def pred_loss(self, sample, logit): # Survival loss
        y_seq = torch.Tensor(sample["y_seq"]).to(self.args.device).float()
        y_mask = torch.Tensor(sample["y_mask"]).to(self.args.device).float()
        loss = F.binary_cross_entropy_with_logits(logit, y_seq, weight=y_mask, reduction='sum') / torch.sum(y_mask)
              
        return loss
    
    def calculate_total_loss(self, cls_loss, pred_loss):
        weighted_cls_loss, weighted_pred_loss = None, None
        if (cls_loss is not None) and (pred_loss is not None) :        
            awl = AutomaticWeightedLoss(self.args)
            loss, (weighted_cls_loss, weighted_pred_loss) = awl([cls_loss, pred_loss])
            
        else:
            if self.cls_enabled:
                weighted_cls_loss = cls_loss
            
            if self.pred_enabled:
                weighted_pred_loss = pred_loss
            
        return weighted_cls_loss, weighted_pred_loss
    
    def forward(self, cls_output_dict, pred_output_dict, cls_sample, pred_sample):
        
        if self.cls_enabled:
            if isinstance(cls_output_dict, list):
                cls_loss = 0.
                for s, o in zip(cls_sample, cls_output_dict):
                    cls_loss += self.cls_loss(s, o)
                cls_loss /= len(cls_output_dict)
            else:
                cls_loss = self.cls_loss(cls_sample, cls_output_dict)
        else:
            cls_loss = None
        
        if self.pred_enabled:
            if isinstance(pred_output_dict, list):
                pred_loss = 0.
                for s, o in zip(pred_sample, pred_output_dict):
                    pred_loss += self.pred_loss(s, o) 
                pred_loss /= len(pred_output_dict)
            else:
                pred_loss = self.pred_loss(pred_sample, pred_output_dict)
        else: 
            pred_loss = None
        
        weighted_cls_loss, weigted_pred_loss = self.calculate_total_loss(cls_loss, pred_loss)
        
        return weighted_cls_loss, weigted_pred_loss