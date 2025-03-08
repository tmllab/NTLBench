import torch
import torch.nn.functional as F
# from bypass_bn import disable_running_stats, enable_running_stats
from abc import ABC, abstractmethod


def cal_stats(out):
    mu = torch.mean(out, dim=0)
    logvar = torch.log(torch.var(out, dim=0))
    return mu, logvar


def compute_rec(reconstructed_inputs, original_inputs):
    return F.mse_loss(reconstructed_inputs, original_inputs, reduction="mean")


def cal_entropy(out):
    # entropy
    return - ((out.softmax(1) * F.log_softmax(out, 1)).sum(1)).mean()


def cal_vae_loss(mu, logvar):
    return - 0.01 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def cal_content_classification(content_pred, class_label, mask=None):
    if mask:
        raise Exception
        content_cls_loss = (F.cross_entropy(content_pred, class_label, reduction="none", ignore_index=-1) * mask.float()).mean()
    else:
        content_cls_loss = F.cross_entropy(content_pred, class_label)
    
    return content_cls_loss


def cal_style_classification(style_pred, aug_label):
    style_cls_loss = F.cross_entropy(style_pred, aug_label)
    return style_cls_loss


def cal_reconstruction(args, rec_content, rec_style, rec_target):
    content_rec_loss = args.rec_coef * (compute_rec(rec_content, rec_target))
    style_rec_loss = args.rec_coef * (compute_rec(rec_style, rec_target))
    return content_rec_loss, style_rec_loss

    
def cal_ssl(args, logits_all, mask_probs):
    logits_w, logits_s = logits_all.chunk(2)
    pseudo_label = torch.softmax(logits_w.detach()/args.T, dim=-1)
    max_probs, max_target = torch.max(pseudo_label, dim=-1)
    fix_mask = max_probs.ge(args.threshold).float()
    ssl_consistency_loss = (F.cross_entropy(logits_s, max_target, reduction='none') * fix_mask).mean()
    mask_probs.update(fix_mask.mean().item())

    return ssl_consistency_loss


def eval_disentangle(config, model_c, model_s, valloaders):
    metric_content_2_label_src = TopkAccuracy((1,))
    metric_content_2_label_tgt = TopkAccuracy((1,))
    metric_content_2_domain = TopkAccuracy((1,))
    metric_style_2_domain = TopkAccuracy((1,))
    metric_style_2_label_src = TopkAccuracy((1,))
    metric_style_2_label_tgt = TopkAccuracy((1,))
    valloader_src = valloaders[0]
    valloader_tgt = valloaders[1]
    model_c.eval()
    model_s.eval()
    with torch.no_grad():
        # source
        for i, (inputs, targets) in enumerate(valloader_src):
            if inputs.shape[1] == 1: 
                inputs = torch.repeat_interleave(inputs, repeats=3, dim=1)
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            targets_domain = torch.zeros([targets.shape[0]]).to(config.device)
            # targets_domain += 1
            
            outputs_c = model_c.forward_full(inputs)
            outputs_s = model_s.forward_full(inputs)

            content_disent = model_s.disentangle(outputs_c['f_rp'])
            style_disent = model_c.disentangle(outputs_s['f_rp'])

            metric_content_2_label_src.update(outputs_c['pred'], targets)
            metric_style_2_label_src.update(style_disent, targets)

            metric_style_2_domain.update(outputs_s['pred'], targets_domain)
            metric_content_2_domain.update(content_disent, targets_domain)
            
        # target 
        for i, (inputs, targets) in enumerate(valloader_tgt):
            if inputs.shape[1] == 1: 
                inputs = torch.repeat_interleave(inputs, repeats=3, dim=1)
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            targets_domain = torch.ones([targets.shape[0]]).to(config.device)
            
            outputs_c = model_c.forward_full(inputs)
            outputs_s = model_s.forward_full(inputs)

            content_disent = model_s.disentangle(outputs_c['f_rp'])
            style_disent = model_c.disentangle(outputs_s['f_rp'])

            metric_content_2_label_tgt.update(outputs_c['pred'], targets)
            metric_style_2_label_tgt.update(style_disent, targets)

            metric_style_2_domain.update(outputs_s['pred'], targets_domain)
            metric_content_2_domain.update(content_disent, targets_domain)

    return {'content_2_label_src': metric_content_2_label_src.get_results(),
            'content_2_label_tgt': metric_content_2_label_tgt.get_results(),
            'content_2_domain': metric_content_2_domain.get_results(),
            'style_2_domain': metric_style_2_domain.get_results(),
            'style_2_label_src': metric_style_2_label_src.get_results(),
            'style_2_label_tgt': metric_style_2_label_tgt.get_results(), }


def test_disentangle_KD(config, model_c, model_s, model_student, valloaders):
    metric_pred_label_src = TopkAccuracy((1,))
    metric_pred_label_tgt = TopkAccuracy((1,))
    metric_pred_domain = TopkAccuracy((1,))
    valloader_src = valloaders[0]
    valloader_tgt = valloaders[1]
    model_c.eval()
    model_s.eval()
    model_student.eval()
    with torch.no_grad():
        # source test samples
        for i, (inputs, targets) in enumerate(valloader_src):
            if inputs.shape[1] == 1: 
                inputs = torch.repeat_interleave(inputs, repeats=3, dim=1)
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            targets_domain = torch.zeros([targets.shape[0]]).to(config.device)
            # targets_domain += 1
            
            outputs = model_student(inputs)
            
            # # features KD
            # outputs = outputs.view(outputs.shape[0], -1)
            # content_pred = model_c.pred(outputs)
            # style_pred = model_s.pred(outputs)
            # metric_pred_label_src.update(content_pred, targets)
            # metric_pred_domain.update(style_pred, targets_domain)

            # pred KD
            metric_pred_label_src.update(outputs, targets)
            
        # target test samples
        for i, (inputs, targets) in enumerate(valloader_tgt):
            if inputs.shape[1] == 1: 
                inputs = torch.repeat_interleave(inputs, repeats=3, dim=1)
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            targets_domain = torch.ones([targets.shape[0]]).to(config.device)
            
            outputs = model_student(inputs)

            # # features KD
            # outputs = outputs.view(outputs.shape[0], -1)
            # content_pred = model_c.pred(outputs)
            # style_pred = model_s.pred(outputs)
            # metric_pred_label_tgt.update(content_pred, targets)
            # metric_pred_domain.update(style_pred, targets_domain)

            # pred KD
            metric_pred_label_tgt.update(outputs, targets)

    return {'pred_label_src': metric_pred_label_src.get_results(),
            'pred_label_tgt': metric_pred_label_tgt.get_results(),
            # 'pred_domain': metric_pred_domain.get_results()
            }



class Metric(ABC):
    @abstractmethod
    def update(self, pred, target):
        """ Overridden by subclasses """
        raise NotImplementedError()
    
    @abstractmethod
    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class Accuracy(Metric):
    def __init__(self):
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs = outputs.max(1)[1]
        self._correct += ( outputs.view(-1)==targets.view(-1) ).sum()
        self._cnt += torch.numel( targets )

    def get_results(self):
        return (self._correct / self._cnt * 100.).detach().cpu()
    
    def reset(self):
        self._correct = self._cnt = 0.0


class TopkAccuracy(Metric):
    def __init__(self, topk=(1, 5)):
        self._topk = topk
        self.reset()
    
    @torch.no_grad()
    def update(self, outputs, targets):
        if len(targets.shape) == 3:
            targets = torch.argmax(targets.squeeze(dim=1), dim=1)
            
        for k in self._topk:
            _, topk_outputs = outputs.topk(k, dim=1, largest=True, sorted=True)
            correct = topk_outputs.eq( targets.view(-1, 1).expand_as(topk_outputs) )
            self._correct[k] += correct[:, :k].view(-1).float().sum(0).item()
        self._cnt += len(targets)

    def get_results(self):
        return tuple( self._correct[k] / self._cnt * 100. for k in self._topk )

    def reset(self):
        self._correct = {k: 0 for k in self._topk}
        self._cnt = 0.0