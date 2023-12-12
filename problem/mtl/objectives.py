import torch

def from_name(names, task_names):
    objectives = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
        'L1Regularization': L1Regularization,
        'L2Regularization': L2Regularization,
        'ddp': DDPHyperbolicTangentRelaxation,
        'deo': DEOHyperbolicTangentRelaxation,
    }

    if task_names is not None:
        return [objectives[n]("labels_{}".format(t), "logits_{}".format(t)) for n, t in zip(names, task_names)]
    else:
        return [ objectives[n]() for n in names ]


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, label_name='labels', logits_name='logits'):
        super().__init__(reduction='mean')
        self.label_name = label_name
        self.logits_name = logits_name

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        return super().__call__(logits, labels)



class BinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):

    def __init__(self, label_name='labels', logits_name='logits', pos_weight=None):
        super().__init__(reduction='mean', pos_weight=torch.Tensor([pos_weight]).cuda() if pos_weight else None)
        self.label_name = label_name
        self.logits_name = logits_name

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        if labels.dtype != torch.float:
            labels = labels.float()
        return super().__call__(logits, labels)





class MSELoss(torch.nn.MSELoss):

    def __init__(self, label_name='labels'):
        super().__init__()
        self.label_name = label_name

    def __call__(self, **kwargs):
        logits = kwargs['logits']
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        return super().__call__(logits, labels)


class L1Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=1)


class L2Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=2)


class DDPHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1):
        self.label_name = label_name
        self.logits_name = logits_name
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[sensible_attribute.bool()]
        s_positive = logits[~sensible_attribute.bool()]

        return 1 / n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(
            torch.tanh(self.c * torch.relu(s_negative))))


class DEOHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1):
        self.label_name = label_name
        self.logits_name = logits_name
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
        s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]

        return 1 / n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(
            torch.tanh(self.c * torch.relu(s_negative))))


"""
Popular problem proposed by

    Carlos Manuel Mira da Fonseca. Multiobjective genetic algorithms with 
    application to controlengineering problems.PhD thesis, University of Sheffield, 1995.

with a concave pareto front.

$ \mathcal{L}_1(\theta) = 1 - \exp{ - || \theta - 1 / \sqrt{d} || $
$ \mathcal{L}_1(\theta) = 1 - \exp{ - || \theta + 1 / \sqrt{d} || $

with $\theta \in R^d$ and $ d = 100$
"""


#
# class Fonseca1():
#
#     def f1(theta):
#         d = len(theta)
#         sum1 = autograd.numpy.sum([(theta[i] - 1.0 / autograd.numpy.sqrt(d)) ** 2 for i in range(d)])
#         f1 = 1 - autograd.numpy.exp(- sum1)
#         return f1
#
#     f1_dx = autograd.grad(f1)
#
#     def __call__(self, **kwargs):
#         return f1(kwargs['parameters'])
#
#     def gradient(self, **kwargs):
#         return f1_dx(kwargs['parameters'])
#
#
# class Fonseca2():
#
#     def f2(theta):
#         d = len(theta)
#         sum1 = autograd.numpy.sum([(theta[i] + 1.0 / autograd.numpy.sqrt(d)) ** 2 for i in range(d)])
#         f1 = 1 - autograd.numpy.exp(- sum1)
#         return f1
#
#     f2_dx = autograd.grad(f2)
#
#     def __call__(self, **kwargs):
#         return f2(kwargs['parameters'])
#
#     def gradient(self, **kwargs):
#         return f2_dx(kwargs['parameters'])





#
# if __name__ == '__main__':
#     problem = Fonseca1()
