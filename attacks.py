import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.distributions import multivariate_normal

class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
      """
      Executes the attack on a batch of samples x. y contains the true labels 
      in case of untargeted attacks, and the target labels in case of targeted 
      attacks. The method returns the adversarially perturbed samples, which
      lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
      performs random initialization and early stopping, depending on the 
      self.rand_init and self.early_stop flags.
      """
      self.model.eval()
      self.model.requires_grad_(False)
      x_org = x.clone().detach()
      x_adv = x.clone().detach()
      y = y.clone().detach()

      if (self.rand_init):
        pertrubation = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x+pertrubation, min=0, max=1).detach()

      for i in range(self.n) :    
        x_adv.requires_grad = True
        outputs = self.model(x_adv)
        loss = self.loss_func(outputs, y).mean()
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        grad_sign = -1 if targeted else 1

        if (self.early_stop):
          top_class_pred = torch.max(outputs, dim=1)
          target_reached = top_class_pred.indices == y if targeted else top_class_pred.indices != y
          grad = (1 - target_reached.int()).reshape(-1,1,1,1) * grad
          is_done = torch.all(target_reached)
          if is_done:
            return x_adv

        x_adv = x_adv.detach() + (self.alpha * grad_sign * torch.sign(grad))
        x_adv = torch.clamp(x_adv, min=(x_org-self.eps), max=(x_org + self.eps))
        x_adv = torch.clamp(x_adv, min=0, max=1).detach_()
                  
      return x_adv


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma=sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
      """
      Executes the attack on a batch of samples x. y contains the true labels 
      in case of untargeted attacks, and the target labels in case of targeted 
      attacks. The method returns:
      1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
          and [x-eps, x+eps].
      2- A vector with dimensionality len(x) containing the number of queries for
          each sample in x.
      """
      queries_by_sample = torch.zeros(x.shape[0], device='cuda')
      self.model.eval()
      self.model.requires_grad_(False)

      x_org = x.clone().detach()
      x_adv = x.clone().detach()
      y = y.clone().detach()

      if (self.rand_init):
        pertrubation = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x+pertrubation, min=0, max=1).detach()

      for i in range(self.n) :    
        x_adv.requires_grad = False
        grad = self._estimate_gradient(x, y)
        grad_sign = -1 if targeted else 1

        if (self.early_stop):
          outputs = self.model(x_adv)
          top_class_pred = torch.max(outputs, dim=1)
          target_reached = top_class_pred.indices == y if targeted else top_class_pred.indices != y
          is_done = torch.all(target_reached)
          if is_done:
            return x_adv, queries_by_sample

          not_reached = 1 - target_reached.int()
          grad = (1 - target_reached.int()).reshape(-1,1,1,1) * grad
          queries_by_sample += (not_reached.cuda() * (2*torch.tensor(self.k, device='cuda')))

        x_adv = x_adv.detach() + (self.alpha * grad_sign * torch.sign(grad))
        x_adv = torch.clamp(x_adv, min=(x_org-self.eps), max=(x_org + self.eps))
        x_adv = torch.clamp(x_adv, min=0, max=1).detach_()
                  
      return x_adv, queries_by_sample


    def _estimate_gradient(self, x, y):
      """
      Esitmate the gradient of the conitional class probability P[y|x] given class y and image x,
      using NES as described in [Ilyas et. al. (18)]
      """
      grad = 0
      N = x.shape[1] * x.shape[2] * x.shape[3]
      dist = multivariate_normal.MultivariateNormal(loc=torch.zeros(N), covariance_matrix=torch.eye(N))
      deltas = dist.sample((self.k, x.shape[0]))
      deltas = deltas.reshape((self.k, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
      deltas = deltas.to('cuda')

      for i in range(self.k):
        eval_point = x + (torch.tensor(self.sigma, device='cuda') * deltas[i])
        outputs = self.model(eval_point)
        loss = self.loss_func(outputs, y)
        grad += loss.reshape(-1, 1, 1, 1) * deltas[i]
        #antithetic sampling
        eval_point = x - (self.sigma * deltas[i])
        outputs = self.model(eval_point)
        loss = self.loss_func(outputs, y)
        grad -= loss.reshape(-1, 1, 1, 1) * deltas[i]

      return grad / (self.k * 2 * self.sigma)


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """
    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
      """
      Executes the attack on a batch of samples x. y contains the true labels 
      in case of untargeted attacks, and the target labels in case of targeted 
      attacks. The method returns the adversarially perturbed samples, which
      lie in the ranges [0, 1] and [x-eps, x+eps].
      """

