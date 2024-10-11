import torch

def within_var(x: torch.Tensor):
    '''
    x: torch.Tensor, shape (N, D, S)
    '''

    N, D, S = x.shape
    if S == 1:
        return torch.zeros(1)
    x = x - x.mean(dim=2, keepdim=True)
    return ((x ** 2).sum(2) / (S))

def between_var(x: torch.Tensor):
    '''
    x: torch.Tensor, shape (N, D, S)
    '''

    N, D, S = x.shape
    m = x.mean(dim=2) #  N, D
    m = m - m.mean(dim=0, keepdim=True)
    return (m ** 2).sum(0) / (N)

def within_cov(x: torch.Tensor):

    '''
    x: torch.Tensor, shape (N, D, S)
    '''

    N, D, S = x.shape
    if S == 1:
        return torch.zeros(D, D)
    x = x - x.mean(dim=2, keepdim=True)

    # S_k = x @ x.mT # Full covariance matrix turns out to be not invertible
    S_k = torch.diag_embed((x ** 2).sum(dim=2)) # We use the variance instead
    S_W = S_k.sum(dim=0)
    return S_W

def between_cov(x: torch.Tensor):
    '''
    x: torch.Tensor, shape (N, D, S)
    '''

    N, D, S = x.shape
    m = x.mean(dim=2)
    m = m - m.mean(dim=0, keepdim=True)
    S_B = m.mT @ m
    return S_B

def cov_criterion(S_W, S_B):
    '''
    S_W: torch.Tensor, shape (D, D)
    S_B: torch.Tensor, shape (D, D)
    '''
    # crit = torch.trace(torch.inverse(S_W) @ S_B) # This is the textbook fisher criterion

    crit = torch.trace(torch.inverse(S_W + S_B) @ S_B) / S_W.shape[0]

    # crit = torch.trace(S_B / (S_W + S_B)) / S_W.shape[0] 

    return (crit, torch.trace(S_B), torch.trace(S_W))

def var_criterion(s_W, s_B):
    
    s_W = s_W.mean(0).sum()
    s_B = s_B.sum()

    crit = s_B / (s_W + s_B)
    
    return crit


def sample_align(x: torch.Tensor, estimator: str = 'var'):
    '''
    x: torch.Tensor, shape (n, d, s)
    OPTIONAL:
    y: torch.Tensor, shape (n)
    '''

    N, D, S = x.shape

    if estimator == 'cov':
        # Within sample covariance
        within = within_cov(x)

        # Between sample covariance
        between = between_cov(x)

        fld, s_b, s_w = cov_criterion(within, between)

    elif estimator == 'var':
        # Within sample variance
        within = within_var(x)
        # Between sample variance
        between = between_var(x)

        crit = var_criterion(within, between)

    return  (crit.item(), within, between)

def class_fld(x: torch.Tensor, y: torch.Tensor, estimator: str = 'var'):
    '''
    x: torch.Tensor, shape (n, d, s)
    y: torch.Tensor, shape (n)
    '''
    N, D, S = x.shape
    classes = y.unique()

    # Reshape samples into [len(classes), d, n*s]
    x_ = []
    for _, c in enumerate(classes):
        x_.append(x[y == c].permute(1, 0, 2).reshape(1, D, -1))
    x = torch.cat(x_, dim=0)

    crit, within, between = sample_align(x, estimator=estimator)

    return (crit.item(), within, between)