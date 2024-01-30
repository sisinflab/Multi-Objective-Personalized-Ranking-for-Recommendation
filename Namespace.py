import torch
class Namespace:
    def __init__(self, settings, experiment):
        self.data = settings['data']
        self.device = torch.device('cuda:' + str(settings['gpu_id']) if torch.cuda.is_available() else 'cpu')
        self.backbone = settings['baseline']
        self.mo_method = settings['wrapper']
        self.mode = experiment['mode']
        self.every = settings['validation_rate']
        self.batch_size = settings['batch_size']
        self.n_epochs = settings['epochs']
        if self.backbone == 'BPRMF':
            self.dim = experiment['dim']
            self.lr = experiment['lr']
            self.weight_decay = experiment['l_2']
        elif self.backbone == 'LightGCN':
            self.dim = experiment['dim']
            self.lr = experiment['lr']
            self.weight_decay = experiment['l_2']
            self.layers = experiment['layers']
            self.normalize = experiment['normalize']

        if self.mo_method == 'PDU':
            self.atk = experiment['atk']
            self.type = experiment['g_n']
            self.ranker = experiment['ranker']
            if self.ranker == 'base':
                self.ablation = experiment['ablation']
        elif self.mo_method == 'multifr':
            self.gamma = experiment['gamma']
            self.temp = experiment['temp']
            self.type = experiment['g_n']
            self.ranker = experiment['ranker']
        elif self.mo_method == 'None':
            self.scale1 = experiment['scale']
        elif self.mo_method == 'PDU_SCALE':
            self.atk = experiment['atk']
            self.ranker = experiment['ranker']
            self.scale1 = experiment['scale']















