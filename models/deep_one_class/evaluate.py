from sklearn import metrics
from optim.ae_trainer import AETrainer



def Average(lst): 
    return sum(lst) / len(lst) 
def deep_one_test(trainer, dataset: Pairs_Dataset(''), net: BaseNet, threshold=None):
    scores1=[]
    logger = logging.getLogger()
    nu = 0.05
    # Set device for network
    device = 'cuda'
    net = net.to(trainer.device)        
    if threshold == None:
    threshold = get_R(trainer, dataset, net, nu)           
       
    # Get test data loader
    _, test_loader = dataset.loaders(batch_size=trainer.batch_size, num_workers=trainer.n_jobs_dataloader)

    # Testing
    logger.info('Starting testing...')
    #start_time = time.time()
    idx_label_score = []
    net.eval()
        
    all_scores=[]
    with torch.no_grad():
                 
        for data in test_loader:
            inputs, labels, idx = data
                

        inputs = inputs.to(trainer.device)
        outputs = net(inputs)
        dist = torch.sum((outputs - trainer.c) ** 2, dim=1)
        if trainer.objective == 'soft-boundary':
            scores = dist - trainer.R ** 2
        else:
            scores = dist.cpu().numpy()
            all_scores.append(scores)
        
        scores = np.concatenate(all_scores)
        scores = np.array(scores)
        scores = (scores > threshold ).astype(np.int)
        score = (scores == 0).sum()

        # Compute AUC
        labels = np.zeros_like(scores)
        trainer.test_auc = metrics.accuracy_score(labels, scores, normalize=True)
        logger.info('Test set accuracy: {:.2f}%'.format(100*trainer.test_auc))
        scores1  = (100*trainer.test_auc)
        logger.info('Finished testing.')
        return labels, scores, scores1, threshold


def pretrain(deepSVDD, dataset, optimizer_name: str = 'adam', lr: float = 1e-5, n_epochs: int = 50,
             lr_milestones: tuple = (), batch_size: int = 200, weight_decay: float = 1e-3, device: str = 'cuda',
             n_jobs_dataloader: int = 0):
    """Pretrains the weights for the Deep SVDD network \phi via autoencoder."""

    deepSVDD.ae_net = build_autoencoder(deepSVDD.net_name)
    deepSVDD.ae_optimizer_name = optimizer_name
    deepSVDD.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                batch_size=batch_size, weight_decay=weight_decay, device=device,
                                n_jobs_dataloader=n_jobs_dataloader)
    deepSVDD.ae_net = deepSVDD.ae_trainer.train(dataset, deepSVDD.ae_net)
    deepSVDD.init_network_weights_from_pretraining()

def get_R(trainer, dataset, net, nu):
    net = net.to(trainer.device)
    train_loader, _ = dataset.loaders(batch_size=trainer.batch_size, num_workers=trainer.n_jobs_dataloader)
    
    all_scores = []
    with torch.no_grad():
        for data in train_loader:
            inputs, _, _ = data
            inputs = inputs.to(trainer.device)
            outputs = net(inputs)
            dist = torch.sum((outputs - trainer.c) ** 2, dim=1)
            all_scores.append(dist.cpu().numpy())
            
    scores = np.concatenate(all_scores)
    return np.percentile(scores, 100*(1-nu))