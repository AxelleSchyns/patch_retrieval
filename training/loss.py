import torch
import numpy as np
import torch.nn.functional as F
import sklearn
import sklearn.preprocessing
from pytorch_metric_learning import losses
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

# https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/efddbf23ccbe267f055867b4e1c7c6693e2447c9/batchminer/rho_distance.py#L13
def distanceweightedsampling(batch, labels, lower_cutoff=0.5, upper_cutoff=1.4, contrastive_p=0.2):
    """
    This methods finds all available triplets in a batch given by the classes provided in labels, and select
    triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.

    Randomly swaps a positive sample with the negative sample for regularization.

    Args:
        batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
        labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        lower_cutoff: float, lower cutoff value for negatives that are too close to anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
        upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
    Returns:
        list of sampled data tuples containing reference indices to the position IN THE BATCH.
    """
    if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
    bs = batch.shape[0]
    distances    = pdist(batch.detach()).clamp(min=lower_cutoff) # Compute all pairwise distances of the samples 

    positives, negatives = [],[]
    labels_visited = []
    anchors = []
    # Browse over the images of the batches to select a positive and negative other sample
    for i in range(bs):
        # Retrieve in neg all the images of the batch with a different label than i and in pos the ones with same label
        neg = labels!=labels[i]; pos = labels==labels[i]

        use_contr = np.random.choice(2, p=[1-contrastive_p, contrastive_p]) # randomly decides if it will use a constrastive or distance_weighted triplet
        if np.sum(pos)>1: # If there are other images of same labels in batch
            anchors.append(i)
            if use_contr:
                positives.append(i)
                pos[i] = 0
                negatives.append(np.random.choice(np.where(pos)[0]))
            else:
                q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i])
                pos[i] = 0
                p = np.random.choice(2, p=[0.99,0.01])
                if p==1:
                    negatives.append(np.random.choice(np.where(pos)[0]))
                    positives.append(np.random.choice(bs,p=q_d_inv))
                else:
                    positives.append(np.random.choice(np.where(pos)[0]))
                    negatives.append(np.random.choice(bs,p=q_d_inv))

    sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]
    return sampled_triplets

# https://github.com/Confusezius/Deep-Metric-Learning-Baselines/blob/60772745e28bc90077831bb4c9f07a233e602797/losses.py#L263
def pdist(A):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()

# https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/efddbf23ccbe267f055867b4e1c7c6693e2447c9/batchminer/rho_distance.py#L50
def inverse_sphere_distances(batch, dist, labels, anchor_label):
        """
        Function to utilise the distances of batch samples to compute their
        probability of occurence, and using the inverse to sample actual negatives to the resp. anchor.

        Args:
            batch:        torch.Tensor(), batch for which the sampling probabilities w.r.t to the anchor are computed. Used only to extract the shape.
            dist:         torch.Tensor(), computed distances between anchor to all batch samples.
            labels:       np.ndarray, labels for each sample for which distances were computed in dist.
            anchor_label: float, anchor label
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        bs,dim       = len(dist),batch.shape[-1]

        #negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
        #Set sampling probabilities of positives to zero
        log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

        q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
        #Set sampling probabilities of positives to zero
        q_d_inv[np.where(labels==anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

        #Normalize inverted distance for probability distr.
        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()

# https://github.com/Confusezius/Deep-Metric-Learning-Baselines/blob/60772745e28bc90077831bb4c9f07a233e602797/losses.py#L416
class MarginLoss(torch.nn.Module):
    def __init__(self, device, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance'):
        """
        Basic Margin Loss as proposed in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            margin:          float, fixed triplet margin (see also TripletLoss).
            nu:              float, regularisation weight for beta. Zero by default (in literature as well).
            beta:            float, initial value for trainable class margins. Set to default literature value.
            n_classes:       int, number of target class. Required because it dictates the number of trainable class margins.
            beta_constant:   bool, set to True if betas should not be trained.
            sampling_method: str, sampling method to use to generate training triplets.
        Returns:
            Nothing!
        """
        super(MarginLoss, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta_val = beta
        self.beta     = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampling_method    = sampling_method
        self.sampler            = distanceweightedsampling
        self.device             = device


    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            margin loss (torch.Tensor(), batch-averaged)
        """

        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        sampled_triplets = self.sampler(batch, labels)

        #Compute distances between anchor-positive and anchor-negative.
        d_ap, d_an = [],[]
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap).to(device=self.device), torch.stack(d_an).to(device=self.device)

        #Group betas together by anchor class in sampled triplets (as each beta belongs to one class).
        if self.beta_constant:
            beta = self.beta.to(device = self.device)
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]].to(device=self.device) for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)
        #Compute actual margin postive and margin negative loss
        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        #Compute normalization constant
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        #Actual Margin Loss
        loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count

        #(Optional) Add regularization penalty on betas.
        if self.nu:
            beta_regularization_loss = torch.sum(beta)
            loss = loss + beta_regularization_loss.type(torch.cuda.FloatTensor)

        return loss

# https://github.com/euwern/proxynca_pp/blob/e2fd551d90ba62f62c722782234f03fabda50320/loss.py#L52
class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, device):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8).to(device=device) ## define the proxy as learnable parameters
        self.scale = scale
        self.device = device

    def forward(self, X, T):
        P = self.proxies
        #note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2

        P = self.scale * F.normalize(P, p = 2, dim = -1) 
        X = self.scale * F.normalize(X, p = 2, dim = -1)

        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = 0
        )

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss
# https://github.com/euwern/proxynca_pp/blob/e2fd551d90ba62f62c722782234f03fabda50320/loss.py#L9
def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0):
    device = T.device
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).to(device=device)

    return T

# https://github.com/euwern/proxynca_pp/blob/e2fd551d90ba62f62c722782234f03fabda50320/similarity.py#L6
def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    #print(error_mask.sum())
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


# https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/efddbf23ccbe267f055867b4e1c7c6693e2447c9/criteria/softmax.py#L12
class NormSoftmax(torch.nn.Module):
    def __init__(self, temperature, n_classes, embed_dim, loss_softmax_lr, device):
        super(NormSoftmax, self).__init__()
        self.temperature = temperature

        self.class_map = torch.nn.Parameter(torch.Tensor(n_classes, embed_dim)).to(device=device)
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv, stdv)

        self.lr = loss_softmax_lr


    def forward(self, batch, labels, **kwargs):
        class_mapped_batch = torch.nn.functional.linear(batch, torch.nn.functional.normalize(self.class_map, dim=1))

        loss = torch.nn.CrossEntropyLoss()(class_mapped_batch/self.temperature, labels.to(torch.long))

        return loss


# https://pchanda.github.io/Siamese_plots_torch/
class SimpleBCELoss(torch.nn.Module): 
    def __init__(self):
        super(SimpleBCELoss,self).__init__()
        self.bce_loss = torch.nn.BCELoss()
            
    def forward(self,output1,output2,label):
        edist = torch.nn.PairwiseDistance(p=2,keepdim=True)(output1,output2)
        edist = torch.sigmoid(edist)
        edist = edist.squeeze(1)
        # change dtype of label to float
        label = label.type(torch.FloatTensor)
        loss_bce = self.bce_loss(edist,label)
        return loss_bce

    
# https://pchanda.github.io/Siamese_plots_torch/
class ContrastiveLoss(torch.nn.Module): 
    """
    Contrastive loss function 
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label = 0 if positive pair, 1 otherwise
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

          
        return loss_contrastive
    
class InfoNCE(torch.nn.Module):
    def __init__(self, contrastive, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.loss = losses.NTXentLoss(temperature=temperature)
        self.contrastive = contrastive
    
    def forward(self, output1, output2, out3):
        if self.contrastive:
            embeds = torch.zeros([3 * output1.shape[0], output1.shape[1]])
            labels = torch.zeros([3 * output1.shape[0]])
            for i in range(len(output1)):
                embeds[3*i] = output1[i]
                embeds[3*i + 1] = output2[i]
                embeds[3*i + 2] = out3[i]
                labels[3*i] = i
                labels[3*i + 1] = i
                labels[3*i + 2] = -i

            loss = self.loss(embeds, labels)
            return loss
        else:
            loss = losses.SelfSupervisedLoss(self.loss)(output1, output2)
            return loss


# Implementation of SoftTriple Loss - https://github.com/idstcv/SoftTriple/tree/master
class SoftTriple(nn.Module):
    def __init__(self, device, la=20, gamma=0.1, tau=0.2, margin=0.01, dim=128, cN=67, K=10): # K = number of centers , cN = number of classes 
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K)).to(device=device)
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).to(device=device)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify