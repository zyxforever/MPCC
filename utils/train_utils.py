"""
Authors: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
class Center_MemoryBank(object):
    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.Tensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
    def reset(self):
        self.ptr = 0
    def update(self, features,targets):
        b = features.size(0)
        print("zhixing")
        
        assert(b + self.ptr <= self.n)
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach())

        self.ptr += b
    def get_label(self):
        return self.targets
    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device
    def cpu(self):
        self.to('cpu')
    def cuda(self):
        self.to('cuda:0')
    def get_feat(self):
      return self.features.cuda()
def gcc_train(cm, cfg, train_loader, model, criterion1, criterion2, optimizer, epoch, aug_feat_memory, org_feat_memory, log_output_file, only_train_pretext=True):
    
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    constrastive_losses = AverageMeter('Constrast Loss', ':.4e')
    cluster_losses = AverageMeter('Cluster Loss', ':.4e')
    consistency_losses = AverageMeter('Consist Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses, constrastive_losses, cluster_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch), output_file=log_output_file)
    model.train()
    count = torch.zeros(20)
    center = torch.zeros(20, 128).to("cuda")
    for i, batch in enumerate(train_loader):
        neighbor_top1_features = None
        neighbor_top2_features = None
        neighbor_top3_features = None
        if only_train_pretext:
            images = batch['image'].cuda(non_blocking=True)
            images_augmented = batch['augmented'].cuda(non_blocking=True)
        else:
            images = batch['image'].cuda(non_blocking=True)
            images_augmented = batch['augmented'].cuda(non_blocking=True)
            neighbor_top1 = batch['neighbor_top1'].cuda(non_blocking=True)
            neighbor_top2 = batch['neighbor_top2'].cuda(non_blocking=True)
            neighbor_top3 = batch['neighbor_top3'].cuda(non_blocking=True)
            neighbor_top1_features, neighbor_top1_cluster_outs = model(neighbor_top1)
            neighbor_top2_features, neighbor_top2_cluster_outs = model(neighbor_top2)
            neighbor_top3_features, neighbor_top3_cluster_outs = model(neighbor_top3)
            neighbor_top1_features = neighbor_top1_features * batch['neighbor_top1_weight'].unsqueeze(-1).cuda()
            neighbor_top2_features = neighbor_top2_features * batch['neighbor_top2_weight'].unsqueeze(-1).cuda()
            neighbor_top3_features = neighbor_top3_features * batch['neighbor_top3_weight'].unsqueeze(-1).cuda()
            b = batch['neighbor_top1_weight'].shape[0]
            fill_one_diag_zero = torch.ones([b, b]).fill_diagonal_(0).cuda()
            neighbor_weights = torch.cat([fill_one_diag_zero + torch.diag(batch['neighbor_top1_weight'].cuda()),
                                          fill_one_diag_zero + torch.diag(batch['neighbor_top2_weight'].cuda()),
                                          fill_one_diag_zero + torch.diag(batch['neighbor_top3_weight'].cuda())], dim=1)

        neighbors = batch['neighbor'].cuda(non_blocking=True)

        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        constrastive_features, cluster_outs = model(input_)
        #print(cluster_outs[0][:256,:].size())
        constrastive_features = constrastive_features.view(b, 2, -1)
        constrastive_features4444 = constrastive_features[:, 0]
        constrastive_features444 = constrastive_features4444.detach()
        _, yy = model(images)
        
       # anchor = constrastive_features[:, 0]
        xxx = F.normalize(cm.get_feat(), p=2, dim=1)
        sim = torch.matmul(constrastive_features444, xxx.T)
        _, indices = sim.topk(k=5, dim=1, largest=True, sorted=True)
        label_temp = cm.get_label()
        #label_temp = torch.Tensor([0,1,2,3,4,5,6,7,8,9])
        #if epoch>50:
           # print(label_temp)
        #print(label_temp.size())
        clas = label_temp.repeat(256, 1).cuda()
        bb = torch.gather(clas, 1, indices)
        lei = torch.mode(bb)[0]
        label = F.one_hot(lei.to(torch.int64), num_classes=20).float()
        label_ture = F.one_hot(targets.to(torch.int64), num_classes=20).float()

        mask = torch.matmul(label, label.T)
        #if epoch>52:
        #    for i in range(10):
        #        print(mask[i])
        softmax = nn.Softmax(dim=1)
        yyy = softmax(yy[0])
        y_ceshi = softmax(cluster_outs[0][:256,:])
        #对角线是否是1
        y_sim = torch.matmul(yyy,yyy.T)
        
        #print(mask.size())
        #print(y_sim.size())
        if not only_train_pretext:
            neighbor_topk_features = torch.cat([neighbor_top1_features, neighbor_top2_features, neighbor_top3_features], dim=0).cuda()
            constrastive_loss = criterion1(constrastive_features, neighbor_topk_features,mask,y_sim,epoch, neighbor_weights, 3)
            #constrastive_loss = criterion1(constrastive_features, neighbor_topk_features, neighbor_weights, 3)
        else:
            constrastive_loss = criterion1(constrastive_features,None,mask,y_sim,epoch,None, 0)
            #constrastive_loss = criterion1(constrastive_features,None,None, 0)

        aug_feat_memory.push(constrastive_features.clone().detach()[:, 1], batch['meta']['index'])

        if not only_train_pretext:
            neighbors_features, neighbors_output = model(neighbors)
            # Loss for every head
            total_loss, consistency_loss, entropy_loss = [], [], []
            for image_and_aug_output_subhead, neighbor_top1_cluster, neighbor_top2_cluster, neighbor_top3_cluster in \
                    zip(cluster_outs, neighbor_top1_cluster_outs, neighbor_top2_cluster_outs, neighbor_top3_cluster_outs):
                image_and_aug_output_subhead = image_and_aug_output_subhead.view(b, 2, -1)
                image_output_subhead = image_and_aug_output_subhead[:, 0]
                #aug_output_subhead = image_and_aug_output_subhead[:, 1]
                neightbor_output_subhead = neighbors_output[0]
                total_loss_, consistency_loss_, entropy_loss_ = criterion2(image_output_subhead,
                                                                           mask,y_sim,epoch,
                                                                           neightbor_output_subhead,
                                                                           #aug_output_subhead,
                                                                           )
                #获得伪标签

                #ologits = softmax(image_output_subhead)
                #y_sim = torch.matmul(ologits, ologits.T)
                like_zero = torch.zeros_like(yyy)
                like_one = torch.ones_like(yyy)
                logits = torch.where(yyy > cfg['alpha'], like_one, like_zero)
                nozero = torch.nonzero(logits)
                for k in nozero:
                    for j in range(20):
                        if k[1] == j:
                            center[j] = center[j] + constrastive_features444[k[0]]
                            count[j] = count[j] + 1
                            #print(count)
                total_loss.append(total_loss_)
                consistency_loss.append(consistency_loss_)
                entropy_loss.append(entropy_loss_)
            cluster_loss = torch.sum(torch.stack(total_loss, dim=0))
        else:
            cluster_loss = torch.tensor([0.0]).cuda()

        loss = 2.0 * constrastive_loss + cluster_loss

        losses.update(loss.item())
        constrastive_losses.update(constrastive_loss.item())
        cluster_losses.update(cluster_loss.item())
        if not only_train_pretext:
            consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
            entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            progress.display(i)
    latent_std=0.001
    cm.reset()
    if epoch>=50:
        for i in range(20):
            #print(count[i])
            center[i] = center[i] / count[i]
            #noise_lb_1 = center[i] + torch.randn_like(center[i]) * latent_std
            #noise_lb_2 = center[i] + torch.randn_like(center[i]) * latent_std
            noise_lb_1 = center[i] +  latent_std
            noise_lb_2 = center[i] +  latent_std
            temp_cent = center[i]
            temp_cent = torch.unsqueeze(temp_cent,0)
            noise_lb_1 = torch.unsqueeze(noise_lb_1,0)
            noise_lb_2 = torch.unsqueeze(noise_lb_2,0)
            cm.update(temp_cent, torch.tensor(i))
            #print('执行updata')
            cm.update(noise_lb_1, torch.tensor(i))
            cm.update(noise_lb_2, torch.tensor(i))
    
    

def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None, output_file=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch), output_file=output_file)
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % 25 == 0:
            progress.display(i)
