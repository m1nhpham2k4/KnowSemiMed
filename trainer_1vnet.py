import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.sam.build_sam import sam_model_registry
import torch.optim as optim
from utils.losses import dice_loss, KDLoss, DiceLoss
import logging
from utils.utils import dice_coef

import numpy as np

from Model.model_1vnet import KnowSAM

ce_loss = torch.nn.CrossEntropyLoss()

GPUdevice = torch.device('cuda', 0)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.criterion_mse = nn.MSELoss()
        self.KDLoss = KDLoss(T=10)
        self.dice_loss = DiceLoss(args.num_classes)

        self.sam_model = sam_model_registry[args.model_type](args, checkpoint=args.sam_checkpoint).to(args.device).train()
        self.SGDL = KnowSAM(args).cuda().train()

        self.optimizer_sam = optim.Adam(self.sam_model.parameters(), lr=args.lr)
        self.optimizer_SGDL = torch.optim.SGD(self.SGDL.parameters(), lr=args.VNet1_lr, momentum=0.9,
                                              weight_decay=0.0001)

        self.best_performance_sam = 0.0
        self.best_performance_SGDL = 0.0

        for n, value in self.sam_model.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            elif "super_prompt" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def entropy_loss(self, p, C=2):
        # p N*C*W*H*D
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
             torch.tensor(np.log(C)).cuda()
        ent = torch.mean(y1)
        return ent

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def mix_up(self, fusion_map_soft, volume_batch, pseudo_label, labeled_label, consistency_weight, patch_size=4,
               top_k=5):
        unlabel_pseudo_label = torch.argmax(pseudo_label.clone(), dim=1)
        entropy_unlab = self.get_entropy_map(fusion_map_soft[self.args.labeled_bs:])
        entropy_lab = self.get_entropy_map(fusion_map_soft[:self.args.labeled_bs])
        pooling = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        entropy_unlab = pooling(entropy_unlab).view(self.args.labeled_bs, -1)
        entropy_lab = pooling(entropy_lab).view(self.args.labeled_bs, -1)

        _, min_indices_flat = torch.topk(entropy_unlab, top_k, largest=True)
        min_indices_2d = torch.stack([min_indices_flat // patch_size, min_indices_flat % patch_size], dim=-1)
        _, min_indices_flat_lab = torch.topk(entropy_lab, top_k, largest=True)
        min_indices_2d_lab = torch.stack([min_indices_flat_lab // patch_size, min_indices_flat_lab % patch_size],
                                         dim=-1)

        labeled_volume_batch = volume_batch[:self.args.labeled_bs]
        unlabeled_volume_batch = volume_batch[self.args.labeled_bs:]

        unlabeled_volume_batch_mix = torch.zeros_like(unlabeled_volume_batch).cuda()
        unlabel_pseudo_label_mix = torch.zeros_like(unlabel_pseudo_label).cuda()
        labeled_volume_batch_mix = torch.zeros_like(labeled_volume_batch).cuda()
        labeled_pseudo_label_mix = torch.zeros_like(labeled_label).cuda()

        patch_h = int(self.args.image_size / patch_size)
        for b in range(self.args.labeled_bs):
            index = min_indices_2d[b]
            img_mask = torch.zeros((self.args.image_size, self.args.image_size)).cuda()
            index_lab = min_indices_2d_lab[b]
            img_mask_lab = torch.zeros((self.args.image_size, self.args.image_size)).cuda()
            for n in index:
                img_mask[n[0] * patch_h: (n[0] + 1) * patch_h, n[1] * patch_h: (n[1] + 1) * patch_h] = 1
            for n in index_lab:
                img_mask_lab[n[0] * patch_h: (n[0] + 1) * patch_h, n[1] * patch_h: (n[1] + 1) * patch_h] = 1

            unlabeled_volume_batch_mix[b] = labeled_volume_batch[b] * img_mask + unlabeled_volume_batch[b] * (1 - img_mask)
            unlabel_pseudo_label_mix[b] = labeled_label[b] * img_mask + unlabel_pseudo_label[b] * (1 - img_mask)

            labeled_volume_batch_mix[b] = unlabeled_volume_batch[b] * img_mask_lab + labeled_volume_batch[b] * (1 - img_mask_lab)
            labeled_pseudo_label_mix[b] = unlabel_pseudo_label[b] * img_mask_lab + labeled_label[b] * (1 - img_mask_lab)

        volume_batch_mix = torch.cat([labeled_volume_batch_mix, unlabeled_volume_batch_mix], dim=0)
        label_batch_mix = torch.cat([labeled_pseudo_label_mix, unlabel_pseudo_label_mix], dim=0)

        pred_VNet1_mix, pred_VNet1_soft_mix = self.SGDL(volume_batch_mix)

        pseudo_label_mix = torch.argmax(pred_VNet1_mix, dim=1)

        VNet1_sup_mixed_loss = ce_loss(pred_VNet1_mix, label_batch_mix.long()) + self.dice_loss(pred_VNet1_soft_mix, label_batch_mix)
        VNet1_enp_mixed_loss = self.entropy_loss(pred_VNet1_soft_mix, C=2)
        VNet1_unsup_mixed_loss = ce_loss(pred_VNet1_mix[self.args.labeled_bs:], pseudo_label_mix[self.args.labeled_bs:].long()) + self.dice_loss(pred_VNet1_soft_mix[self.args.labeled_bs:], pseudo_label_mix[self.args.labeled_bs:])

        VNet1_mixed_loss = VNet1_sup_mixed_loss + 0.9 * VNet1_enp_mixed_loss + consistency_weight * VNet1_unsup_mixed_loss

        return VNet1_mixed_loss

    def train(self, volume_batch, label_batch, iter_num):
        image_embeddings = self.sam_model.image_encoder(volume_batch)
        pred_VNet1, pred_VNet1_soft = self.SGDL(volume_batch)

        points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)
        low_res_masks_all = torch.empty((self.args.batch_size, 0, int(self.args.image_size/4), int(self.args.image_size/4)), device=self.args.device)

        for i in range(self.args.num_classes):
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,
                boxes=boxes_embedding[i],
                masks=F.interpolate(pred_VNet1[:, i, ...].unsqueeze(1).clone().detach(), size=(64, 64), mode='bilinear')
            )

            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.args.multimask,
            )

            low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)

        pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size), mode="bilinear", align_corners=False)
        pred_sam_soft = torch.softmax(pred_sam, dim=1)

        VNet1_sup_loss = ce_loss(pred_VNet1[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_VNet1_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        VNet1_enp_loss = self.entropy_loss(pred_VNet1_soft, C=2)
        VNet1_kd_loss = self.KDLoss(pred_VNet1.permute(0, 2, 3, 1).reshape(-1, 2), pred_sam.clone().detach().permute(0, 2, 3, 1).reshape(-1, 2))

        sam_sup_loss = ce_loss(pred_sam[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_sam_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])

        consistency_weight = self.get_current_consistency_weight(iter_num // int(self.args.max_iterations/self.args.consistency_rampup)) * 10

        VNet1_loss = VNet1_sup_loss + VNet1_kd_loss + 0.9 * VNet1_enp_loss

        if iter_num > self.args.mixed_iterations:
            VNet1_mixed_loss = self.mix_up(pred_VNet1_soft, volume_batch, pred_sam_soft[self.args.labeled_bs:], label_batch[:self.args.labeled_bs], consistency_weight)
            SGDL_loss = VNet1_loss + VNet1_mixed_loss
        else:
            SGDL_loss = VNet1_loss

        sam_loss = sam_sup_loss

        self.optimizer_sam.zero_grad()
        self.optimizer_SGDL.zero_grad()

        sam_loss.backward()
        SGDL_loss.backward()

        self.optimizer_sam.step()
        self.optimizer_SGDL.step()

        lr_ = self.args.lr * (1.0 - iter_num / self.args.max_iterations)
        VNet1_lr_ = self.args.VNet1_lr * (1.0 - iter_num / self.args.max_iterations)

        for param_group in self.optimizer_sam.param_groups:
            param_group['lr'] = lr_
        for param_group in self.optimizer_SGDL.param_groups:
            param_group['lr'] = VNet1_lr_

        logging.info('iteration %d : '
                    '  sam_loss : %f'
                    '  sam_lr_ : %10f'
                    '  SGDL_loss : %f'
                    '  VNet1_lr_ : %10f'
                    % (iter_num,
                        sam_loss.item(),
                        lr_,
                        SGDL_loss.item(),
                        VNet1_lr_))

        return {
            "sam_loss": sam_loss.item(),
            "SGDL_loss": SGDL_loss.item(),
            "VNet1_loss": VNet1_loss.item(),
            "VNet1_sup_loss": VNet1_sup_loss.item(),
            "VNet1_enp_loss": VNet1_enp_loss.item(),
            "VNet1_kd_loss": VNet1_kd_loss.item(),
            "sam_sup_loss": sam_sup_loss.item(),
            "consistency_weight": consistency_weight,
            "sam_lr": lr_,
            "SGDL_lr": VNet1_lr_
        }

    def val(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.SGDL.eval()

        avg_dice_sam = 0.0
        avg_dice_SGDL = 0.0

        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            image_embeddings = self.sam_model.image_encoder(val_image)
            pred_VNet1, pred_VNet1_soft = self.SGDL(val_image)

            points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)

            low_res_masks_all = torch.empty(
                (1, 0, int(self.args.image_size / 4), int(self.args.image_size / 4)),
                device=self.args.device)
            with torch.no_grad():
                for i in range(self.args.num_classes):
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=None,
                        boxes=boxes_embedding[i],
                        masks=F.interpolate(pred_VNet1[:, i, ...].unsqueeze(1).clone().detach(), size=(64, 64), mode='bilinear')
                    )
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.args.multimask,
                    )
                    low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
            pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size))
            pred_sam_soft = torch.softmax(pred_sam, dim=1)
            dice_sam = dice_coef(val_label, pred_sam_soft, thr=0.5)
            avg_dice_sam += dice_sam

            dice_SGDL = dice_coef(val_label, pred_VNet1_soft, thr=0.5)
            avg_dice_SGDL += dice_SGDL

        avg_dice_sam = avg_dice_sam / len(val_loader)
        avg_dice_SGDL = avg_dice_SGDL / len(val_loader)

        logging.info('iteration %d : '
                     '  sam_mean_dice : %f '
                     '  SGDL_mean_dice : %f '
                    % (iter_num, avg_dice_sam, avg_dice_SGDL))

        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            save_best_sam = os.path.join(snapshot_path, 'sam_best_model.pth')
            torch.save(self.sam_model.state_dict(), save_best_sam)

        if avg_dice_SGDL > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_SGDL
            save_best_SGDL = os.path.join(snapshot_path, 'SGDL_best_model.pth')
            torch.save(self.SGDL.state_dict(), save_best_SGDL)
        self.sam_model.train()
        self.SGDL.train()
        return {
            "sam_mean_dice": float(avg_dice_sam),
            "SGDL_mean_dice": float(avg_dice_SGDL)
        }