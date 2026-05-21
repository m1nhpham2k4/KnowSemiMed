import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from Model.prompt import Super_Prompt

class MaskDecoderWrapper(nn.Module):
    def __init__(self, mask_decoder, wrapper):
        super().__init__()
        self.mask_decoder = mask_decoder
        self.wrapper = wrapper

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        out = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=self.wrapper._last_high_res_features
        )
        return out[0], out[1]

class PromptEncoderWrapper(nn.Module):
    def __init__(self, prompt_encoder, image_embedding_size):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.image_embedding_size = image_embedding_size

    def forward(self, points, boxes, masks):
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.prompt_encoder.embed_dim), device=self.prompt_encoder._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self.prompt_encoder._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = boxes
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self.prompt_encoder._embed_masks(masks)
        else:
            dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        pe = self.prompt_encoder.get_dense_pe()
        return F.interpolate(pe, size=self.image_embedding_size, mode="bilinear", align_corners=False)

    def _get_batch_size(self, points, boxes, masks):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

class SAM2Wrapper(nn.Module):
    def __init__(self, args, sam2_model):
        super().__init__()
        self.sam2 = sam2_model
        self.super_prompt = Super_Prompt(in_chns=args.in_channels, class_num=args.num_classes, point_nums=args.point_nums, box_nums=args.box_nums)
        self._last_high_res_features = None
        vit_patch_size = 16
        image_embedding_size = args.image_size // vit_patch_size
        self.image_embedding_size = (image_embedding_size, image_embedding_size)

    @property
    def prompt_encoder(self):
        return PromptEncoderWrapper(self.sam2.sam_prompt_encoder, self.image_embedding_size)

    @property
    def mask_decoder(self):
        return MaskDecoderWrapper(self.sam2.sam_mask_decoder, self)

    def image_encoder(self, x):
        backbone_out = self.sam2.forward_image(x)
        self._last_high_res_features = backbone_out["backbone_fpn"][0:2]
        return backbone_out["vision_features"]
