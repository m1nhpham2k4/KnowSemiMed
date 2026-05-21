import os
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import eval
from Model.model import KnowSAM

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay a mask on an RGB image."""
    overlay = image.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask > 0, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])
    return overlay.astype(np.uint8)

def get_entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map

from skimage.measure import label
def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 2):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./PROMISE12',
                        help='Name of Experiment')

    parser.add_argument('--dataset', type=str, default='/Test',
                        help='Name of Experiment')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--image_size', type=list, default=256,
                        help='patch size of network input')
    parser.add_argument('--point_nums', type=int, default=10, help='points number')
    parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
    parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument('--thd', type=bool, default=False, help='3d or not')

    parser.add_argument('--SGDL_model_path', type=str,
                        default="./Results/promise_5_SAM2/SGDL_best_model.pth",
                        help='model weight path')

    parser.add_argument('--save_dir', type=str, default='./visualization_results', help='directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    bilinear = False
    Largest = False
    data_transforms = build_transforms(args)

    test_dataset_list = ["test_promise",]

    os.makedirs(args.save_dir, exist_ok=True)

    for test_dataset_name in test_dataset_list:
        test_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split=test_dataset_name,
                                     transform=data_transforms["valid_test"])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        SGDL_model = KnowSAM(args, bilinear=bilinear).to(args.device).train()
        SGDL_checkpoint = torch.load(args.SGDL_model_path)
        SGDL_model.load_state_dict(SGDL_checkpoint)
        SGDL_model.eval()

        avg_dice_list = []
        avg_hd95_list = []
        avg_iou_list = []
        
        print(f"Start processing and visualizing {test_dataset_name}...")
        for i_batch, sampled_batch in enumerate(test_loader):
            test_image, test_label, ori_image = sampled_batch["image"].cuda(), sampled_batch["label"].cuda(), sampled_batch["ori_image"].cuda()
            pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL_model(test_image)
            fusion_map_soft = torch.softmax(fusion_map, dim=1)

            if Largest:
                pseudo_label = torch.argmax(fusion_map_soft, dim=1)
                fusion_map_soft = get_ACDC_2DLargestCC(pseudo_label).unsqueeze(0)

            eval_list = eval(test_label, fusion_map_soft, thr=0.5)

            avg_dice_list.append(eval_list[0])
            avg_iou_list.append(eval_list[1])
            avg_hd95_list.append(eval_list[2])

            # ---------------- Visualization ---------------- #
            # Extract mask from prediction
            if fusion_map_soft.shape[1] > 1:
                pred_mask = torch.argmax(fusion_map_soft, dim=1).squeeze(0).cpu().numpy()
            else:
                pred_mask = (fusion_map_soft > 0.5).squeeze(0).squeeze(0).cpu().numpy()
                
            gt_mask = test_label.squeeze(0).cpu().numpy()
            if len(gt_mask.shape) == 3: # (1, H, W)
                gt_mask = gt_mask[0]
                
            # Extract RGB image
            img_np = ori_image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # Create overlays
            gt_overlay = overlay_mask(img_rgb, gt_mask, color=(0, 255, 0)) # Green for Ground Truth
            pred_overlay = overlay_mask(img_rgb, pred_mask, color=(255, 0, 0)) # Red for Prediction
            
            # Plotting
            plt.figure(figsize=(20, 5))
            
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(img_rgb)
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(gt_mask, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            plt.title("Prediction Mask")
            plt.imshow(pred_mask, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.title("Prediction Overlay")
            plt.imshow(pred_overlay)
            plt.axis('off')
            
            save_path = os.path.join(args.save_dir, f"{test_dataset_name}_sample_{i_batch}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            # ----------------------------------------------- #

        avg_dice = np.mean(avg_dice_list)
        avg_hd95 = np.mean(avg_hd95_list)
        avg_iou = np.mean(avg_iou_list)

        print(test_dataset_name, " :")
        print("avg_dice: ", avg_dice)
        print("avg_iou: ", avg_iou)
        print("avg_hd95: ", avg_hd95)
