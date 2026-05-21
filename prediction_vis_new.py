import os
import cv2
import torch
import argparse
import torch.nn.functional as F
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import eval
from Model.model import KnowSAM
from PIL import Image, ImageDraw, ImageFont

from skimage.measure import label


# =========================================================
# Device
# =========================================================

def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    return torch.device(device_arg)


# =========================================================
# Visualization giống file 2VNet bạn gửi
# =========================================================

def mask_to_rgb(mask, color):
    """
    Chuyển mask nhị phân sang ảnh RGB.
    """
    mask = mask > 0
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = color
    return rgb

def make_overlay(image, gt_mask, pred_mask, alpha=0.55):
    """
    Overlay:
    - Ground truth: xanh lá cây
    - Prediction: đỏ
    """
    overlay = image.copy()

    gt_mask = gt_mask > 0
    pred_mask = pred_mask > 0

    # Ground truth màu xanh lá
    overlay[gt_mask] = (
        0.45 * overlay[gt_mask] + np.array([0, 255, 0]) * alpha
    ).astype(np.uint8)

    # Prediction màu đỏ
    overlay[pred_mask] = (
        0.45 * overlay[pred_mask] + np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)

    return overlay

def make_row_arrays(image, gt_mask, pred_mask):
    """
    Tạo 4 cột:
    1. Image
    2. Ground Truth: xanh lá cây
    3. Prediction: đỏ
    4. Overlay: Image + GT xanh lá + Prediction đỏ
    """
    gt_mask = gt_mask > 0
    pred_mask = pred_mask > 0

    gt_green = mask_to_rgb(gt_mask, (0, 255, 0))
    pred_red = mask_to_rgb(pred_mask, (255, 0, 0))
    overlay = make_overlay(image, gt_mask, pred_mask)

    return [image, gt_green, pred_red, overlay]


def draw_text_center(draw, box, text, font, fill=(0, 0, 0)):
    x1, y1, x2, y2 = box

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = draw.textsize(text, font=font)

    x = x1 + (x2 - x1 - text_w) // 2
    y = y1 + (y2 - y1 - text_h) // 2

    draw.text((x, y), text, font=font, fill=fill)

def save_single_visualization(row_images, save_path, image_size=256):
    """
    Lưu từng sample riêng thành 1 hàng x 4 cột.
    """
    titles = ["Image", "Ground truth", "Prediction", "Overlay"]

    cell_w = image_size
    cell_h = image_size
    title_h = 45
    gap = 8
    border = 12

    rows = 1
    cols = 4

    width = border * 2 + cols * cell_w + (cols - 1) * gap
    height = border * 2 + title_h + gap + rows * cell_h

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for col, title in enumerate(titles):
        x1 = border + col * (cell_w + gap)
        y1 = border
        x2 = x1 + cell_w
        y2 = y1 + title_h
        draw_text_center(draw, (x1, y1, x2, y2), title, font)

    y = border + title_h + gap

    for col_idx, arr in enumerate(row_images):
        x = border + col_idx * (cell_w + gap)

        img = Image.fromarray(arr).convert("RGB")
        img = img.resize((cell_w, cell_h))
        canvas.paste(img, (x, y))

    canvas.save(save_path)
def save_report_grid(samples, save_path, image_size=256):
    """
    Lưu hình tổng hợp cho báo cáo:
    Image | Ground truth | Prediction | Overlay
    """
    if len(samples) == 0:
        print("No samples to save report grid.")
        return

    titles = ["Image", "Ground truth", "Prediction", "Overlay"]

    cell_w = image_size
    cell_h = image_size
    title_h = 45
    gap = 8
    border = 12

    rows = len(samples)
    cols = 4

    width = border * 2 + cols * cell_w + (cols - 1) * gap
    height = border * 2 + title_h + gap + rows * cell_h + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for col, title in enumerate(titles):
        x1 = border + col * (cell_w + gap)
        y1 = border
        x2 = x1 + cell_w
        y2 = y1 + title_h
        draw_text_center(draw, (x1, y1, x2, y2), title, font)

    for row_idx, row_images in enumerate(samples):
        y = border + title_h + gap + row_idx * (cell_h + gap)

        for col_idx, arr in enumerate(row_images):
            x = border + col_idx * (cell_w + gap)

            img = Image.fromarray(arr).convert("RGB")
            img = img.resize((cell_w, cell_h))
            canvas.paste(img, (x, y))

    canvas.save(save_path)


def prepare_ori_image(ori_image):
    """
    Chuyển ori_image từ tensor sang ảnh RGB uint8.
    Giữ logic giống code cũ: nếu ảnh đang là BGR thì đổi sang RGB.
    """
    img_np = ori_image[0].permute(1, 2, 0).detach().cpu().numpy()

    if img_np.max() <= 1.0:
        img_np = img_np * 255.0

    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    if img_np.ndim == 2:
        img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
    elif img_np.shape[-1] == 1:
        img_rgb = np.repeat(img_np, 3, axis=-1)
    else:
        # Giữ giống code cũ: BGR -> RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img_rgb


# =========================================================
# Model helper
# =========================================================

def get_entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map



# =========================================================
# Main
# =========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        default='./ColonDB',
        help='Dataset root path'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='/ColonDB_5',
        help='Dataset folder'
    )

    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='output channel of network'
    )

    parser.add_argument(
        '--in_channels',
        type=int,
        default=3,
        help='input channel of network'
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='patch size of network input'
    )

    parser.add_argument(
        '--point_nums',
        type=int,
        default=10,
        help='points number'
    )

    parser.add_argument(
        '--box_nums',
        type=int,
        default=1,
        help='boxes number'
    )

    parser.add_argument(
        '--mod',
        type=str,
        default='sam_adpt',
        help='mod type: seg, cls, val_ad'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='vit_b',
        help='sam model_type'
    )

    parser.add_argument(
        '--thd',
        type=bool,
        default=False,
        help='3d or not'
    )

    parser.add_argument(
        '--SGDL_model_path',
        type=str,
        default='./Results_20/ColonDB_10_MedSAM/SGDL_best_model.pth',
        help='model weight path'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./visualization_results_ColonDB',
        help='directory to save visualizations'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='device: auto, cuda, mps, cpu'
    )

    args = parser.parse_args()

    device = get_device(args.device)
    args.device = str(device)

    print(f"Using device: {device}")

    bilinear = False
    Largest = False

    data_transforms = build_transforms(args)

    test_dataset_list = ["test_CVC-ColonDB"]

    os.makedirs(args.save_dir, exist_ok=True)

    for test_dataset_name in test_dataset_list:
        test_dataset = build_Dataset(
            args,
            data_dir=args.data_path + args.dataset,
            split=test_dataset_name,
            transform=data_transforms["valid_test"]
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )

        SGDL_model = KnowSAM(args, bilinear=bilinear).to(device).train()

        SGDL_checkpoint = torch.load(
            args.SGDL_model_path,
            map_location=device
        )

        SGDL_model.load_state_dict(SGDL_checkpoint)
        SGDL_model.eval()

        avg_dice_list = []
        avg_hd95_list = []
        avg_iou_list = []

        report_samples = []

        print(f"Start processing and visualizing {test_dataset_name}...")

        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(test_loader):
                test_image = sampled_batch["image"].to(device)
                test_label = sampled_batch["label"].to(device)
                ori_image = sampled_batch["ori_image"].to(device)

                pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL_model(test_image)

                fusion_map_soft = torch.softmax(fusion_map, dim=1)

                eval_list = eval(test_label, fusion_map_soft, thr=0.5)

                avg_dice_list.append(eval_list[0])
                avg_iou_list.append(eval_list[1])
                avg_hd95_list.append(eval_list[2])

                # =================================================
                # Visualization theo style file 2VNet
                # =================================================

                # Prediction mask
                if fusion_map_soft.shape[1] > 1:
                    pred_mask = torch.argmax(
                        fusion_map_soft,
                        dim=1
                    ).squeeze(0).detach().cpu().numpy()
                else:
                    pred_mask = (
                        fusion_map_soft > 0.5
                    ).squeeze(0).squeeze(0).detach().cpu().numpy()

                pred_mask = pred_mask > 0

                # Ground truth mask
                gt_mask = test_label.squeeze(0).detach().cpu().numpy()

                if len(gt_mask.shape) == 3:
                    gt_mask = gt_mask[0]

                gt_mask = gt_mask > 0

                # Original image
                img_rgb = prepare_ori_image(ori_image)

                # Tạo 5 cột:
                # Image | Ground Truth | Prediction | Map | Overlay
                row_images = make_row_arrays(
                    img_rgb,
                    gt_mask,
                    pred_mask
                )

                # Lưu từng sample riêng
                single_save_path = os.path.join(
                    args.save_dir,
                    f"{test_dataset_name}_sample_{i_batch}.png"
                )

                save_single_visualization(
                    row_images,
                    single_save_path,
                    image_size=args.image_size
                )

                # Gom lại để tạo hình tổng hợp báo cáo
                report_samples.append(row_images)

        # Lưu hình tổng hợp cho báo cáo
        report_save_path = os.path.join(
            args.save_dir,
            f"{test_dataset_name}_report_visualization_grid.png"
        )

        save_report_grid(
            report_samples,
            report_save_path,
            image_size=args.image_size
        )

        avg_dice = np.mean(avg_dice_list)
        avg_iou = np.mean(avg_iou_list)
        avg_hd95 = np.mean(avg_hd95_list)

        print(test_dataset_name, " :")
        print("avg_dice: ", avg_dice)
        print("avg_iou: ", avg_iou)
        print("avg_hd95: ", avg_hd95)

        print(f"Saved individual visualizations to: {args.save_dir}")
        print(f"Saved report visualization grid to: {report_save_path}")