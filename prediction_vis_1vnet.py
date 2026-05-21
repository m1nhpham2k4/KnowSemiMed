import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from Model.model_1vnet import KnowSAM
from utils.utils import eval

import matplotlib.pyplot as plt


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run KnowSAM inference and save visualization results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, default="./ColonDB", help="Dataset root path.")
    parser.add_argument("--dataset", type=str, default="/ColonDB_30", help="Relative dataset path.")
    parser.add_argument("--test_datasets", nargs="+", default=[ "test_CVC-ColonDB"], help="Test splits to evaluate.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of model input channels.")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size.")
    parser.add_argument(
        "--SGDL_model_path",
        "--model_path",
        dest="model_path",
        type=str,
        default="./Results/ColonDB_30_1VNet/SGDL_best_model.pth",
        help="Path to the trained KnowSAM checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, for example 'cuda' or 'cpu'.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary foreground mask extraction.")
    parser.add_argument("--largest_cc", action="store_true", help="Keep only the largest connected foreground component.")
    parser.add_argument("--save_vis", dest="save_vis", action="store_true", help="Save visualization outputs.")
    parser.add_argument("--no_save_vis", dest="save_vis", action="store_false", help="Disable visualization saving.")
    parser.set_defaults(save_vis=True)
    parser.add_argument("--show_vis", action="store_true", help="Display each visualization figure during inference.")
    parser.add_argument("--vis_dir", type=str, default="./Results/ColonDB_30_1VNet/prediction_vis", help="Directory to store visualization outputs.")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="Overlay alpha for segmentation blending.")
    parser.add_argument(
        "--max_vis",
        type=int,
        default=20,
        help="Maximum number of samples to visualize per dataset. Use a negative value to save all samples.",
    )
    return parser


def resolve_device(device_name):
    if "cuda" in device_name and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def get_largest_connected_component(segmentation):
    if segmentation.ndim != 3:
        raise ValueError(f"Expected segmentation with shape [B, H, W], got {tuple(segmentation.shape)}")

    components = []
    for batch_index in range(segmentation.shape[0]):
        binary_mask = segmentation[batch_index].detach().cpu().numpy().astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            components.append(binary_mask)
            continue

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = (labels == largest_label).astype(np.uint8)
        components.append(largest_component)

    stacked = np.stack(components, axis=0)
    return torch.from_numpy(stacked).to(segmentation.device)


def to_hwc_uint8(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[..., None]

    image = np.nan_to_num(image)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def tensor_to_label_map(label_tensor):
    if isinstance(label_tensor, torch.Tensor):
        label_tensor = label_tensor.detach().cpu().numpy()

    label_tensor = np.asarray(label_tensor)
    if label_tensor.ndim == 4:
        label_tensor = label_tensor[0]
    if label_tensor.ndim == 3:
        if label_tensor.shape[0] == 1:
            label_tensor = label_tensor[0]
        else:
            label_tensor = np.argmax(label_tensor, axis=0)

    label_map = np.rint(label_tensor).astype(np.uint8)
    return label_map


def probability_to_prediction_map(probabilities, threshold):
    if probabilities.shape[1] == 1:
        prediction = (probabilities[:, 0] > threshold).to(torch.uint8)
    else:
        prediction = torch.argmax(probabilities, dim=1).to(torch.uint8)
    return prediction


def build_palette(num_classes):
    base_palette = np.array(
        [
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [0, 255, 255],
            [255, 255, 0],
            [255, 0, 255],
        ],
        dtype=np.uint8,
    )

    if num_classes <= len(base_palette):
        return base_palette[:num_classes]

    extra_colors = []
    for class_index in range(len(base_palette), num_classes):
        extra_colors.append(
            [
                (37 * class_index) % 255,
                (97 * class_index) % 255,
                (17 * class_index) % 255,
            ]
        )
    return np.concatenate([base_palette, np.array(extra_colors, dtype=np.uint8)], axis=0)


def colorize_label_map(label_map, palette):
    safe_label_map = np.clip(label_map, 0, len(palette) - 1)
    return palette[safe_label_map]


def label_to_grayscale(label_map, num_classes):
    if num_classes <= 2:
        return (label_map > 0).astype(np.uint8) * 255

    scale = 255 / max(1, num_classes - 1)
    return np.clip(label_map.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def overlay_segmentation(original_bgr, segmentation_bgr, label_map, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = ((1.0 - alpha) * original_bgr.astype(np.float32) + alpha * segmentation_bgr.astype(np.float32)).astype(np.uint8)
    overlay = original_bgr.copy()
    foreground = label_map > 0
    overlay[foreground] = blended[foreground]
    return overlay


def prepare_visualization_tensors(ori_image, gt_label, pred_label, palette, num_classes, overlay_alpha):
    original_bgr = to_hwc_uint8(ori_image)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    gt_gray = label_to_grayscale(gt_label, num_classes)
    pred_gray = label_to_grayscale(pred_label, num_classes)

    segmentation_bgr = colorize_label_map(pred_label, palette)
    segmentation_rgb = cv2.cvtColor(segmentation_bgr, cv2.COLOR_BGR2RGB)

    overlay_bgr = overlay_segmentation(original_bgr, segmentation_bgr, pred_label, overlay_alpha)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return {
        "original_bgr": original_bgr,
        "original_rgb": original_rgb,
        "gt_gray": gt_gray,
        "pred_gray": pred_gray,
        "segmentation_bgr": segmentation_bgr,
        "segmentation_rgb": segmentation_rgb,
        "overlay_bgr": overlay_bgr,
        "overlay_rgb": overlay_rgb,
    }


def save_visualization_bundle(output_root, sample_name, visuals, dataset_name, metrics, show_vis):
    output_root.mkdir(parents=True, exist_ok=True)
    component_dirs = {
        "figures": output_root / "figures",
        "original": output_root / "original",
        "gt_mask": output_root / "gt_mask",
        "pred_mask": output_root / "pred_mask",
        "segmentation": output_root / "segmentation",
        "overlay": output_root / "overlay",
    }
    for directory in component_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(component_dirs["original"] / f"{sample_name}.png"), visuals["original_bgr"])
    cv2.imwrite(str(component_dirs["gt_mask"] / f"{sample_name}.png"), visuals["gt_gray"])
    cv2.imwrite(str(component_dirs["pred_mask"] / f"{sample_name}.png"), visuals["pred_gray"])
    cv2.imwrite(str(component_dirs["segmentation"] / f"{sample_name}.png"), visuals["segmentation_bgr"])
    cv2.imwrite(str(component_dirs["overlay"] / f"{sample_name}.png"), visuals["overlay_bgr"])

    figure_path = component_dirs["figures"] / f"{sample_name}.png"
    figure, axes = plt.subplots(1, 5, figsize=(22, 5))
    figure.suptitle(
        f"{dataset_name} | {sample_name} | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | HD95: {metrics['hd95']:.4f}",
        fontsize=12,
    )

    panels = [
        (visuals["original_rgb"], "Original Image", None),
        (visuals["gt_gray"], "Ground Truth Mask", "gray"),
        (visuals["pred_gray"], "Predicted Mask", "gray"),
        (visuals["segmentation_rgb"], "Segmentation Map", None),
        (visuals["overlay_rgb"], "Segmentation Overlay", None),
    ]

    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")

    if show_vis:
        plt.show(block=False)
        plt.pause(0.001)

    plt.close(figure)


def build_sample_name(sample_path, sample_index):
    sample_stem = Path(sample_path).stem
    return f"{sample_index:04d}_{sample_stem}"


def run_inference(args):
    device = resolve_device(args.device)
    data_transforms = build_transforms(args)
    palette = build_palette(args.num_classes)
    save_root = Path(args.vis_dir)
    data_root = args.data_path + args.dataset

    model = KnowSAM(args, bilinear=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    for test_dataset_name in args.test_datasets:
        test_dataset = build_Dataset(
            args,
            data_dir=data_root,
            split=test_dataset_name,
            transform=data_transforms["valid_test"],
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

        avg_dice_list = []
        avg_hd95_list = []
        avg_iou_list = []
        saved_visualizations = 0

        with torch.inference_mode():
            for batch_index, sampled_batch in enumerate(test_loader):
                test_image = sampled_batch["image"].to(device, non_blocking=True)
                test_label = sampled_batch["label"].to(device, non_blocking=True)
                ori_image = sampled_batch["ori_image"]

                pred_VNet1, pred_VNet1_soft = model(test_image)
                fusion_map = pred_VNet1
                fusion_map_soft = torch.softmax(fusion_map, dim=1)
                pred_label_tensor = probability_to_prediction_map(fusion_map_soft, args.threshold)

                if args.largest_cc:
                    pred_label_tensor = get_largest_connected_component((pred_label_tensor > 0).to(torch.uint8))

                pred_eval_tensor = pred_label_tensor.unsqueeze(1).to(torch.float32)
                eval_list = eval(test_label, pred_eval_tensor, thr=args.threshold)

                avg_dice_list.append(eval_list[0])
                avg_iou_list.append(eval_list[1])
                avg_hd95_list.append(eval_list[2])

                should_save = args.save_vis and (args.max_vis < 0 or saved_visualizations < args.max_vis)
                if not should_save:
                    continue

                gt_label_map = tensor_to_label_map(test_label)
                pred_label_map = tensor_to_label_map(pred_label_tensor)
                visuals = prepare_visualization_tensors(
                    ori_image=ori_image,
                    gt_label=gt_label_map,
                    pred_label=pred_label_map,
                    palette=palette,
                    num_classes=args.num_classes,
                    overlay_alpha=args.overlay_alpha,
                )

                sample_path = test_dataset.sample_list[batch_index]
                sample_name = build_sample_name(sample_path, batch_index)
                dataset_output_root = save_root / test_dataset_name
                save_visualization_bundle(
                    output_root=dataset_output_root,
                    sample_name=sample_name,
                    visuals=visuals,
                    dataset_name=test_dataset_name,
                    metrics={"dice": eval_list[0], "iou": eval_list[1], "hd95": eval_list[2]},
                    show_vis=args.show_vis,
                )
                saved_visualizations += 1

        avg_dice = float(np.mean(avg_dice_list)) if avg_dice_list else 0.0
        avg_hd95 = float(np.mean(avg_hd95_list)) if avg_hd95_list else 0.0
        avg_iou = float(np.mean(avg_iou_list)) if avg_iou_list else 0.0

        print(f"{test_dataset_name} :")
        print(f"avg_dice: {avg_dice:.6f}")
        print(f"avg_iou: {avg_iou:.6f}")
        print(f"avg_hd95: {avg_hd95:.6f}")
        if args.save_vis:
            print(f"saved_visualizations: {saved_visualizations}")
            print(f"visualization_dir: {save_root / test_dataset_name}")


if __name__ == "__main__":
    parser = build_parser()
    run_inference(parser.parse_args())