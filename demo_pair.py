# This file provides pair-image inference for FlowSeek.

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append("core")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.parser import parse_args
from flowseek import FlowSeek
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return tensor[None]


def resize_pair(image1, image2, max_size):
    if max_size is None or max_size <= 0:
        return image1, image2, 1.0

    _, _, height, width = image1.shape
    scale = min(1.0, float(max_size) / float(max(height, width)))
    if scale == 1.0:
        return image1, image2, scale

    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    image1 = F.interpolate(image1, size=(new_height, new_width), mode="bilinear", align_corners=False)
    image2 = F.interpolate(image2, size=(new_height, new_width), mode="bilinear", align_corners=False)
    return image1, image2, scale


@torch.no_grad()
def run_flow(args, model, image1, image2):
    if args.input_scale != 0:
        image1 = F.interpolate(image1, scale_factor=2 ** args.input_scale, mode="bilinear", align_corners=False)
        image2 = F.interpolate(image2, scale_factor=2 ** args.input_scale, mode="bilinear", align_corners=False)

    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow = output["flow"][-1]

    if args.input_scale != 0:
        down_scale = 0.5 ** args.input_scale
        flow = F.interpolate(flow, scale_factor=down_scale, mode="bilinear", align_corners=False) * down_scale

    return flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="Path to FlowSeek config json")
    parser.add_argument("--model", required=True, type=str, help="Path to FlowSeek checkpoint")
    parser.add_argument("--image1", required=True, type=str, help="Path to the first frame")
    parser.add_argument("--image2", required=True, type=str, help="Path to the second frame")
    parser.add_argument("--output_dir", default="demo_pair_outputs", type=str, help="Directory for outputs")
    parser.add_argument("--input_scale", default=0, type=int, help="Power-of-two scale used before inference")
    parser.add_argument("--max_size", default=960, type=int, help="Resize longer side before inference; <=0 disables")
    parser.add_argument("--save_resized_inputs", action="store_true", help="Save resized inputs used by the model")

    args = parse_args(parser)

    if not torch.cuda.is_available():
        raise RuntimeError("FlowSeek currently expects CUDA because the model code calls .cuda() internally.")

    os.makedirs(args.output_dir, exist_ok=True)

    image1 = read_image(args.image1)
    image2 = read_image(args.image2)
    if image1.shape[-2:] != image2.shape[-2:]:
        raise ValueError(f"Input image sizes differ: {image1.shape[-2:]} vs {image2.shape[-2:]}")

    image1, image2, resize_scale = resize_pair(image1, image2, args.max_size)
    image1 = image1.cuda()
    image2 = image2.cuda()

    model = FlowSeek(args)
    load_ckpt(model, args.model)
    model = model.cuda().eval()

    flow = run_flow(args, model, image1, image2)
    if resize_scale != 1.0:
        flow = F.interpolate(flow, scale_factor=1.0 / resize_scale, mode="bilinear", align_corners=False) / resize_scale

    flow_np = flow[0].permute(1, 2, 0).detach().cpu().numpy()
    np.save(os.path.join(args.output_dir, "flow.npy"), flow_np)

    flow_vis = flow_to_image(flow_np, convert_to_bgr=True)
    cv2.imwrite(os.path.join(args.output_dir, "flow_vis.png"), flow_vis)

    if args.save_resized_inputs:
        image1_np = image1[0].permute(1, 2, 0).detach().cpu().numpy()
        image2_np = image2[0].permute(1, 2, 0).detach().cpu().numpy()
        cv2.imwrite(os.path.join(args.output_dir, "image1_used.png"), cv2.cvtColor(image1_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output_dir, "image2_used.png"), cv2.cvtColor(image2_np, cv2.COLOR_RGB2BGR))

    print(f"Saved flow array to {os.path.join(args.output_dir, 'flow.npy')}")
    print(f"Saved flow visualization to {os.path.join(args.output_dir, 'flow_vis.png')}")
    print(f"Flow shape: {flow_np.shape}")


if __name__ == "__main__":
    main()
