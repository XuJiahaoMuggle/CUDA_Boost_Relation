import yaml
import torch
import os
import argparse
from typing import *

from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from models.yolo import Model, attempt_load

import quant_helper

def prepare_dataloader(
        data_cfg="./data/coco128.yaml", 
        batch_size: int = 32, 
        type: str = "train",
        cache: str = "ram",
        workers: int = 0
): 
    with open(data_cfg, encoding="utf-8") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        data_dict = check_dataset(data_dict)
    calib_data_path = data_dict['train'] if type == "train" else data_dict["val"]
    dataloader = create_dataloader(
        path=calib_data_path,
        imgsz=640,
        batch_size=batch_size,
        stride=32,
        hyp=None,
        rect=True,
        cache=cache,
        workers=workers * 2,
        image_weights=False
    )[0]
    return dataloader


def prepare_model(
        weight: str, 
        model_cfg="./models/yolov5n.yaml", 
        ch=3, 
        nc=80, 
        device=torch.device("cpu")
):
    # model = attempt_load(weight, device=device, inplace=True, fuse=True)
    ckpt = torch.load(weight, map_location="cpu")
    model: Model = Model(model_cfg, ch=ch, nc=nc).to(device)
    state_dict = ckpt["model"].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()
    return model    


def run_yolov5_ptq(args):
    weight: str = args.weight
    model_cfg: str = args.model_cfg
    ch: int = args.ch
    nc: int = args.nc
    device = torch.device("cpu") if args.device == "cpu" else torch.device(''.join(["cuda:", args.device]))

    data_cfg: str = args.data_cfg
    batch_size: int = args.batch_size   
    cache: str = args.cache
    workers: int = args.workers 
    num_batch: int = args.num_batch
    
    model = prepare_model(weight, model_cfg, ch, nc, device)
    dataloader = prepare_dataloader(data_cfg, batch_size, "train", cache, workers)
    ignore_layers: List = args.ignore_layers

    num_batch: int = args.num_batch
    calib_method: str = args.calib_method

    save_path: str = args.save_path
    quant_helper.set_calibrator_method(use_per_channel=True, calib_method=calib_method)
    quant_helper.replace_to_quantization_module(model, ignore_layers)
    if calib_method == "histogram":
        hist_percentile: List[float] = args.hist_percentile
        for each in hist_percentile:
            quant_model = quant_helper.calibrate_model(
                model, 
                dataloader, 
                device, 
                num_batch, 
                method = "percentile",
                percentile = each
            )
            calib_output = os.path.join(save_path, f"yolov5n-percentile-{each}-{num_batch * dataloader.batch_size}.pth")
            torch.save(quant_model, calib_output)
        for each in ["mse", "entropy"]:
            quant_model = quant_helper.calibrate_model(
                model, 
                dataloader, 
                device, 
                num_batch, 
                method = each
            )
            calib_output = os.path.join(save_path, f"yolov5n-percentile-{each}-{num_batch * dataloader.batch_size}.pth")
            torch.save(quant_model, calib_output)
    else:
        quant_model = quant_helper.calibrate_model(
            model, 
            dataloader, 
            device, 
            num_batch, 
            method = "max"
        )
        calib_output = os.path.join(save_path, f"yolov5n-max-{calib_method}-{num_batch * dataloader.batch_size}.pth")
        torch.save(quant_model, calib_output)
        quant_helper.export_onnx_ptq(
            quant_model, 
            os.path.join(save_path, f"yolov5n-max-{num_batch * dataloader.batch_size}.onnx"),
            device=torch.device("cpu")
        )
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='./yolov5n.pt', help='initial weights path')
    parser.add_argument('--model_cfg', type=str, default='./models/yolov5n.yaml', help='model configure path')
    parser.add_argument('--nc', type=int, default=80, help='number of classes')
    parser.add_argument('--ch', type=int, default=3, help='number of channels')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--data_cfg', type=str, default='./data/coco128.yaml', help='data configure path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_batch', type=int, default=32, help='number of batch')
    parser.add_argument('--ignore_layers', default=['model.24.m.0', 'model.24.m.1', 'model.24.m.2'], help='the layers that skip quantization')
    parser.add_argument('--calib_method', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--hist_percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--save_path', type=str, default="./pth_files")
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_yolov5_ptq(args)