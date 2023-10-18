import yaml
import torch
import os
from pathlib import Path
import argparse
from typing import *

from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from models.yolo import Model, attempt_load, Detect

import val
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
        rank=-1,
        pad=0.5,
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
    model = attempt_load(weight, device=device, inplace=True, fuse=True)
    # ckpt = torch.load(weight, map_location="cpu")
    # model: Model = Model(model_cfg, ch=ch, nc=nc).to(device)
    # state_dict = ckpt["model"].float().state_dict()
    # model.load_state_dict(state_dict, strict=False)
    model.float()
    model.eval()
    # with torch.no_grad():
    #     model.fuse()
    #     model.to(device)
    return model    


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, data_cfg: str, device, val_loader):
    model.to(device)
    model.eval()
    with open(data_cfg, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    results, _, _ = val.run(
        data_dict,
        batch_size=val_loader.batch_size,
        imgsz=640,
        model=model,
        iou_thres=0.65,
        single_cls=False,
        dataloader=val_loader,
        save_dir=None,
        save_json=False,
        verbose=False,
        plots=False
    )
    map50_95 = list(results)[3]
    map50 = list(results)[2]
    return map50_95, map50


@torch.no_grad()
def evaluate_pqt_models(models: List[str], data_cfg, device, val_loader, export_best: bool, export_path):
    best_model = None
    best_map = 0.
    best_model_name = ""
    for each in models:
        model = torch.load(each, map_location="cpu")
        each = Path(each).as_posix()
        map50_95, map50 = evaluate_accuracy(model, data_cfg, device, val_loader)
        print(f'{each} evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50, map50_95))
        if best_map < map50_95:
            best_map = map50_95
            best_model = model
            best_model_name = each.split('/')[-1]
            best_model_name = best_model_name[0:-4] if best_model_name.endswith(".pth") else best_model_name[0:-3]
    if export_best:
        best_model.export = True
        for k, m in best_model.named_modules():
            if isinstance(m, Detect):
                m.inplace = False
                m.export = True
        quant_helper.export_onnx_ptq(
            best_model, 
            os.path.join(export_path, "".join([best_model_name, ".onnx"])),
            device=torch.device("cpu")
        )
        best_model.export = False


@torch.no_grad()
def run_yolov5_ptq(args):
    # prepare parameters
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
    calib_method: str = args.calib_method
    save_pth_path: str = args.save_pth_path
    ignore_layers: List = args.ignore_layers
    # load model & train dataloder 
    model = prepare_model(weight, model_cfg, ch, nc, device).cuda()
    train_dataloader = prepare_dataloader(data_cfg, batch_size, "train", cache, workers)
    # initialize calibrate & insert qdq node.
    quant_helper.set_calibrate_method(use_per_channel=True, calib_method=calib_method)
    quant_helper.replace_to_quantization_module(model, ignore_layers)
    # collect statics.
    quant_helper.collect_statics(model, train_dataloader, device, num_batch)
    if calib_method == "histogram":
        hist_percentile: List[float] = args.hist_percentile
        for each in hist_percentile:
            quant_helper.compute_amax(model, device=device, method="percentile", percentile=each)
            calib_output = os.path.join(save_pth_path, f"yolov5n-percentile-{each}-{num_batch * train_dataloader.batch_size}.pth")
            torch.save(model, calib_output)
    else:  
        quant_helper.compute_amax(model, method="max")
        calib_output = os.path.join(save_pth_path, f"yolov5n-max-{num_batch * train_dataloader.batch_size}.pth")
        torch.save(model, calib_output)

    if args.evaluate:
        workspace_path = os.getcwd()
        workspace_path = os.path.join(workspace_path, "pth_files")
        assert os.path.exists(workspace_path), f"No such file or dir called {workspace_path}"
        models = [os.path.join(workspace_path, each) for each in os.listdir(workspace_path)]
        device = torch.device("cpu") if args.device == "cpu" else torch.device(''.join(["cuda:", args.device]))
        val_dataloader = prepare_dataloader(data_cfg, batch_size, "val", cache, workers)
        evaluate_pqt_models(models, data_cfg, device, val_dataloader, args.export_onnx, args.save_onnx_path)

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
    parser.add_argument('--calib_method', type=str, choices=["max", "histogram"], default="histogram")
    parser.add_argument('--hist_percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--save_pth_path', type=str, default="./pth_files")
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--export_onnx', type=bool, default=True, help='export onnx model or not"')
    parser.add_argument('--save_onnx_path', type=str, default="./onnx_files")
    parser.add_argument('--evaluate', type=bool, default=True, help="evaluate map for each model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_yolov5_ptq(args)
