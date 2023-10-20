import json

from tqdm import tqdm
from typing import *

import torch
import onnx
import onnxsim

import pytorch_quantization.nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib as quant_calib
from absl import logging as quant_logging


def set_calibrate_method(use_per_channel: bool = False, calib_method: str = "histogram") -> None:
    """set_calibrate_method
    Brief:
        For all the layer's input, use per-tensor quantization, and the calibrator method of input and weight are the same. 
    Args:
        use_per_channel: for each layer's weight use per-channel quantization.
        calib_method: the calibrator method.
    """
    # For input use per-tensor scale.
    quant_desc_input = tensor_quant.QuantDescriptor(calib_method = calib_method, axis = None)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    if not use_per_channel:
        # For weight use per-tensor scale too.
        quant_desc_weight = tensor_quant.QuantDescriptor(calib_method = calib_method, axis = None)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
    quant_logging.set_verbosity(quant_logging.ERROR)


def replace_to_quantization_module(model: torch.nn.Module, ignore: List[str] = None) -> None:
    """replace_to_quantization_module
    Brief:
        Replace torch.nn block to quant_nn block.
    Args:
        model - torch.nn.Module: the model to be replaced.
        ignore - List[str]: the layers do not implement quantization.
    """
    replace_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        nn_hash = id(getattr(entry.orig_mod, entry.mod_name))
        replace_dict[nn_hash] = entry.replace_mod
    dfs(model, ignore, replace_dict, "")


def dfs(model: torch.nn.Module, ignore: List[str], replace_dict: Dict, record: str = ""):
    """dfs
    Brief: 
        Deep first search to replace all the layers that can be replaced.
    Args: 
        model - torch.nn.Module: the model to be replaced.
        ignore - List[str]: the layers do not implement quantization.
        replace_dict - Dict: key: the hash value of torch.nn.Module -> value: the block of quant_nn
        record - str: the record each layer.
    """
    for name, sub_model in model._modules.items():
        path = name if record == "" else record + '.' + name
        dfs(sub_model, ignore, replace_dict, path)
        nn_hash = id(type(sub_model))
        if nn_hash in replace_dict.keys():
            if ignore is not None and path in ignore:
                continue
            model._modules[name] = transfer_torch_to_quantizaion(sub_model, replace_dict[nn_hash])


def transfer_torch_to_quantizaion(nn_instance: torch.nn.Module, quant_module_cls: object):
    """transfer_torch_to_quantizaion
    Brief:
        Transfer the instance of torch.nn.Module to the instance of quant.nn.
    Args:
        nn_instance - torch.nn.Module
        quant_module_cls - object
    """
    quant_instance = quant_module_cls.__new__(quant_module_cls)
    for k, v in vars(nn_instance).items():
        setattr(quant_instance, k, v)

    def __init__(self):
        # input only
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)
            if isinstance(self.input_quantizer._calibrator, quant_calib.HistogramCalibrator):
                self.input_quantizer._calibrator._torch_hist = True
        # input & weight
        elif isinstance(self, quant_nn_utils.QuantMixin): 
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=False)
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self.input_quantizer._calibrator, quant_calib.HistogramCalibrator):
                self.input_quantizer._calibrator._torch_hist = True
                self.weight_quantizer._calibrator._torch_hist = True
    __init__(quant_instance)
    return quant_instance


class QuantCalibCtx():
    def __init__(self, model: torch.nn.Module):
        self.model = model
    
    def apply(self, flag: bool):
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if flag:
                    if module._calibrator is not None:
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()
                else:
                    if module._calibrator is not None:
                        module.enable_quant()
                        module.disable_calib()
                    else:
                        module.enable()

    def __enter__(self):
        self.apply(True)

    def __exit__(self, exec_type, exec_val, exec_tb):
        self.apply(False)


def quant_calib_wrapper(calib_func):
    """quant_calib_wrapper
    Brief:
        When collection data, the calib option must be on and the quant option must be off.
        After collection data, the calib option must be off and the quant option must be on.
    """
    def wrapper(*args, **kwargs):
        model = kwargs["model"] if "model" in kwargs.keys() else args[0]
        quant_calib_ctx = QuantCalibCtx(model)
        quant_calib_ctx.apply(True)
        calib_func(*args, **kwargs)
        quant_calib_ctx.apply(False)
    return wrapper


@quant_calib_wrapper
@torch.no_grad()
def collect_statics(model: torch.nn.Module, dataloader, device, num_batch=200):
    """collect_statics
    Brief: 
        Use dataloder to calibrate the quantization model.
    Args:
        model - torch.nn.Module
        dataloader
        device
        num_batch
    """
    model.to(device)
    for idx, datas in tqdm(enumerate(dataloader), total=min(len(dataloader), num_batch)):
        imgs: torch.Tensor = datas[0].to(device, non_blocking=True).float() / 255.0
        model(imgs)
        if idx > num_batch:
            break


@torch.no_grad()
def compute_amax(model: torch.nn.Module, device, **kwargs):
    """compute_amax
    Brief: 
        Compute amax after calibration.
    Args:
        model - torch.nn.Module
        device
        **kwargs
            method - str: must be one of "max", "percentile", "mse", "entropy"
            percentile - float only valie when methos is "percentile"
    """
    model.to(device)
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, quant_calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                # if module.amax is not None:
                #     module.amax.to(device)


class QuantExportOnnxCtx():
    def __init__(self):
        if quant_nn.TensorQuantizer.use_fb_fake_quant is True:
            quant_nn.TensorQuantizer.use_fb_fake_quant = False

    def apply(self, flag: bool):
        quant_nn.TensorQuantizer.use_fb_fake_quant = flag

    def __enter__(self):
        self.apply(True)

    def __exit__(self, exec_type, exec_val, exec_tb):
        self.apply(False)


def quant_export_onnx_wrapper(export_func):
    """quant_export_onnx_wrapper
    Brief:
        Set quant_nn.TensorQuantizer.use_fb_fake_quant before exporting.
        After expotring set quant_nn.TensorQuantizer.use_fb_fake_quant to Fasle.
    """
    def wrapper(*args, **kwargs):
        quant_export_onnx_ctx = QuantExportOnnxCtx()
        quant_export_onnx_ctx.apply(True)
        export_func(*args, **kwargs)
        quant_export_onnx_ctx.apply(False)
    return wrapper


@quant_export_onnx_wrapper
@torch.no_grad()
def export_onnx_ptq(model: torch.nn.Module, save_path: str, device, dynamic_batch: bool = False, simp: bool = True):
    """export_onnx_ptq
    Brief:
        Export onnx model.
    Args:
        model - torch.nn.Module
        save_path - str
        device
        dynamic_batch - bool
        simp - bool
    """
    model.eval()
    model.to(device)
    input_dummy = torch.randn(1, 3, 640, 640, device=device)
    model.model[-1].concat = True
    grid_old_func = model.model[-1]._make_grid
    model.model[-1]._make_grid = lambda *args: [torch.from_numpy(item.cpu().data.numpy()).to(item.device) for item in grid_old_func(*args)]    
    torch.onnx.export(
        model,
        args=input_dummy,
        f=save_path,
        opset_version=13,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}} if dynamic_batch else None
    )
    model.model[-1].concat = False
    model.model[-1]._make_grid = grid_old_func
    if simp:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save_model(onnx_model, save_path)


def contain_quant_layer(model: torch.nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True
    return False


class QuantDisable():
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled
    
    def __enter__(self):
        self.apply(disabled=True)  
        return self
    
    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)


class QuantEnable():
    def __init__(self, model: torch.nn.Module):     
        self.model = model
        
    def apply(self, enabled=True):  
        for name, module in self.model.named_modules():
               if isinstance(module, quant_nn.TensorQuantizer):
                   module._disabled = not enabled
    
    def __enter__(self):
        self.apply(enabled=True)
        return self
    
    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)


class SummaryTool():
    def __init__(self, file: str):
        self.file = file
        self.data = []
    
    def append(self, item):
        self.data.append(item)
    
    def write(self):
        with open(self.file, 'w') as f:
            json.dump(self.data, f, indent=4)


# TODO: This may be useless, cause the situation is not common.
def sensitive_analysis_wrapper(sensitive_func, model: torch.nn.Module):
    print("Doing sensitive analysis ...")
    if not hasattr(model, "__len__"):
        print("Only Sequential model supports sensitive analysis wrapper.")
        return sensitive_func
    
    def wrapper(*args, **kwargs):
        for i in range(0, len(model)):
            with QuantDisable(model[i]) as quant_disable:
                sensitive_func(*args, **kwargs)

    return wrapper


class QuantAdd(torch.nn.Module):
    def __init__(self, quant: bool):
        super(QuantAdd, self).__init__()
        self.quant = quant
        if quant:
            self.input_quant0 = quant_nn.TensorQuantizer(
                quant_nn_utils.QuantDescriptor(num_bits=8, calib_method="histogram")
            )
            self.input_quant0._calibrator._torch_hist = True
            self.input_quant1 = self.input_quant0
            self.use_fb_fake_quant = True
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.quant:
            return self.input_quant0(x) + self.input_quant1(y)
        return x + y


if __name__ == "__main__":
    import torchvision
    torch.manual_seed(12345)
    model = torchvision.models.resnet50()
    print(len(model))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    replace_to_quantization_module(model)
    export_onnx_ptq(model, "quant_onnx.onnx", device)
