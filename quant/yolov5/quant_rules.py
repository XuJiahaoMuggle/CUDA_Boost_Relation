from typing import *
from onnx.onnx_pb import ModelProto

import os
import torch
import onnx

import quant_helper


"""
Custom rules to handle concat and maxpool node.
"""
def find_all_nodes_with_input(model: ModelProto, input_name: str) -> List[str]:
    """find_all_nodes_with_input
    Brief:
        Find all nodes whose input is input_name.
    Args:
        model - ModelProto
        input_name - str
    """
    all = []
    for node in model.graph.node:
        if input_name in node.input:
            all.append(node)
    return all


def find_one_node_with_input(model: ModelProto, input_name: str):
    """find_one_node_with_input
    Brief:
        Find one node whose input is input_name
    Args:
        model - ModelProto
        input_name - str
    """
    for node in model.graph.node:
        if input_name in node.input:
            return node


def find_conv_node_with_qnode(model: ModelProto, qnode):
    """find_conv_node_with_qnode
    Brief:
        Find conv node with its upper quantization node.
    Args:
        model - ModelProto
        qnode
    """
    dqnode = find_one_node_with_input(model, qnode.output[0])
    conv_node = find_one_node_with_input(model, dqnode.output[0])
    return conv_node
    

def find_one_node_with_output(model: ModelProto, output_name: str):
    """find_one_node_with_output
    Brief:
        Find one node whose output is output_name.
    Args:
        model - ModelProto
        output_name - str
    """
    for node in model.graph.node:
        if output_name in node.output:
            return node


def find_conv_name_with_weight(model: ModelProto, weight_name: str) -> str:
    """find_conv_name_with_weight
    Brief:
        Find conv's name with its weight.
    Args:
        model - ModelProto
        weight_name - str
    """
    dqnode = find_one_node_with_output(model, weight_name)
    qnode = find_one_node_with_output(model, dqnode.input[0])
    return ".".join(qnode.input[0].split('.')[:-1])



def find_quant_pairs(onnx_path: str):
    assert os.path.exists(onnx_path), f"No such file called {onnx_path}"
    model = onnx.load(onnx_path)
    match_pairs = []
    for node in model.graph.node:
        if node.op_type == "Concat":
            # 找到所有以Concat节点作为输入的节点
            nodes = find_all_nodes_with_input(model, node.output[0])
            # Concat节点下面的QuantizerLinear类型节点下面的Conv节点的名字
            major = None
            # 当找到的节点的类型是QuantizeLinear类型时
            for qnode in nodes:
                if qnode.op_type != "QuantizeLinear":
                    continue
                # 找到与QuantizerLinear相关联的conv节点
                conv_node = find_conv_node_with_qnode(model, qnode)
                # 第一次找到了Concat下面的Conv节点
                if major is None:
                    major = find_conv_name_with_weight(model, conv_node.input[1])
                else:
                    match_pairs.append([major, find_conv_name_with_weight(model, conv_node.input[1])])

                for subnode in model.graph.node:
                    if subnode.op_type == "QuantizeLinear" and subnode.input[0] in node.input:
                        sub_conv_node = find_conv_node_with_qnode(model, subnode)
                        match_pairs.append([major, find_conv_name_with_weight(model, sub_conv_node.input[1])])
        elif node.op_type == "MaxPool":
            qnode = find_one_node_with_input(model, node.output[0])
            if not qnode or qnode.op_type != "QuantizeLinear":
                continue
            conv_node = find_conv_node_with_qnode(model, qnode)
            major = find_conv_name_with_weight(model, conv_node.input[1])

            same_input_nodes = find_all_nodes_with_input(model, node.input[0])
            for same_input_node in same_input_nodes:
                if same_input_node.op_type == "QuantizeLinear": 
                    conv_node = find_conv_node_with_qnode(model, same_input_node)
                    match_pairs.append([major, find_conv_name_with_weight(model, conv_node.input[1])])
    return match_pairs


def get_attr_with_path(model: torch.nn.Module, path: str):
    for each in path.split('.'):
        model = getattr(model, each)
    return model


def apply_custom_rules_to_model(model: torch.nn.Module):
    quant_helper.export_onnx_ptq(model, "tmp.onnx", device=torch.device("cpu"))
    pairs = find_quant_pairs("tmp.onnx")
    for major, sub in pairs:
        print(f"Rules: {sub} match to {major}")
        get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
    os.remove("tmp.onnx")


if __name__ == "__main__":
    onnx_file = os.path.join(os.getcwd(), "onnx_files", "yolov5n-percentile-99.99-1024.onnx")
    match_pairs = find_quant_pairs(onnx_file)
    for each in match_pairs:
        print(each)
