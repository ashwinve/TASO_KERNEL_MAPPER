from .core import *
import onnx
from onnx import helper, TensorProto, numpy_helper, mapping
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data

import numpy as np

import onnxruntime as ort
from onnx import helper, shape_inference

import sys

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

# correspond to https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
def onnx_datatype_tostring(dtype):
    if dtype == 0:
        return 'UNDEFINED'
    elif dtype == 1:
        return 'FLOAT'
    elif dtype == 2:
        return 'UINT8'
    elif dtype == 3:
        return 'INT8'
    elif dtype == 4:
        return 'UINT16'
    elif dtype == 5:
        return 'INT16'
    elif dtype == 6:
        return 'INT32'
    elif dtype == 7:
        return 'INT64'
    elif dtype == 8:
        return 'STRING'
    elif dtype == 9:
        return 'BOOL'
    elif dtype == 10:
        return 'FLOAT16'
    elif dtype == 11:
        return 'DOUBLE'
    elif dtype == 12:
        return 'UINT32'
    elif dtype == 13:
        return 'UINT64'
    elif dtype == 14:
        return 'COMPLEX64'
    elif dtype == 15:
        return 'COMPLEX128'
    elif dtype == 16:
        return 'BFLOAT16'
    else:
        raise Exception('Unknown onnx datatype')

def _check_output(taso_output, onnx_output):
    # TODO: check output match
    return True

def _parse_attribute(attributes):
    atts = dict()
    for att in attributes:
        if att.type == onnx.AttributeProto.INT:
            atts[att.name] = att.i
        elif att.type == onnx.AttributeProto.INTS:
            atts[att.name] = att.ints
        elif att.type == onnx.AttributeProto.FLOAT:
            atts[att.name] = att.f
        elif att.type == onnx.AttributeProto.STRING:
            atts[att.name] = att.s
        elif att.type == onnx.AttributeProto.TENSOR:
            atts[att.name] = att.t
        else:
            assert False, "Unsupported Attribute Type: {}".format(att.type)
    return atts

def _get_conv_pool_pads_attr(attrs):
    if ("auto_pad" in attrs):
        padding = attrs["auto_pad"]
        if isinstance(padding, bytes):
            padding = padding.decode()
        if (padding=='SAME_LOWER') or (padding=='SAME_UPPER'):
            pads = "SAME"
        elif (padding=='VALID'):
            pads = "VALID"
        else:
            assert padding=='NOTSET', "Unrecogonized auto_pad value: {}".format(padding)
        # Note that we always think conv1x1 has SAME padding
        # This will allow fusing enlarged convs
        if sum(attrs['kernel_shape']) <= 2:
            pads = "SAME"
        if padding != 'NOTSET':
            return pads
    # Assume zero padding if the pads are missing
    if "pads" not in attrs:
        attrs['pads'] = [0 for i in range(len(attrs['kernel_shape'])*2)]
    # Note that we always think conv1x1 has SAME padding
    # This will allow fusing enlarged convs
    if sum(attrs["pads"]) == 0 and sum(attrs['kernel_shape']) > 2:
        pads = "VALID"
    else:
        pads = "SAME"
    return pads

def _get_list_from_initializer(initializer, name):
    for data in initializer:
        if data.name == name:
            ret = list()
            if data.int64_data != []:
                for dim in data.int64_data:
                    ret.append(dim)
            elif data.raw_data and data.raw_data != []:
                ret_in_array = numpy_helper.to_array(data)
                for dim in ret_in_array:
                        ret.append(dim)
            return ret
    raise InputNotFoundError
    return []

def _get_inputs(op, graph, tensors, initializer):
    inputs = list()
    for i in op.input:
        if i == '': continue
        
        input_tensor = None
        if i in tensors:
            input_tensor = tensors[i]
            # try:
            #     print(input_tensor.dim(0), input_tensor.dim(1), input_tensor.dim(2), input_tensor.dim(3))
            # except AssertionError:
            #     try:
            #         print(input_tensor.dim(0), input_tensor.dim(1), input_tensor.dim(2))
            #     except AssertionError:
            #         try:
            #             print(input_tensor.dim(0), input_tensor.dim(1))
            #         except AssertionError:
            #             print(input_tensor.dim(0))
        else:
            for init in initializer:
                if init.name == i:
                    input_tensor = graph.new_weight(
                        dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
                    break
        if input_tensor is None:
            raise InputNotFoundError
            return []
        inputs.append(input_tensor)
        
    return inputs
        
def _add(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    outputs = graph.add(inputs[0], inputs[1])
    return outputs

def _argmax(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ArgMax requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axis = attrs["axis"]
    axes_list = [axis]
    outputs = graph.reduce_argmax(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _argmin(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ArgMin requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axis = attrs["axis"]
    axes_list = [axis]
    outputs = graph.reduce_argmin(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _batchnorm(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    if 'epsilon' in attrs:
        epsilon = attrs['epsilon']
    else:
        epsilon = -1
    outputs = graph.batchnorm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon)
    return outputs

def _cast(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    #assert len(op.input) == 1, "Cast requires exactly one input"
    #input_tensor = None
    #if op.input[0] in tensors:
    #    input_tensor = tensors[op.input[0]]
    #else:
    #    for init in initializer:
    #        if init.name == op.input[0]:
    #            input_tensor = graph.new_weight(
    #                dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
    #            break
    #assert input_tensor is not None, "Input Tensor Not Found"
    attrs = _parse_attribute(op.attribute)
    to_type = onnx_datatype_tostring(attrs["to"])
    outputs = graph.cast(input=inputs[0], datatype=to_type)
    return outputs

def _ceil(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Ceil requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.ceil(inputs[0])
    return outputs

def _concat(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    axis = attrs["axis"]
    outputs = graph.concat(axis, inputs)
    return outputs

def _constant(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    # TODO: Currently do not support sparse value
    assert "value" in attrs, "Do not support sparse value for Constant"
    tensor = attrs["value"]
    dims = list()
    for dim in tensor.dims:
        dims.append(dim)
    weight_data = numpy_helper.to_array(tensor)
    outputs = graph.new_weight(dims=tuple(dims), data=weight_data)
    return outputs

def _conv2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    if "group" not in attrs:
        group = 1 # default 1
    else:
        group = attrs["group"]
    pads = _get_conv_pool_pads_attr(attrs)
    strides = attrs["strides"]
    dilations = attrs["dilations"]
    outputs = graph.conv2d(input=inputs[0], weight=inputs[1], strides=strides, padding=pads,
                           dilations=dilations)
    
    if len(inputs) > 2:
        dim = inputs[2].dim(0)
        reshaped_bias = graph.reshape(inputs[2], (1, dim, 1, 1))
        outputs = graph.add(outputs, reshaped_bias)
    return outputs

def _div(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Div takes exactly two inputs"
    outputs = graph.div(x=inputs[0], y=inputs[1])
    return outputs

def _dropout(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Dropout takes exactly one input"
    attrs = _parse_attribute(op.attribute)
    rate = attrs["ratio"]
    outputs = graph.dropout(input=inputs[0], rate=rate)
    return outputs

def _equal(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Equal takes exactly two inputs"
    outputs = graph.equal(x=inputs[0], y=inputs[1])
    return outputs

def _exp(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Exp requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.exp(input=inputs[0])
    return outputs

def _expand(op, graph, tensors, initializer):
    # Returns an output tensor shape equal to the spec in 'shape' input
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Expand requires exactly two input"
    
    _shape = get_data_from_init_list(op, initializer, 1)
    outputs = graph.expand(inputs[0], tuple(_shape))
    return outputs

def _erf(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Erf requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.erf(input=inputs[0])
    return outputs

def _flatten(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Flatten requires exactly one input"
    shape = []
    shape.append(inputs[0].dim(0))
    dim = 1
    for i in range(1, inputs[0].nDim):
        dim *= inputs[0].dim(i)
    shape.append(dim)
    outputs = graph.reshape(inputs[0], tuple(shape))
    return outputs

def _gemm(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    if "transA" in attrs and attrs["transA"] == 1:
        inputs[0] = graph.transpose(inputs[0], (1,0), shuffle=True)
    if "transB" in attrs and attrs["transB"] == 1:
        inputs[1] = graph.transpose(inputs[1], (1,0), shuffle=True)
    outputs = graph.matmul(inputs[0], inputs[1])
    
    if len(inputs) > 2:
        dim = inputs[2].dim(0)
        dims_tuple = np.ones(outputs.nDim)
        dims_tuple[-1] = dim
        reshaped_bias = graph.reshape(inputs[2], tuple(dims_tuple))
        outputs = graph.add(outputs, reshaped_bias)
    return outputs

def _greater(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Greater takes exactly two inputs"
    outputs = graph.greater(inputs[0], inputs[1])
    return outputs

def _identity(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Identity takes exactly one input"
    outputs = graph.dropout(inputs[0], 0.0)
    return outputs

def _leakyrelu(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "LeakyRelu requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    alpha = attrs["alpha"]
    outputs = graph.leakyrelu(input=inputs[0], alpha=alpha)
    return outputs

def _less(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Less takes exactly two inputs"
    outputs = graph.less(inputs[0], inputs[1])
    return outputs

def _log(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Log requires exactly one input"
    
    attrs = _parse_attribute(op.attribute)
    outputs = graph.log(input=inputs[0])
    return outputs

def _logical_not(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Not requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.logical_not(input=inputs[0])
    return outputs

def _matmul(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "MatMul takes exactly two inputs"
    
    
    if(inputs[0].nDim != inputs[1].nDim):  
        # check if dims are the same. if not, broadcast inputs[1] to inputs[0]
        numDims = inputs[0].nDim
        i_dims = np.zeros(numDims)
        for j in range(numDims):
            i_dims[j] = inputs[0].dim(j)
        
        dims = [inputs[0].dim(i) for i in range(numDims)]
        j = numDims - 1
        # iterate backwards through dims and overwrite with 1s
        for i in range(inputs[1].nDim - 1, -1, -1):
            dims[j] = 1
            j = j -1
        
        # find initalizer index to update raw_data
        
        for init in initializer:
            if init.name == op.input[1]:
                break
            
        weight_data = numpy_helper.to_array(init)
        ones = np.ones(dims)
        b_weight_data = (ones * weight_data).astype(weight_data.dtype)
        weight_tensor_proto = b_weight_data.tobytes()
        
        # update initializer list with broadcasted weight data
        for init in initializer:
            if init.name == op.input[1]:
                # Modify the raw_data field of the initializer
                init.raw_data = weight_tensor_proto

                # Modify the dimensions of the initializer
                init.dims[:] = b_weight_data.shape
        
        # print("input shape ", i_dims)
        # print("weight_data ", weight_data.shape, weight_data.dtype)
        # print("b_weight_data ", b_weight_data.shape, b_weight_data.dtype)
        
        # verify
        for init in initializer:
            if init.name == op.input[1]:
                v_weight_data = numpy_helper.to_array(init)
                assert(np.all(b_weight_data == v_weight_data))
                
                # print("v_weight_data", v_weight_data.shape)
        
        b_weight_tensor = graph.new_weight(dims=tuple(b_weight_data.shape), data=b_weight_data)
        outputs = graph.matmul(inputs[0], b_weight_tensor)
        
    else:
        outputs = graph.matmul(inputs[0], inputs[1])
    return outputs

def _min(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Min takes exactly two inputs"
    outputs = graph.min(inputs[0], inputs[1])
    return outputs

def _mul(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Mul takes exactly two inputs"
    outputs = graph.mul(inputs[0], inputs[1])
    return outputs

def _pad(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    try:
        pads = np.array(attrs['pads'])
        #byte string is directly parseable by Cython; no need to decode("utf-8")
        pad_mode = attrs['mode']
        input_rank = inputs[0].nDim
        # First N (rank of input tensor) defines the pad_before for each axis
        pad_before = pads[:input_rank]
        
        # Last N (rank of input tensor) defines the pad_before for each axis
        pad_after = pads[input_rank:]
        
        outputs = graph.pad(inputs[0], tuple(pad_before), tuple(pad_after), float(0), pad_mode)
        return outputs
    
    except KeyError:
        # TODO: support padding as an input as per new version of ONNX
        assert "pads as input not supported; currently supports padding as attribute"

def _prelu(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "PRelu requires exactly two inputs"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.prelu(x=inputs[0], slope=inputs[1])
    return outputs

def _pow(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Pow requires exactly two inputs"
    outputs = graph.pow(inputs[0], inputs[1])
    return outputs

def _max(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Max takes exactly two inputs"
    outputs = graph.max(inputs[0], inputs[1])
    return outputs

def _maxpool2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "MaxPool2D requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    kernels = attrs["kernel_shape"]
    strides = attrs["strides"]
    pads = _get_conv_pool_pads_attr(attrs)
    outputs = graph.maxpool2d(input=inputs[0], kernels=kernels, strides=strides, padding=pads)
    return outputs

def _avgpool2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "AvgPool2D requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    kernels = attrs["kernel_shape"]
    strides = attrs["strides"]
    pads = _get_conv_pool_pads_attr(attrs)
    outputs = graph.avgpool2d(input=inputs[0], kernels=kernels, strides=strides, padding=pads)
    return outputs

def _globalavgpool2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "GlobalAvgPool2D requires exactly one input"
    dim = inputs[0].dim(inputs[0].nDim-1)
    kernels = [dim, dim]
    strides = [1, 1]
    pads = "VALID"
    outputs = graph.avgpool2d(input=inputs[0], kernels=kernels, strides=strides, padding=pads)
    return outputs

def _reducemax(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceMax requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    
    try:
        keepdims = attrs["keepdims"]
    except KeyError:
        # Refer: https://github.com/onnx/onnx/blob/main/docs/Operators.md
        keepdims = 1
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        # Workaround to convert negative axes index to positive axes index
        # Required by TASO internal
        if i < 0:
            axes_list.append(inputs[0].nDim + i)
        else:
            axes_list.append(i)
    outputs = graph.reduce_max(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reducemean(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceMean requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    
    try:
        keepdims = attrs["keepdims"]
    except KeyError:
        # Refer: https://github.com/onnx/onnx/blob/main/docs/Operators.md
        keepdims = 1
        
    axes_ints = attrs["axes"]
    axes_list = list()
    
    for i in axes_ints:
        # Workaround to convert negative axes index to positive axes index
        # Required by TASO internal
        if i < 0:
            axes_list.append(inputs[0].nDim + i)
        else:
            axes_list.append(i)
            
    outputs = graph.reduce_mean(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reducemin(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceMin requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    
    try:
        keepdims = attrs["keepdims"]
    except KeyError:
        # Refer: https://github.com/onnx/onnx/blob/main/docs/Operators.md
        keepdims = 1
        
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        # Workaround to convert negative axes index to positive axes index
        # Required by TASO internal
        if i < 0:
            axes_list.append(inputs[0].nDim + i)
        else:
            axes_list.append(i)
    outputs = graph.reduce_min(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reduceprod(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceProd requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    
    try:
        keepdims = attrs["keepdims"]
    except KeyError:
        # Refer: https://github.com/onnx/onnx/blob/main/docs/Operators.md
        keepdims = 1
        
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        axes_list.append(i)
    outputs = graph.reduce_prod(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reducesum(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceSum requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    
    try:
        keepdims = attrs["keepdims"]
    except KeyError:
        # Refer: https://github.com/onnx/onnx/blob/main/docs/Operators.md
        keepdims = 1
        
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        axes_list.append(i)
    outputs = graph.reduce_sum(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reshape(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2
    shape = list()
    for data in initializer:
        if data.name == op.input[1]:
            shape = list()
            if data.int64_data != []:
                for dim in data.int64_data:
                    shape.append(dim)
            elif data.raw_data and data.raw_data != []:
                shape_in_array = numpy_helper.to_array(data)
                for dim in shape_in_array:
                    shape.append(dim)
    
    outputs = graph.reshape(inputs[0], tuple(shape))
    return outputs

def get_data_from_init_list(op, initializer, index_num):
    extracted_list = list()
    for data in initializer:
        if data.name == op.input[index_num]:
            if data.int64_data != []:
                for val in data.int64_data:
                    extracted_list.append(val)
            elif data.raw_data and data.raw_data != []:
                extracted_list = numpy_helper.to_array(data)
    return extracted_list
                
def _resize(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) >= 2, "Resize takes at least two inputs"
    if(len(op.input)==4):
        assert not(op.input[2] != '' and op.input[3] != ''), "Resize node CANNOT specify both scales and sizes"
    
    attrs = _parse_attribute(op.attribute)
    
    # Setting default attribute values to start off
    coord_transf_mode = 'half-pixel'.encode('utf-8')
    cubic_coeff_a = -0.75
    mode = 'nearest'.encode('utf-8')
    nearest_mode = 'round_prefer_floor'.encode('utf-8')
    
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
    # inputs[0]: Input tensor on which to perform resize op
    # inputs[1]: ROI (optional)
    # inputs[2]: scales (optional)
    # inputs[3]: sizes (optional)
    
    # Resize behaviour is set by attributes, not relevant for TASO
    # Output tensor dimensions is calculated from 'scales' or 'sizes' input
    # scale = length_resized / length_original => out_size[i] = scale * input_size[i]
    # Only scales or sizes can be specified;
    # interpretation of sizes depends on keep_aspect_ratio_policy : string (default is stretch)
        # not applicable for 'scale', only 'sizes'
        # if 'stretch' mode, orginal aspect ratio is ignored and out_size[i] = sizes[i]
    
    sizes = list()
    if(len(op.input) == 2):
        # Likely redirected Upsample node (deprecated) to resize
        scales = get_data_from_init_list(op, initializer, 1)
        input_tensor_dims = list()
        for i in range(inputs[0].nDim):
            input_tensor_dims.append(inputs[0].dim(i))
        sizes = np.multiply(input_tensor_dims, scales)
        
        # Upsample (deprecated) only specifies mode
        mode = attrs['mode']  # default: 'nearest'
        
    elif(len(op.input) == 3):
        # extract scales and compute sizes for output tensor
        scales = get_data_from_init_list(op, initializer, 2)
        input_tensor_dims = list()
        for i in range(inputs[0].nDim):
            input_tensor_dims.append(inputs[0].dim(i))
        sizes = np.multiply(input_tensor_dims, scales)
        
    elif(len(op.input) == 4):
        sizes = get_data_from_init_list(op, initializer, 3)
    
    if(len(op.input) != 2):
        # Reading in attributes for actual Resize node
        # Guard check required due to Upsample (deprecated) -> redirection to Resize node
        coord_transf_mode = attrs['coordinate_transformation_mode'] # default: 'half-pixel'
        cubic_coeff_a = attrs['cubic_coeff_a'] # default: -0.75
        mode = attrs['mode']  # default: 'nearest'
        nearest_mode = attrs['nearest_mode'] #default: 'round_prefer_floor'
    
    outputs = graph.resize(inputs[0], tuple(sizes), coord_transf_mode, cubic_coeff_a, mode, nearest_mode)
    return outputs

# TensorFlow resize_nearest_neighbor
# https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/resize-nearest-neighbor
def _resize_nearest_neighbor(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "ResizeNearestNeighbor takes exactly two inputs"
    shape = list()
    for data in initializer:
        if data.name == op.input[1]:
            for dim in data.int64_data:
                shape.append(dim)
    assert len(shape) == 2, "ResizeNeareestNeighbor: new size cannot be statically inferred"
    outputs = graph.resize_nearest_neighbor(input=inputs[0], new_height=shape[0], new_width=shape[1])
    return outputs

# TensorFlow crop_and_resize
# https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/crop-and-resize
def _crop_and_resize(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 4, "CropAndResize takes exactly four inputs"
    outputs = graph.crop_and_resize(inputs[0], inputs[1], inputs[2], inputs[3])
    return outputs

def _relu(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Relu requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.relu(input=inputs[0])
    return outputs

def _round(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Round requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.round(inputs[0])
    return outputs

def _shape(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs)== 1, "Shape requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.shape(inputs[0])
    return outputs

def _sigmoid(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Sigmoid requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.sigmoid(input=inputs[0])
    return outputs

def _softplus(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Softplus requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.softplus(input=inputs[0])
    return outputs

def _size(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Size requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.size(inputs[0])
    return outputs

def _slice(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) >= 3, "Slice requires at least 3 inputs"
    assert len(inputs) <= 5, "Slice takes at most 5 inputs"
    start = _get_list_from_initializer(initializer, op.input[1])
    # replace INT_MAX with 999999
    for i in range(len(start)):
        start[i] = min(999999, start[i])
    end = _get_list_from_initializer(initializer, op.input[2])
    # replace INT_MAX with 999999
    for i in range(len(end)):
        end[i] = min(999999, end[i])
    if len(op.input) > 3:
        axes = _get_list_from_initializer(initializer, op.input[3])
    else:
        axes = None
    if len(op.input) > 4:
        steps = _get_list_from_initializer(initializer, op.input[4])
    else:
        steps = None
    
    num_inputs = len(op.input)
    outputs = graph.slice(inputs[0], start, end, axes, steps, num_inputs)
    return outputs

def _split(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Split requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    axis = attrs["axis"]
    split_ints = attrs["split"]
    if type(split_ints) is not list:
        origin_dim = inputs[0].dim(axis)
        split_list = [int(origin_dim / split_ints)] * split_ints
        outputs = graph.split(inputs[0], axis, split_list)
    else:
        split_list = list()
        for i in split_ints:
            split_list.append(i)
        outputs = graph.split(inputs[0], axis, split_list)
    return outputs

def _sqrt(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Sqrt requires exactly one input"
    #input_tensor = None
    #if op.input[0] in tensors:
    #    input_tensor = tensors[op.input[0]]
    #else:
    #    for init in initializer:
    #        if init.name == op.input[0]:
    #            input_tensor = graph.new_weight(
    #                dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
    #            break
    #assert input_tensor is not None, "Input Tensor Not Found"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.sqrt(input=inputs[0])
    return outputs

def _squeeze(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Squeeze takes exactly one input"
    attrs = _parse_attribute(op.attribute)
    axes_ints = attrs["axes"]
    axes = list()
    for i in axes_ints:
        axes.append(i)
    outputs = graph.squeeze(input=inputs[0], axes=tuple(axes))
    return outputs

def _strided_slice(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 4, "StrideSlice takes exactly four inputs"
    start = _get_list_from_initializer(initializer, op.input[1])
    end = _get_list_from_initializer(initializer, op.input[2])
    steps = _get_list_from_initializer(initializer, op.input[3])
    attrs = _parse_attribute(op.attribute)
    begin_mask = attrs["begin_mask"]
    end_mask = attrs["end_mask"]
    ellipsis_mask = attrs["ellipsis_mask"]
    new_axis_mask = attrs["new_axis_mask"]
    shrink_axis_mask = attrs["shrink_axis_mask"]
    # TODO: support new_axis and shrink axis
    assert new_axis_mask == 0, "Non zero new_axis_mask is not supported yet"
    assert shrink_axis_mask == 0, "Non zero shrink_axis_mask is not supported yet"
    # TODO: current assume that strided slice returns the original tensor
    outputs = graph.slice(inputs[0], None, None, None, None)
    return outputs

def _sub(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Sub takes exactly two inputs"
    outputs = graph.sub(x=inputs[0], y=inputs[1])
    return outputs

def _sum(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "TASO assumes Sum takes exactly two inputs. Submit a github issue when you see this."
    outputs = graph.add(inputs[0], inputs[1])
    return outputs

def _tanh(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Tanh requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.tanh(input=inputs[0])
    return outputs

def _transpose(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Transpose requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    perm_ints = attrs["perm"]
    perm = list()
    for i in perm_ints:
        perm.append(i)
    outputs = graph.transpose(inputs[0], tuple(perm), shuffle=True)
    return outputs

def _unsqueeze(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Unsqueeze takes exactly one input"
    #input_tensor = None
    #if op.input[0] in tensors:
    #    input_tensor = tensors[op.input[0]]
    #else:
    #    for init in initializer:
    #        if init.name == op.input[0]:
    #            input_tensor = graph.new_weight(
    #                dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
    #            break
    #assert input_tensor is not None, "Input Tensor Not Found"
    attrs = _parse_attribute(op.attribute)
    axes_ints = attrs["axes"]
    axes = list()
    for i in axes_ints:
        axes.append(i)
    outputs = graph.unsqueeze(input=inputs[0], axes=tuple(axes))
    return outputs

# Add all supported operators
xf_operators = dict()
xf_operators['Add'] = _add
xf_operators['ArgMax'] = _argmax
xf_operators['ArgMin'] = _argmin
xf_operators['BatchNormalization'] = _batchnorm
xf_operators['Cast'] = _cast
xf_operators['Ceil'] = _ceil
xf_operators['Concat'] = _concat
xf_operators["Constant"] = _constant
xf_operators['Conv'] = _conv2d
xf_operators['Div'] = _div
xf_operators['Dropout'] = _dropout
xf_operators['Equal'] = _equal
xf_operators['Exp'] = _exp
xf_operators['Expand'] = _expand
xf_operators['Erf'] = _erf
xf_operators['Flatten'] = _flatten
xf_operators['Gemm'] = _gemm
xf_operators['Greater'] = _greater
xf_operators['Identity'] = _identity
xf_operators['LeakyRelu'] = _leakyrelu
xf_operators['Less'] = _less
xf_operators['Log'] = _log
xf_operators['Pad'] = _pad
xf_operators['PRelu'] = _prelu
xf_operators['Pow'] = _pow
xf_operators['ReduceMax'] = _reducemax
xf_operators['ReduceMean'] = _reducemean
xf_operators['ReduceMin'] = _reducemin
xf_operators['ReduceProd'] = _reduceprod
xf_operators['ReduceSum'] = _reducesum
xf_operators['Reshape'] = _reshape
xf_operators['Resize'] = _resize
xf_operators['Relu'] = _relu
xf_operators['Round'] = _round
xf_operators['MatMul'] = _matmul
xf_operators['Max'] = _max
xf_operators['MaxPool'] = _maxpool2d
xf_operators['Min'] = _min
xf_operators['Mul'] = _mul
xf_operators['Not'] = _logical_not
xf_operators['AveragePool'] = _avgpool2d
xf_operators['GlobalAveragePool'] = _globalavgpool2d
xf_operators['Shape'] = _shape
xf_operators['Size'] = _size
xf_operators['Sigmoid'] = _sigmoid
xf_operators['Slice'] = _slice
xf_operators['Softplus'] = _softplus
xf_operators['Split'] = _split
xf_operators['Sqrt'] = _sqrt
xf_operators['Squeeze'] = _squeeze
xf_operators['StridedSlice'] = _strided_slice
xf_operators['Sub'] = _sub
xf_operators['Sum'] = _sum
xf_operators['Transpose'] = _transpose
xf_operators['Tanh'] = _tanh
xf_operators['Unsqueeze'] = _unsqueeze
xf_operators['Upsample'] = _resize  # Upsample is deprecated, redirect to _resize


def new_graph(print_measurements = False):
    graph = core.PyGraph()
    if print_measurements:
        graph.print_measurements()
    return graph

def load_onnx(filename):
    '''
    Load a onnx file and return a Graph

    @params
    filename is a string containing a file name
    @return
    Loaded in-memory Graph
    '''
    
    graph = core.PyGraph()
    model = onnx.load(filename)
    tensors = dict()
    for t in model.graph.input:
        dims = list()
        for d in t.type.tensor_type.shape.dim:
            dims.append(d.dim_value)
        weight_data = None
        for weight in model.graph.initializer:
            if (weight.name == t.name):
                weight_data = numpy_helper.to_array(weight)
        # We classify an input to be a pure input if we cannot find its weights
        if weight_data is None:
            tensors[t.name] = graph.new_input(dims=tuple(dims))
        else:
            tensors[t.name] = graph.new_weight(dims=tuple(dims), data=weight_data)

    # Add initializers not in the inputs
    for weight in model.graph.initializer:
        if weight.name not in tensors:
            if weight.dims:
                dims = list(weight.dims)
                weight_data = numpy_helper.to_array(weight)
                tensors[weight.name] = graph.new_weight(dims=tuple(dims), data=weight_data)
    
    # Reorder nodes to satisfy data dependencies
    tensor_owner = dict()
    name_to_op = dict()
    idx = 0
    for op in model.graph.node:
        # Assign a name to the node if empty
        if len(op.name) == 0:
            op.name = op.op_type + '_' + str(idx)
        idx += 1
        name_to_op[op.name] = op
        for output in op.output:
            tensor_owner[output] = op.name
    out_edges = dict()
    dependents = dict()
    node_list = list()
    for op in model.graph.node:
        dependents[op.name] = 0
        for input in op.input:
            if input in tensor_owner:
                dependents[op.name] += 1
                input_node = tensor_owner[input]
                if input_node not in out_edges:
                    out_edges[input_node] = list()
                out_edges[input_node].append(op.name)
        if dependents[op.name] == 0:
            node_list.append(op.name)
    idx = 0
    while idx < len(node_list):
        opname = node_list[idx]
        if opname in out_edges:
            for e in out_edges[opname]:
                dependents[e] -= 1
                if dependents[e] == 0:
                    node_list.append(e)
        idx += 1
    assert len(node_list) == len(model.graph.node), "Internal error when reording ONNX operators"
    
    # Add nodes into TASO graph
    cnt = 0
    for opname in node_list:
        op = name_to_op[opname]
        print(cnt, op.op_type, opname)
        cnt += 1
        if op.op_type in xf_operators:
            try:
                # TODO: Assigning to model.graph.initializer due to explicit broadcast semantics in TASO
                outputs = xf_operators[op.op_type](op, graph, tensors, model.graph.initializer)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                assert len(outputs) == len(op.output), "Number of output tensors mismatch"
                for i in range(len(outputs)):
                    assert _check_output(outputs[i], op.output[i])
                    tensors[op.output[i]] = outputs[i]
                
                # if op.op_type == 'MatMul':
                #     for init in model.graph.initializer:
                #         if init.name == op.input[1]:
                #             v_weight_data = numpy_helper.to_array(init)
                #             print("in load_onnx:", v_weight_data.shape)
                
                # print(out_edges['/model/decode_head/Reshape'])
                # print(out_edges['/model/decode_head/Resize'])
                # print(tensor_owner['/model/decode_head/Reshape_output_0'])
                # # print(tensors['/model/decode_head/Concat_output_0'])
                # sys.exit()
            except InputNotFoundError:
                print("Cannot find input tensor for operator: name({}) type({}) (Skipped)".format(opname, op.op_type))
                continue
        else:
            print("Found unsupported ONNX operator: {} (Skipped)".format(op.op_type))
            continue
    return graph

input_weight_names = dict()
input_weight_names['Add'] = ['input1', 'input2']
input_weight_names['AveragePool'] = ['input']
input_weight_names['BatchNormalization'] = ['input', 'scale', 'bias', 'mean', 'var']
input_weight_names['Concat'] = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6']
input_weight_names['Conv'] = ['input', 'weight', 'bias']
input_weight_names['Div'] = ['input1', 'input2']
input_weight_names['Expand'] = ['input', 'shape']
input_weight_names['MatMul'] = ['input', 'weight']
input_weight_names['Mul'] = ['input1', 'input2']
input_weight_names['Pow'] = ['input1', 'input2']
input_weight_names['Pad'] = ['input', 'pads']
input_weight_names['Reshape'] = ['input', 'shape']
input_weight_names['Resize'] = ['input', 'roi', 'scales', 'sizes']
input_weight_names['Sub'] = ['input1', 'input2']
input_weight_names['BroadcastAdd'] = ['input1', 'input2']
input_weight_names['Transpose'] = ['input']
input_weight_names['Slice'] = ['input', 'starts', 'ends', 'axes', 'steps']
input_weight_names['Split'] = ['input', 'split']
input_weight_names["Squeeze"] = ['input', 'axes']
input_weight_names['ReduceSum'] = ['input', 'axes']
input_weight_names['ReduceMean'] = ['input', 'axes']
input_weight_names['ReduceMax'] = ['input', 'axes']

operator_attrs = dict()
operator_attrs['Add'] = []
operator_attrs['ArgMax'] = []
operator_attrs['ArgMin'] = []
operator_attrs['AveragePool'] = ['kernel_shape', 'pads', 'strides']
operator_attrs['BatchNormalization'] = ['epsilon'] # TODO: Add momentum
operator_attrs['Cast'] = []
operator_attrs['Ceil'] = []
operator_attrs['Concat'] = ['axis']
operator_attrs['Conv'] = ['dilations', 'group', 'kernel_shape', 'pads', 'strides']
operator_attrs['Div'] = []
operator_attrs['Dropout'] = []
operator_attrs['Erf'] = []
operator_attrs['Exp'] = []
operator_attrs['Expand'] = []
operator_attrs['Gemm'] = [] # this is fissioned into Matmul + Add
operator_attrs['Greater'] = []
operator_attrs['Identity'] = []
operator_attrs['Less'] = []
operator_attrs['Log'] = []
operator_attrs['Pad'] = ['mode']
operator_attrs['Pow'] = []
operator_attrs['MatMul'] = []
operator_attrs['MaxPool'] = ['kernel_shape', 'pads', 'strides']
operator_attrs['Mul'] = []
operator_attrs['Shape'] = []
operator_attrs['Sigmoid'] = []
operator_attrs['Slice'] = []
operator_attrs['Softplus'] = []
operator_attrs['Split'] = ['axis'] # 'split': After opset 11, moved from operator attr to op input
operator_attrs["Squeeze"] = [] # 'axes': After opset 11, moved from operator attr to op input
operator_attrs['Sqrt'] = []
operator_attrs['StridedSlice'] = []
operator_attrs['Sub'] = []
operator_attrs['Relu'] = []
operator_attrs['LeakyRelu'] = ['alpha']
operator_attrs['Reshape'] = ['allowzero']
operator_attrs['Resize'] = ['coordinate_transformation_mode', 'cubic_coeff_a', 'mode', 'nearest_mode']
operator_attrs['Tanh'] = []
operator_attrs['Transpose'] = ['perm']
operator_attrs['Unsqueeze'] = ['axes']
# TODO: register this or catch and generate a Resize node?
operator_attrs['Upsample'] = ['nearest']
operator_attrs['BroadcastAdd'] = []
operator_attrs['ReduceMax'] = ['keepdims']
operator_attrs['ReduceMean'] = ['keepdims']
operator_attrs['ReduceSum'] = ['keepdims']

def _input_tensor_name(graph, inedge, op):
    intype = graph.get_operator_type(inedge['srcOp'])
    if intype == "Input":
        return "data"
    elif intype == "Weight":
        mytype = graph.get_operator_type(op)
        return "{}{}_{}".format(mytype, op['guid'], input_weight_names[mytype][inedge['dstIdx']])
    else:
        return _output_tensor_name(graph, inedge['srcOp'], inedge['srcIdx'])

def _output_tensor_name(graph, op, idx):
    type = graph.get_operator_type(op)
    return "{}{}_fwd{}".format(type, op['guid'], idx)

def _add_node_attribute(graph, node, op, optype):
    for key in operator_attrs[optype]:
        val = graph.get_operator_attr(op, key)
        attr = helper.make_attribute(key, val)
        node.attribute.append(attr)

# Helper function to register tensors that should be at Op input
def export_register_input_tensor(inputs, graph_inputs_dict, graph_initializers, mytype, op, py_array, input_idx):
    input_tensor_name = "{}{}_{}".format(mytype, op['guid'], input_weight_names[mytype][input_idx])
    inputs.append(input_tensor_name)
    graph_inputs_dict[input_tensor_name] = helper.make_tensor_value_info(input_tensor_name, TensorProto.INT64, [len(py_array)])
    graph_initializers.append(helper.make_tensor(input_tensor_name, TensorProto.INT64, [len(py_array)], py_array))
    
    return inputs, graph_inputs_dict, graph_initializers
                    
def export_onnx(graph):
    '''
    Export a XFlow graph to an ONNX graph
    @params
    graph is a XFlow graph

    @return
    A in-memory ONNX graph
    '''
    opList = graph.get_operator_list()
    graph_nodes = list()
    graph_inputs_dict = dict()
    graph_initializers = list()
    graph_outputs = list()
    output_guids = dict()
    for op in opList:
        mytype = graph.get_operator_type(op)
        inedges = graph.get_input_edges(op)
        # print("op.guid={} mytype={} inedges={}".format(op['guid'], mytype, len(inedges)))
        inputs = list()
        for e in inedges:
            intype = graph.get_operator_type(e['srcOp'])
            inputs.append(_input_tensor_name(graph, e, op))
            output_guids.pop((e['srcOp']['guid'], e['srcIdx']), None)
            if intype == 'Input' or intype == 'Weight':
                new_input_name = _input_tensor_name(graph, e, op)
                if new_input_name not in graph_inputs_dict.keys():
                    graph_inputs_dict[new_input_name] = helper.make_tensor_value_info(_input_tensor_name(graph, e, op),
                                        TensorProto.FLOAT, graph.get_input_dims(op, e['dstIdx']))
            if intype == 'Weight':
                graph_initializers.append(helper.make_tensor(_input_tensor_name(graph, e, op),
                                            TensorProto.FLOAT, graph.get_input_dims(op, e['dstIdx']),
                                            graph.get_weight_value(e['srcOp'])))
        
        # account for ONNX operators where attributes became input tensors
        if mytype == 'Reshape':
            # inputs.append('Reshape_attr{}'.format(op['guid']))
            shape = graph.get_output_dims(op, 0)
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, shape, 1)
            # graph_inputs.append(helper.make_tensor_value_info('Reshape_attr{}'.format(op['guid']), TensorProto.INT64, [len(shape)]))
            # graph_initializers.append(helper.make_tensor('Reshape_attr{}'.format(op['guid']), TensorProto.INT64, [len(shape)], shape))
        
        elif mytype == 'Slice':
            num_inputs = graph.get_operator_int_attr(op, 'num_inputs')
            assert num_inputs >= 3, "Slice requires at least 3 inputs"
            assert num_inputs <= 5, "Slice takes at most 5 inputs"
            if(num_inputs == 3):
                starts = graph.get_op_init_vector(op, 'starts')
                ends = graph.get_op_init_vector(op, 'ends')
                
                inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, starts, 1)
                
                inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, ends, 2)
            
            elif(num_inputs == 5):
                starts = graph.get_op_init_vector(op, 'starts')
                ends = graph.get_op_init_vector(op, 'ends')
                axes = graph.get_op_init_vector(op, 'axes')
                steps = graph.get_op_init_vector(op, 'steps')
                
                inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, starts, 1)
                
                inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, ends, 2)
                
                inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, axes, 3)
                
                inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, steps, 4)
        
        elif mytype == 'Split':
            splits = graph.get_split_lens(op)
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, splits, 1)
        elif mytype in ['ReduceSum', 'ReduceMean', 'ReduceMax', 'Squeeze']:
            axes = graph.get_operator_attr(op, 'axes')
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, axes, 1)
        elif mytype == 'Pad':
            pads = graph.get_op_init_vector(op, 'pads')
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, pads, 1)
        elif mytype == 'Expand':
            shape = graph.get_output_dims(op, 0)
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, shape, 1)
        elif mytype == 'Resize':
            # empty list for roi specification
            # if required regsiter into Resize node
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, [], 1)
            
            # empty list for scales specification
            # One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified.
            # If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this
            # operator's input list.
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, "", 2)
            
            sizes = graph.get_op_init_vector(op, 'sizes')
            inputs, graph_inputs_dict, graph_initializers = export_register_input_tensor(inputs, graph_inputs_dict, 
                                                                                        graph_initializers, 
                                                                                        mytype, op, sizes, 3)
        outputs = list()
        for i in range(graph.get_num_outputs(op)):
            outputs.append(_output_tensor_name(graph, op, i))
            output_guids[(op['guid'], i)] = op
        node = helper.make_node(mytype, inputs, outputs, '{}{}'.format(mytype, op['guid']))
        
        if(mytype == 'Conv'):
            for key in operator_attrs[mytype]:
                # print("getting: ", key)
                val = graph.get_operator_attr(op, key)
                # print("Got: ", val)
                attr = helper.make_attribute(key, val)
                node.attribute.append(attr)
        else:
            _add_node_attribute(graph, node, op, mytype)
        graph_nodes.append(node)
        
    for guid, idx in output_guids:
        op = output_guids[(guid, idx)]
        graph_outputs.append(helper.make_tensor_value_info(_output_tensor_name(graph, op, idx),
                             TensorProto.FLOAT, graph.get_output_dims(op, idx)))
    
    graph_inputs = list(graph_inputs_dict.values())
    onnx_graph = helper.make_graph(graph_nodes, 'main', graph_inputs, graph_outputs, graph_initializers)
    onnx_model = helper.make_model(onnx_graph, producer_name='TASO Optimized Model')
    return onnx_model

def optimize(graph, alpha = 1.0, budget = 1000, print_subst = False):
    return graph.optimize(alpha, budget, print_subst)

# Current TASO Version
__version__ = "0.1.0"
