using MAT
using Knet
include("../models/vgg.jl")

function convert_params(style, value)
    if style == "conv_weight"
        return Param(reverse(reverse(permutedims(value, (4, 3, 2, 1)), dims=1), dims=2))
    elseif style == "fc_weight"
        return Param(value)
    elseif style == "conv_bias"
        return Param(reshape(value, (1, 1, length(value), 1)))
    elseif style == "fc_bias"
        return Param(reshape(value, (length(value), 1)))
    elseif style == "bn_moments"
        value1 = vec(value[1])
        value2 = vec(value[2])
        return bnmoments(mean=reshape(value1, (1, 1, length(value1), 1)), var=reshape(value2, (1, 1, length(value2), 1)))
    elseif style == "bn_params"
        value1 = vec(value[1])
        value2 = vec(value[2])
        return Param(vcat(value1, value2))
    end
end

"""
Inject weights to VGG11 from weights dict extracted from .pth file. 

Sample weigth extraction code with python:

```
def load_pth():
    loaded = torch.load("vgg11-bbd30ac9.pth")
    w_dict = { }
    for k, v in loaded.items():
        cpu_val = v.cpu().detach().numpy()
        w_dict[k] = cpu_val
    savemat("vgg11_weights.mat", w_dict)
```

# Keywords

- model (VGG): model structre to be loaded.
- dict_path (String): path of the vgg11_weights.mat file.

"""
function load_vgg11_from_pytorch_dict!(model::VGG, dict_path::String)
    dict = matread(dict_path)
    # handling pred layer
    model.classifier.layers[2].w = convert_params("fc_weight", dict["classifier.0.weight"])
    model.classifier.layers[2].b = convert_params("fc_bias", dict["classifier.0.bias"])
    model.classifier.layers[5].w = convert_params("fc_weight", dict["classifier.3.weight"])
    model.classifier.layers[5].b = convert_params("fc_bias", dict["classifier.3.bias"])
    model.classifier.layers[8].w = convert_params("fc_weight", dict["classifier.6.weight"])
    model.classifier.layers[8].b = convert_params("fc_bias", dict["classifier.6.bias"])
    # handling feature layers
    model.features.layers[1].w = convert_params("conv_weight", dict["features.0.weight"])
    model.features.layers[1].b = convert_params("conv_bias", dict["features.0.bias"])
    model.features.layers[4].w = convert_params("conv_weight", dict["features.3.weight"])
    model.features.layers[4].b = convert_params("conv_bias", dict["features.3.bias"])
    model.features.layers[7].w = convert_params("conv_weight", dict["features.6.weight"])
    model.features.layers[7].b = convert_params("conv_bias", dict["features.6.bias"])
    model.features.layers[9].w = convert_params("conv_weight", dict["features.8.weight"])
    model.features.layers[9].b = convert_params("conv_bias", dict["features.8.bias"])
    model.features.layers[12].w = convert_params("conv_weight", dict["features.11.weight"])
    model.features.layers[12].b = convert_params("conv_bias", dict["features.11.bias"])
    model.features.layers[14].w = convert_params("conv_weight", dict["features.13.weight"])
    model.features.layers[14].b = convert_params("conv_bias", dict["features.13.bias"])
    model.features.layers[17].w = convert_params("conv_weight", dict["features.16.weight"])
    model.features.layers[17].b = convert_params("conv_bias", dict["features.16.bias"])
    model.features.layers[19].w =  convert_params("conv_weight", dict["features.18.weight"])
    model.features.layers[19].b = convert_params("conv_bias", dict["features.18.bias"])
end