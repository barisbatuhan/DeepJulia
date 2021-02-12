include("../core/blocks.jl")
include("../operations.jl")

# TODO: ADD PRETRAINED FOR VGG's
# DISCUSS substraction of means and division of stds, PYTORCH also uses RGB,  mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

"""
Returns the VGG11 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true
- batchnorm (Bool, optional): set true to have BatchNorm2d in convolutional layers of model. Default: false

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.
"""
function VGG11(; pretrained::Bool=false, 
    include_top::Bool=true,
    batchnorm::Bool=false,
    flip_conv_kernels::Bool=false)::VGG
    return _vgg(VGGType11, pretrained, include_top, batchnorm, flip_conv_kernels)
end

"""
Returns the VGG13 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true
- batchnorm (Bool, optional): set true to have BatchNorm2d in convolutional layers of model. Default: false

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.
"""
function VGG13(; pretrained::Bool=false,
     include_top::Bool=true,
     batchnorm::Bool=false,
     flip_conv_kernels::Bool=false)::VGG
    return _vgg(VGGType13, pretrained, include_top, batchnorm, flip_conv_kernels)
end


"""
Returns the VGG16 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true
- batchnorm (Bool, optional): set true to have BatchNorm2d in convolutional layers of model. Default: false

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.
"""
function VGG16(; pretrained::Bool=false,
    include_top::Bool=true,
    batchnorm::Bool=false,
    flip_conv_kernels::Bool=false)::VGG
    return _vgg(VGGType16, pretrained, include_top, batchnorm, flip_conv_kernels)
end


"""
Returns the VGG19 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true
- batchnorm (Bool, optional): set true to have BatchNorm2d in convolutional layers of model. Default: false

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.
"""
function VGG19(; pretrained::Bool=false, 
    include_top::Bool=true,
    batchnorm::Bool=false,
    flip_conv_kernels::Bool=false)::VGG
    return _vgg(VGGType19, pretrained, include_top, batchnorm, flip_conv_kernels)
end

@enum VGGType begin
    VGGType11
    VGGType13
    VGGType16
    VGGType19
end

vgg_configs = Dict{VGGType,Array{Union{Integer,Char}}}(
    VGGType11 => [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    VGGType13 => [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    VGGType16 => [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    VGGType19 => [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        )

"""
The common VGG struct, which is used in all of the VGG modules. For detailed information, please 
The common VGG struct, which is used in all of the VGG modules. For detailed information, please 
visit the PyTorch VGG source code. To define a custom VGG module, this struct may be used.

"""
mutable struct VGG
    features::Sequential
    avg_pool::AdaptivePool
    classifier::Sequential
    include_top::Bool

    function VGG(features::Sequential; num_classes=1000, include_top::Bool=true, dropout_prob::Float64=0.5)
        avg_pool = AdaptivePool((7,7); mode=1)
        if include_top
            classifier = Sequential([
                Flatten(),
                Linear(512 * 7 * 7, 4096),
                Relu(),
                Dropout(dropout_prob),
                Linear(4096, 4096),
                Relu(),
                Dropout(dropout_prob),
                Linear(4096, num_classes),
            ])
        else
            classifier = nothing
        end
        return new(features, avg_pool, classifier, include_top)
    end
end

function (vgg::VGG)(x)
    x = vgg.features(x)
    if vgg.include_top
        x = vgg.avg_pool(x)
        x = vgg.classifier(x)
        return x
    else
        return x
    end
end

function _make_layers_vgg(arch::VGGType; batchnorm::Bool=false, flip_conv_kernels::Bool=false)::Sequential
    layers = []
    config::Array{Union{Integer,Char}} = vgg_configs[arch]
    in_channels = 3
    for v in config
        if v == 'M'
            push!(layers, Pool(; window=2, stride=2, mode=0))
        else
            conv2d = Conv2d(in_channels, v, 3; padding=1, flip_kernel=flip_conv_kernels)
            if batchnorm
                push!(layers, [conv2d, BatchNorm2d(v), Relu()]...)
            else
                push!(layers, [conv2d, Relu()]...)
            end
        in_channels = v
        end
    end
    return Sequential(layers)
end

function _vgg(arch::VGGType, pretrained::Bool, include_top::Bool, batchnorm::Bool, flip_conv_kernels::Bool = false)::VGG
    if pretrained
        return throw(AssertionError("PRETRAINED MODEL NOT IMPLEMENTED YET"))
    else
        features =_make_layers_vgg(arch; batchnorm=batchnorm, flip_conv_kernels=flip_conv_kernels)
        return VGG(features; include_top=include_top)
    end
end