include("../core/blocks.jl")
include("../operations.jl")

"""
Returns the ResNet18 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.

"""
function ResNet18(;pretrained::Bool=false, include_top::Bool=true)
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, include_top)
end

"""
Returns the ResNet34 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.

"""
function ResNet34(;pretrained::Bool=false, include_top::Bool=true)
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, include_top)
end


"""
Returns the ResNet50 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.

"""
function ResNet50(;pretrained::Bool=false, include_top::Bool=true)
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, include_top)
end


"""
Returns the ResNet101 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.

"""
function ResNet101(;pretrained::Bool=false, include_top::Bool=true)
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, include_top)
end


"""
Returns the ResNet152 model. If pretrained parameter is set to true, please make sure that the weight
file is downloaded and placed to the directory "DeepJulia/weights/". The download link to the weights
can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds
(0.224, 0.225, 0.229) before feeding into this model.

"""
function ResNet152(;pretrained::Bool=false, include_top::Bool=true)
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, include_top)
end


"""
Bottleneck Structure required for larger ResNet structures (>= 50). For detailed information, please 
visit the PyTorch ResNet source code.
"""
mutable struct Bottleneck
    conv1; bn1; conv2; bn2; conv3; bn3; activ; downsample; expansion; stride;

    function Bottleneck(inplanes::Int, planes::Int; stride::Int=1, 
        downsample=nothing, groups::Int=1, base_width::Int=64, dilation::Int=1)

        expansion = 4
        width = floor(Int, planes * (base_width / 64)) * groups
        conv1 = Conv2d(inplanes, width, 1, bias=false)
        bn1 = BatchNorm2d(width)
        conv2 = Conv2d(width, width, 3, stride=stride, groups=groups, padding=1, dilation=dilation, bias=false)
        bn2 = BatchNorm2d(width)
        conv3 = Conv2d(width, planes * expansion, 1, bias=false)
        bn3 = BatchNorm2d(planes * expansion)
        activ = Relu()

        return new(conv1, bn1, conv2, bn2, conv3, bn3, activ, downsample, expansion, stride)
    end
end

function (bot::Bottleneck)(x)
    x_val = bot.activ(bot.bn1(bot.conv1(x))) # First ConvBnRelu block
    x_val = bot.activ(bot.bn2(bot.conv2(x_val))) # Second ConvBnRelu block
    x_val = bot.bn3(bot.conv3(x_val)) # Third ConvBn block

    if bot.downsample === nothing
        return bot.activ(x + x_val)
    else
        return bot.activ(bot.downsample(x) + x_val)
    end
end


"""
BasicBlock Structure required for small ResNet structures (<= 34). For detailed information, please 
visit the PyTorch ResNet source code.
"""
mutable struct BasicBlock
    conv1; bn1; activ; conv2; bn2; downsample; expansion; stride;
    
    function BasicBlock(inplanes::Int, planes::Int; stride::Int=1, 
        downsample=nothing, groups::Int=1, base_width::Int=64, dilation::Int=1)
        
        if groups != 1 || base_width != 64 || dilation > 1
            println("[ERROR] BasicBlock only supports groups=1, dilation=1 and base_width=64")
            return nothing
        end

        conv1 = Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=false)
        bn1 = BatchNorm2d(planes)
        activ = Relu()
        conv2 = Conv2d(planes, planes, 3, padding=1, bias=false)
        bn2 = BatchNorm2d(planes)

        return new(conv1, bn1, activ, conv2, bn2, downsample, 1, stride)
    end
end

function (bb::BasicBlock)(x)
    x_val = bb.activ(bb.bn1(bb.conv1(x))) # First ConvBnRelu block
    x_val = bb.bn2(bb.conv2(x_val)) # Second ConvBn block

    if bb.downsample === nothing
        return bb.activ(x + x_val)
    else
        
        return bb.activ(bb.downsample(x) + x_val)
    end
end


"""
The common ResNet struct, which is used in all of the ResNet modules. For detailed information, please 
visit the PyTorch ResNet source code. To define a custom ResNet module, this struct may be used.
"""
mutable struct ResNet
    init_layer; layer1; layer2; layer3; layer4; pred_layer; include_top;

    function ResNet(block::DataType, layers::Array{Int}; num_classes=1000, 
        groups::Int=1, width_per_group::Int=64, replace_stride_with_dilation=nothing, include_top=true)
        
        values = Dict(:dilation => 1, :inplanes => 64, :groups => groups, :base_width => width_per_group)
        
        if replace_stride_with_dilation === nothing
            replace_stride_with_dilation = [false, false, false]
        elseif length(replace_stride_with_dilation) != 3
            println("[ERROR] Replace_stride_with_dilation should be None or a 3-element tuple")
            return nothing
        elseif !(block <: Bottleneck || block <: BasicBlock)
            println("[ERROR] The input block has to be an instance of Bottleneck or BasicBlock.")
            return nothing
        end

        expansion = block <: Bottleneck ? 4 : 1
        
        init_layer = Sequential([
            Conv2d(3, values[:inplanes], 7, stride=2, padding=3, bias=false),
            BatchNorm2d(values[:inplanes]),
            Relu(),
            Pool(window=3, padding=1, stride=2)
        ])
        layer1 = _make_layer(block, 64, layers[1], values) 
        layer2 = _make_layer(block, 128, layers[2], values, stride=2, dilate=replace_stride_with_dilation[1]) 
        layer3 = _make_layer(block, 256, layers[3], values, stride=2, dilate=replace_stride_with_dilation[2]) 
        layer4 = _make_layer(block, 512, layers[4], values, stride=2, dilate=replace_stride_with_dilation[3]) 
        
        pred_layer = nothing;
        if include_top
            pred_layer = Sequential([
                GlobalPool(),
                Flatten(),
                Linear(512 * expansion, num_classes)
            ])
        end
        return new(init_layer, layer1, layer2, layer3, layer4, pred_layer, include_top)
    end
end

function (rn::ResNet)(x; return_intermediate=false)
    init_out = rn.init_layer(x)
    l1 = rn.layer1(init_out)
    l2 = rn.layer2(l1)
    l3 = rn.layer3(l2)
    l4 = rn.layer4(l3)
    if rn.include_top
        out = rn.pred_layer(l4)
        if return_intermediate
            return l1, l2, l3, l4, out
        else
            return out
        end
    else
        if return_intermediate
            return l1, l2, l3, l4
        else
            return l4
        end
    end
end

function _make_layer(block::DataType, planes::Int, blocks::Int, values::Dict; stride::Int=1, dilate::Bool=false)

    downsample = nothing
    previous_dilation = deepcopy(values[:dilation])
    expansion = block <: Bottleneck ? 4 : 1

    if dilate
        values[:dilation] *= stride
        stride = 1
    end
    if stride != 1 || values[:inplanes] != planes * expansion
        downsample = Sequential([
            Conv2d(values[:inplanes], planes * expansion, 1, stride=stride, bias=false),
            BatchNorm2d(planes * expansion)
        ])
    end

    layers = []
    push!(layers, block(
        values[:inplanes], planes, stride=stride, downsample=downsample, groups=values[:groups],
        base_width = values[:base_width], dilation=previous_dilation
    ))
    values[:inplanes] = planes * expansion

    for _ in 2:blocks
        push!(layers, block(
            values[:inplanes], planes, groups=values[:groups],
            base_width = values[:base_width], dilation=values[:dilation]
        ))
    end

    return Sequential(layers)
end


function _resnet(arch::String, block::DataType, layers::Array{Int}, pretrained::Bool, include_top::Bool)
    if pretrained
        model = load_model("../weights/" * arch * ".jld2")
        if !include_top
            model.include_top = include_top
            model.pred_layer = nothing
        end
        return model
    else
        return ResNet(block, layers, include_top=include_top)
    end
end

