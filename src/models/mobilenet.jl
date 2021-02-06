include("../core/blocks.jl")
include("../operations.jl")

"""
MobileNetV2 complete model adopted from PyTorch source code. If pretrained parameter is set to true, please make 
sure that the weight file is downloaded and placed to the directory "DeepJulia/weights/". The download link to 
the weights can be found at: https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing

# Keywords:

- num_classes (int): Number of classes
- width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
- residual_setting: Network structure
- round_nearest (int): Round the number of channels in each layer to be a multiple of this number. Set to 1 to turn off rounding
- pretrained (Bool, optional): set true to load the ImageNet-pretrained model. Default: false
- include_top (Bool, optional): set false to discard the final prediction Linear layer. Default: true

Note: Please scale images to [0,1] range, substract the means (0.406, 0.456, 0.485) and divide to stds (0.224, 0.225, 0.229) 
before feeding into this model. Also, the model only works on GPU currently. Please do not use it on CPU.
"""
mutable struct MobileNetV2
    chain::Sequential; pred_module;

    function MobileNetV2(; 
        num_classes=1000, width_mult=1.0, round_nearest=8, 
        pretrained::Bool=false, include_top::Bool=true, residual_setting=nothing)
        
        if pretrained
            model = load_model("./weights/mobilenetv2.jld2")
            if !include_top
                model.pred_module = nothing
            end
            return model
        end
        
        if residual_setting === nothing || size(residual_setting, 2) != 4
            residual_setting = [ # t, c, n, s
                1  16 1 1; 
                6  24 2 2;
                6  32 3 2;
                6  64 4 2;
                6  96 3 1;
                6 160 3 2;
                6 320 1 1;
            ]
        end  
        
        input_channel = _make_divisible(32 * width_mult, round_nearest) 
        last_channel = _make_divisible(1280 * max(1.0, width_mult), round_nearest)

        features = []
        push!(features, ConvBnRelu(3, input_channel, 3, stride=2, padding=1, leaky=0, bias=false))
        
        for i in 1:size(residual_setting, 1)
            t, c, n, s = residual_setting[i,:]'
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in 1:n
                stride = i == 1 ? s : 1
                push!(features, InvertedResidual(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
            end
        end
        # building last several layers
        push!(features, ConvBnRelu(input_channel, last_channel, 1, leaky=0))
           
        chain = Sequential(features)
        pred_module = include_top ? Sequential([
            GlobalPool(),
            Flatten(),
            Dropout(0.2),
            Linear(last_channel, num_classes)
        ]) : nothing
        
        return new(chain, pred_module)
    end

end

function (mv2::MobileNetV2)(x)
    if !(typeof(x) <: KnetArray)
        println("[ERROR] MobileNet does not work in CPU! Please move your model and input to GPU and declare as KnetArrays!")
        return nothing
    end
    x_val = mv2.chain(x)
    if mv2.pred_module !== nothing
        return mv2.pred_module(x_val)
    else
        return x_val
    end
end

"""
Inverted Residual Block used specifically for MobileNet models. Implementation is directly adopted from
PyTorch source code. Please check it for any extra information.
"""
mutable struct InvertedResidual
    chain::Sequential; use_res_connect::Bool; out_channels::Int64; _is_cn::Bool;

    function InvertedResidual(inp, oup; stride=1, expand_ratio=1, use_residual=true, last_leaky=nothing)
        hidden_dim = round(Int64, inp * expand_ratio)
        use_res_connect = stride == 1 && inp == oup && use_residual
        hidden_leaky = (last_leaky !== nothing && last_leaky > 0) ? last_leaky : 0
        layers = []
        
        if expand_ratio != 1 # converting the out dimension to the wanted format
            push!(layers, ConvBnRelu(inp, hidden_dim, 1, leaky=hidden_leaky, bias=false))
        end

        push!(layers, ConvBnRelu(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, leaky=hidden_leaky, bias=false))
        push!(layers, Conv2d(hidden_dim, oup, 1, bias=false))
        push!(layers, BatchNorm2d(oup))
        if last_leaky !== nothing
            push!(layers, Relu(leaky=last_leaky))
        end
        _is_cn = stride > 1

        return new(Sequential(layers), use_res_connect, oup, _is_cn)
    end
end

function (ir::InvertedResidual)(x)
    if ir.use_res_connect
        return x + ir.chain(x)
    else
        return ir.chain(x)
    end
end

# HELPER METHODS

function _make_divisible(v, divisor; min_value=8)
    if min_value === nothing
        min_value = divisor
    end
    new_v = max(min_value, floor(Int, floor(Int, v + divisor / 2) / divisor) * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v
        new_v += divisor
    end
    return new_v
end