using Knet

include("./initializers.jl")

"""
A struct representation of the Seuqential layer, which is used to stack different layers 
one after another

# Keywords:
- layers (array): Array of layers to stack on top of each other

"""
mutable struct Sequential; layers::Array; end
function (c::Sequential)(x) 
    x_val = c.layers[1](x)
    for l in c.layers[2:end]
        x_val = l(x_val) 
    end
    return x_val
end


"""
A struct representation of the linear operation. Constructor is designed just like PyTorch.

# Keywords:

- in_features (int): size of each input sample
- out_features (int): size of each output sample
- bias (bool, optional): If "true", adds a learnable bias to the output. Default: "true"
- init (optional): Weight initialization structure, default is kaiming initialization.

"""
mutable struct Linear 
    w; b; train_mode;

    function Linear(in_features::Int, out_features::Int; bias::Bool=true, init=xavier)
        w = Param(init(out_features, in_features))
        b = bias == false ? nothing : Param(zeros(out_features, 1))
        return new(w, b, true)
    end
end

function (l::Linear)(x)
    if l.train_mode
        x_val = l.w * mat(x)
        x_val = l.b === nothing ? x_val : x_val .+ l.b
        return x_val
    else
        x_val = value(l.w) * mat(x)
        x_val = l.b === nothing ? x_val : x_val .+ value(l.b)
        return x_val
    end
end

"""
A struct representation of the convolution operation. Constructor is designed just like PyTorch.

# Keywords:

- in_channels (int): Number of channels in the input image
- out_channels (int): Number of channels produced by the convolution
- kernel_size (int or tuple): Size of the convolving kernel
- stride (int or tuple, optional): Stride of the convolution. Default: 1
- padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
- dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
- groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
- bias (bool, optional): If "true", adds a learnable bias to the output. Default: "true"
- init (optional): Weight initialization structure, default is kaiming initialization.
- flip_kernel (bool, optional): true for normal convolution, false for cross correlation. Default: "true"

"""
mutable struct Conv2d
    w; b; padding; stride; dilation; groups; flip_kernel; train_mode;

    function Conv2d(
        in_channels::Int, 
        out_channels::Int, 
        kernel_size::Union{Tuple, Int}; 
        padding::Int=0, 
        stride::Int=1,
        dilation::Int=1,
        groups::Int=1,
        bias::Bool=true,
        init=kaiming,
        flip_kernel::Bool=true 
        )

        if typeof(kernel_size) <: Tuple && length(kernel_size) != 2
            println("[ERROR] Given kernel size tuple has a length more than 2!")
            return nothing
        elseif !(typeof(kernel_size) <: Tuple || typeof(kernel_size) <: Int)
            println("[ERROR] Given kernel size parameter is not a tuple or integer!")
            return nothing
        elseif mod(in_channels, groups) != 0 || mod(out_channels, groups) != 0
            println("[ERROR] Group numbers in convolution is not divisible to either input or output dimension!")
            return nothing
        else
            kernel_size = typeof(kernel_size) <: Int ? (kernel_size, kernel_size) : kernel_size
        end

        in_dims = floor(Int64, in_channels / groups) # needed for grouped convolutions
        flip_mode = flip_kernel == true ? 0 : 1

        w = Param(init(kernel_size[1], kernel_size[2], in_dims, out_channels))
        b = bias == false ? nothing : Param(zeros(1, 1, out_channels, 1))

        return new(w, b, padding, stride, dilation, groups, flip_mode, true)
    end
end

function (conv::Conv2d)(x)
    if conv.train_mode
        x_val = conv4(conv.w, x, padding=conv.padding, stride=conv.stride, 
            dilation=conv.dilation, mode=conv.flip_kernel, group=conv.groups)
        if conv.b !== nothing
            return x_val .+ conv.b
        else
            return x_val
        end
    else
        x_val = conv4(value(conv.w), x, padding=conv.padding, stride=conv.stride, 
            dilation=conv.dilation, mode=conv.flip_kernel, group=conv.groups)
        if conv.b !== nothing
            return x_val .+ value(conv.b)
        else
            return x_val
        end
    end
end


"""
A struct representation for 2D Batch Normalization.

# Keywords:
- num_features (int): expected channel size of the input (C in WxHxCxN sized input).
- momentum (float, optional): the value used for the running_mean and running_var computation. Default: 0.1
- eps (float, optional): a value added to the denominator for numerical stability. Default: 1e-5
"""
mutable struct BatchNorm2d
    bnmoments; bnparams; train_mode; eps;

    function BatchNorm2d(num_features; momentum=0.1, eps=1e-5)
        mom = bnmoments(momentum=momentum)
        par = Param(bnparams(Float32, num_features))
        return new(mom, par, true, eps)
    end
end

function (bn::BatchNorm2d)(x)
    if bn.train_mode
        return batchnorm(x, bn.bnmoments, bn.bnparams, eps=bn.eps, training=bn.train_mode)
    else
        return batchnorm(x, bn.bnmoments, value(bn.bnparams), eps=bn.eps, training=bn.train_mode)
    end
end


"""
A struct representation for Dropout operation. Only applied in training mode.

# Keywords:
- dropout_prob (float): probability to zero an element. Please set between 0 and 0.5.
"""
mutable struct Dropout 
    prob; train_mode;

    function Dropout(dropout_prob)
        return new(dropout_prob, true)
    end
end

function (d::Dropout)(x)
    if d.train_mode
        return dropout(x, d.prob)
    else
        return x
    end
end


"""
Adaptive Pooling layer for 4-dimensional inputs.

# Keywords

- output_size (Tuple): desired output size after pooling operation.
- mode (Int, optional): "0" for max pooling, "1" for average by also including padding. Default: "1"

"""
mutable struct AdaptivePool 
    output_size::Union{Int, Tuple}
    mode::Int

    function AdaptivePool(output_size::Union{Int, Tuple{Int, Int}}; mode::Int=1)
        if typeof(output_size) <: Int
            output_size = (output_size, output_size)
        end
        return new(output_size, mode)
    end
end

function (adaptive_pool::AdaptivePool)(x)
    input_size = size(x)[1:2]
    stride = Int.(floor.(input_size ./ adaptive_pool.output_size))
    kernel_size = input_size .- (adaptive_pool.output_size .- 1) .* stride
    pool_result = pool(x; window=kernel_size, stride=stride, padding=0, mode=adaptive_pool.mode)
    return pool_result
end

"""
Pooling layer for 4-dimensional inputs.

# Keywords

- window (Int or Tuple, optional): window size for pooling. Default: "2"
- padding (Int, optional): padding size for pooling. Default: "0"
- stride (Int, Tuple or nothing, optional): stride size for pooling. 
Default is set to window size, if stride=nothing is passed to the constructor.
- mode (Int, optional): "0" for max pooling, "1" for average by also including padding, 
"2" for average pooling without including paddings.

"""
mutable struct Pool
    window::Union{Int, Tuple}; padding::Int; stride::Int; mode::Int;

    function Pool(;window::Union{Int, Tuple{Int, Int}}=2, padding::Int=0, stride=nothing, mode::Int=0)
        stride = stride === nothing ? window : stride
        return new(window, padding, stride, mode)
    end
end

function (po::Pool)(x) 
    return pool(x, window=po.window, padding=po.padding, stride=po.stride, mode=po.mode)
end


"""
A layer for flattening an input before putting into a linear layer.
"""
mutable struct Flatten; end;
(fl::Flatten)(x) = (mat(x))


"""
A struct representation for the ReLU and Leaky ReLU activations.

# Keywords:
- leaky (float, optional): if given 0, then ReLU is applied, given otherwise, Leaky ReLU is applied. Default: "0"
"""
mutable struct Relu
    leaky

    function Relu(;leaky=0)
        return new(leaky)
    end
end

function (rl::Relu)(x)
    if rl.leaky == 0
        return relu.(x)
    else
        return max(rl.leaky .* x, x)
    end
end



