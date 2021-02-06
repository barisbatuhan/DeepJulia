include("./layers.jl")
include("./initializers.jl")

"""
An intermediate block combining Convolution + Batch Normalization + Relu. If leaky
parameter is set to nothing, then the Relu operation is ignored.

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
- momentum (float, optional): the value used for the running_mean and running_var computation. Default: 0.1
- eps (float, optional): a value added to the denominator for numerical stability. Default: 1e-5
- leaky (float, optional): if given 0, then ReLU is applied, given otherwise, Leaky ReLU is applied. Default: "0"
"""
mutable struct ConvBnRelu
    conv; bn; relu;

    function ConvBnRelu(
        in_channels::Int, 
        out_channels::Int, 
        kernel_size::Union{Tuple, Int}; 
        padding::Int=0, 
        stride::Int=1,
        dilation::Int=1,
        groups::Int=1,
        bias::Bool=true,
        init=kaiming,
        momentum=0.1,
        eps=1e-5,
        leaky=0,
        flip_kernel::Bool=true)
    
        activ = leaky === nothing ? nothing : Relu(leaky=leaky)
        
        return new(
            Conv2d(
                in_channels, out_channels, kernel_size, padding=padding,
                stride=stride, dilation=dilation, groups=groups, bias=bias,
                init=init, flip_kernel=flip_kernel
            ),
            BatchNorm2d(out_channels, momentum=momentum, eps=eps),
            activ
        )
    end
end

function (cbr::ConvBnRelu)(x)
    x_val = cbr.conv(x)
    x_val = cbr.bn(x_val)
    if cbr.relu === nothing
        return x_val
    else
        return cbr.relu(x_val)
    end
end


"""
A residual block which may have 2 arms: downsample path and residual path.

# Keywords:

- residual (a layer or model): the model or layer that the input will be passed as residual block.
- downsample (a layer or model, optional): if downsampling is required, then the downsampling layer. Default: "nothing"
- leaky (Float32, optional): if it is set to a float, then either ReLU or Leaky ReLU is applie dto final output.
If leaky=nothing is passed to the function, then no activation is performed. Default: "0"
"""
mutable struct ResidualBlock
    downsample; residual; activ;

    function ResidualBlock(residual; downsample=nothing, leaky=0)
        activ = leaky !== nothing ? Relu(leaky=leaky) : nothing
        return new(downsample, residual, activ)
    end
end

function (rb::ResidualBlock)(x)
    x_val = rb.residual(x)
    
    if rb.downsample === nothing
        x_val = x + x_val
    else
        x_val_down = rb.downsample(x)
        x_val = x_val_down + x_val
    end

    if rb.activ !== nothing
        return rb.activ(x_val)
    else
        return x_val
    end
end