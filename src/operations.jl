using Knet

include("core/layers.jl")

"""
Moves the input to the GPU device.

# Keywords:
- input: the input to move 
- dtype (optional): the data type where all weights and biased will be reconstructed. Default: KnetArray{Float32}

Note: It can be used to move the models and inputs in Array{Union{Int, Float32, Float64}} format. For more
complicated inputs, please convert them to KnetArrays manually.
"""
function to_gpu(input; dtype=KnetArray{Float32})
    return _to(input, dtype)
end

"""
Moves the input to the CPU device.

# Keywords:
- input: the input to move 
- dtype (optional): the data type where all weights and biased will be reconstructed. Default: Array{Float32}

Note: It can be used to move the models and inputs in KnetArray{Union{Int, Float32, Float64}} format. For more
complicated inputs, please convert them to KnetArrays manually.
"""
function to_cpu(input; dtype=Array{Float32})
    return _to(input, dtype)
end

function _to(input, dtype)
    if input <: Array || input <: KnetArray
        return convert(dtype, input)
    elseif typeof(input) <: BatchNorm2d
        input.bnparams = Param(convert(dtype, value(input.bnparams)))
        if input.bnmoments.mean !== nothing
            input.bnmoments = bnmoments(
                mean=convert(dtype, input.bnmoments.mean), 
                var=convert(dtype, input.bnmoments.var)
                )
        end
    elseif typeof(input) <: Sequential
        for i in 1:length(input.layers)
            input.layers[i] = _to(input.layers[i], dtype)
        end
    elseif typeof(input) <: Union{Conv2d, Linear}
        input.w = Param(convert(dtype, value(input.w)))
        input.b = input.b === nothing ? nothing : Param(convert(dtype, value(input.b)))
    else
        fields = fieldnames(typeof(input))
        for field in fields
            if typeof(typeof(getfield(input, field))) <: DataType
                setproperty!(input, field, _to(getfield(input, field), dtype))
            end
        end
    end
    return input
end

"""
Enables gradient storage and puts the model to training mode.

# Keywords:
- model: the model to move 
"""
function set_train_mode(model)
    return _set_mode(model, true)
end


"""
Disables gradient storage and puts the model to evaluation mode.

# Keywords:
- model: the model to move 
"""
function set_eval_mode(model)
    return _set_mode(model, false)
end

function _set_mode(model, mode::Bool)
    if typeof(model) <: Sequential
        for i in 1:length(model.layers)
            model.layers[i] = _set_mode(model.layers[i], mode)
        end
    elseif :train_mode in fieldnames(typeof(model))
        model.train_mode = mode
    else
        fields = fieldnames(typeof(model))
        for field in fields
            if typeof(typeof(getfield(model, field))) <: DataType
                setproperty!(model, field, _set_mode(getfield(model, field), mode))
            end
        end
    end
    return model
end

"""
For loading a model. Please include the model code before use this method.
"""
function load_model(file_name)
    return Knet.load(file_name, "model")
end

"""
For saving a model.
"""
function save_model(model, file_name)
    Knet.save(file_name, "model", model)
end
