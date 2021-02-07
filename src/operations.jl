using Knet

include("core/layers.jl")

"""
Moves the input to the GPU device.

# Keywords:
- model: the model to move 
- dtype (optional): the data type where all weights and biased will be reconstructed. Default: KnetArray{Float32}

Note: It can be used to move the models only. Please convert the inputs manually by convert(KnetArray{Float32}, input).
"""
function to_gpu(model; dtype=KnetArray{Float32})
    return _to(model, dtype)
end

"""
Moves the model to the CPU device.

# Keywords:
- model: the model to move 
- dtype (optional): the data type where all weights and biased will be reconstructed. Default: Array{Float32}

Note: It can be used to move the models only. Please convert the inputs manually by convert(Array, input).
"""
function to_cpu(model; dtype=Array{Float32})
    return _to(model, dtype)
end

function _to(model, dtype)
    if model <: Array || model <: KnetArray
        for i in 1:length(model)
            model[i] = _to(model[i], dtype)
        end
    elseif typeof(model) <: BatchNorm2d
        model.bnparams = Param(convert(dtype, value(model.bnparams)))
        if model.bnmoments.mean !== nothing
            model.bnmoments = bnmoments(
                mean=convert(dtype, model.bnmoments.mean), 
                var=convert(dtype, model.bnmoments.var)
                )
        end
    elseif typeof(model) <: Sequential
        for i in 1:length(model.layers)
            model.layers[i] = _to(model.layers[i], dtype)
        end
    elseif typeof(model) <: Union{Conv2d, Linear}
        model.w = Param(convert(dtype, value(model.w)))
        model.b = model.b === nothing ? nothing : Param(convert(dtype, value(model.b)))
    else
        fields = fieldnames(typeof(model))
        for field in fields
            if typeof(typeof(getfield(model, field))) <: DataType
                setproperty!(model, field, _to(getfield(model, field), dtype))
            end
        end
    end
    return model
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
