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
    if typeof(model) <: BatchNorm2d
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
        if length(fields) == 0 && !(typeof(model) <: Flatten)
            return nothing
        end
        for field in fields
            field_var = getfield(model, field)
            if typeof(field_var) <: Array{String, 1} || typeof(field_var) <: Array{UInt8, 1}
                return nothing
            end
            if typeof(field_var) <: Array 
                for idx in 1:length(field_var)
                    new_field_var = _to(field_var[idx], dtype)
                    if new_field_var !== nothing
                        field_var[idx] = new_field_var
                    end
                end
                setproperty!(model, field, field_var)
            else
                new_field_var = _to(field_var, dtype)
                if new_field_var !== nothing
                    setproperty!(model, field, new_field_var)
                end
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
        if length(fields) == 0 && !(typeof(model) <: Flatten)
            return nothing
        end
        for field in fields
            field_var = getfield(model, field)
            if typeof(field_var) <: Array{String, 1} || typeof(field_var) <: Array{UInt8, 1}
                return nothing
            end
            if typeof(field_var) <: Array 
                for idx in 1:length(field_var)
                    new_field_var = _set_mode(field_var[idx], mode)
                    if new_field_var !== nothing
                        field_var[idx] = new_field_var
                    end
                end
                setproperty!(model, field, field_var)
            else
                new_field_var = _set_mode(field_var, mode)
                if new_field_var !== nothing
                    setproperty!(model, field, new_field_var)
                end
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


"""
Has to be used before changing the optimizer or the learning rate. In Knet, once 
the optimizer is called with a learning rate, then that optimizer is assigned to 
all the Params of the network. If an optimizer is assigned to a Param, then calling
the model with a different optimizer does not change anything. This methods removes
all the optimizer assignments to ach of the network Params.

# Keywords:

- model (a network in training mode): the model to reset its optimizer information
"""
function zero_grad(model)
    for p in params(model)
        if p.opt !== nothing
            p.opt = nothing
        end
    end
    return model
end