include("../models/resnet.jl")
include("../operations.jl")

function convert_params(style, value)
    if style == "conv_weight"
        value = value[:data].tolist()
        return Param(reverse(reverse(permutedims(value, (4, 3, 2, 1)), dims=1), dims=2))
    elseif style == "fc_weight"
        value = value[:data].tolist()
        return Param(value)
    elseif style == "conv_bias"
        value = value[:data].tolist()
        return Param(reshape(value, (1, 1, length(value), 1)))
    elseif style == "fc_bias"
        value = value[:data].tolist()
        return Param(reshape(value, (length(value), 1)))
    elseif style == "bn_moments"
        value1 = vec(value[1][:data].tolist())
        value2 = vec(value[2][:data].tolist())
        return bnmoments(mean=reshape(value1, (1, 1, length(value1), 1)), var=reshape(value2, (1, 1, length(value2), 1)))
    elseif style == "bn_params"
        value1 = vec(value[1][:data].tolist())
        value2 = vec(value[2][:data].tolist())
        return Param(vcat(value1, value2))
    end
end

function load_resnet_from_pytorch(model, t_dict::Dict)
    # restoring fully connected ones
    model.pred_layer.layers[3].w = convert_params("fc_weight", t_dict["fc.weight"])
    model.pred_layer.layers[3].b = convert_params("fc_bias", t_dict["fc.bias"])
    # restoring initial 7x7 block
    model.init_layer.layers[1].w = convert_params("conv_weight", t_dict["conv1.weight"])
    model.init_layer.layers[2].bnparams = convert_params("bn_params", [t_dict["bn1.weight"], t_dict["bn1.bias"]])
    model.init_layer.layers[2].bnmoments = convert_params("bn_moments", [t_dict["bn1.running_mean"], t_dict["bn1.running_var"]])
    # restoring other ones
    for k in sort([keys(t_dict)...])
        if !occursin("layer", k); continue; end;
        firstdot = findfirst(".", k[8:end])[1] + 7;
        idx = parse(Int64, k[8:firstdot-1]) + 1; dotidx = findfirst(".", k[firstdot+1:end])[1];
        layer_sym = Symbol(k[1:6]); sublayer_sym = Symbol(k[firstdot+1:firstdot+dotidx-1]);
        
        # println(k, " ", layer_sym, " ", sublayer_sym)
        
        layer = getfield(model, layer_sym)
        chain = getfield(layer, :layers)
        sublayer = getfield(chain[idx], sublayer_sym)
        
        sym = nothing; val = nothing; param_sym = nothing;
        if occursin("downsample.1", k) || occursin("bn", k)
            if occursin("weight", k)
                sym = :bnparams; param_sym = "bn_params";
                val = [t_dict[k], t_dict[k[1:end-6] * "bias"]]
            elseif occursin("running_mean", k)
                sym = :bnmoments; param_sym = "bn_moments";
                val = [t_dict[k], t_dict[k[1:end-4] * "var"]]
            end
        elseif occursin("downsample.0", k) || occursin("conv", k)
            if occursin("weight", k)
                sym = :w; param_sym = "conv_weight";
                val = t_dict[k]
            elseif occursin("bias", k)
                sym = :b; param_sym = "conv_bias";
                val = t_dict[k]
            end
        end
        
        if val === nothing || param_sym === nothing
            continue
        end
        
        if occursin("downsample", k)
            if param_sym == "conv_weight" 
                sublayer.layers[1].w = convert_params(param_sym, val)
            elseif param_sym == "conv_bias" 
                sublayer.layers[1].b = convert_params(param_sym, val)
            elseif param_sym == "bn_moments" 
                sublayer.layers[2].bnmoments = convert_params(param_sym, val)
            elseif param_sym == "bn_params" 
                sublayer.layers[2].bnparams = convert_params(param_sym, val)
            end   
        else
            setproperty!(sublayer, sym, convert_params(param_sym, val))
        end
        
        setproperty!(chain[idx], sublayer_sym, sublayer)
        setproperty!(layer, :layers, chain)
        setproperty!(model, layer_sym, layer)
    end
    
    return model
end

rn_path = "../../pytorch_weights/resnet152.pth"

d = torch.load(rn_path)

model = ResNet152()
model = load_resnet_from_pytorch(model, d)
model = set_eval_mode(model)
print("Model loaded!")