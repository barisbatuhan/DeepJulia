include("../models/mobilenet.jl")
include("../operations.jl")

function convert_params(style, value)
    atype = KnetArray{Float32}
    if style == "conv_weight"
        value = value[:data].tolist()
        return Param(convert(atype, reverse(reverse(permutedims(value, (4, 3, 2, 1)), dims=1), dims=2)))
    elseif style == "fc_weight"
        value = convert(atype, value[:data].tolist())
        return Param(value)
    elseif style == "conv_bias"
        value = convert(atype, value[:data].tolist())
        return Param(reshape(value, (1, 1, length(value), 1)))
    elseif style == "fc_bias"
        value = convert(atype, value[:data].tolist())
        return Param(reshape(value, (length(value), 1)))
    elseif style == "bn_moments"
        value1 = convert(atype, vec(value[1][:data].tolist()))
        value2 = convert(atype, vec(value[2][:data].tolist()))
        return bnmoments(mean=reshape(value1, (1, 1, length(value1), 1)), var=reshape(value2, (1, 1, length(value2), 1)))
    elseif style == "bn_params"
        value1 = convert(atype, vec(value[1][:data].tolist()))
        value2 = convert(atype, vec(value[2][:data].tolist()))
        return Param(vcat(value1, value2))
    end
end

function load_mobilenet_from_pytorch(model, t_dict::Dict)
    # loading fc layer at the end
    model.pred_module.layers[4].w = convert_params("fc_weight", t_dict["classifier.1.weight"])
    model.pred_module.layers[4].b = convert_params("fc_bias", t_dict["classifier.1.bias"])
    # loading the initial ConvBnRelu layer
    model.chain.layers[1].conv.w = convert_params("conv_weight", t_dict["features.0.0.weight"])
    model.chain.layers[1].bn.bnparams = convert_params("bn_params", [t_dict["features.0.1.weight"], t_dict["features.0.1.bias"]])
    model.chain.layers[1].bn.bnmoments = convert_params("bn_moments", [t_dict["features.0.1.running_mean"], t_dict["features.0.1.running_var"]])
    # loading the last ConvBnRelu layer
    model.chain.layers[19].conv.w = convert_params("conv_weight", t_dict["features.18.0.weight"])
    model.chain.layers[19].bn.bnparams = convert_params("bn_params", [t_dict["features.18.1.weight"], t_dict["features.18.1.bias"]])
    model.chain.layers[19].bn.bnmoments = convert_params("bn_moments", [t_dict["features.18.1.running_mean"], t_dict["features.18.1.running_var"]])
    # loading inverted residual layers
    for k in sort([keys(t_dict)...])
        if !occursin("conv", k); continue; end;
        dot2 = findfirst(".", k[10:end])[1] + 9; dot3 = findfirst(".", k[dot2+1:end])[1] + dot2;
        feature_idx = parse(Int64, k[10:dot2-1]) + 1
        layer_idx = parse(Int64, k[dot3+1]) + 1
        
        # println(k)
        
        if occursin(r"[0-9]{1}[.]{1}0[.]{1}weight", k)
            setproperty!(model.chain.layers[feature_idx].chain.layers[layer_idx].conv, :w, convert_params("conv_weight", t_dict[k]))
        
        elseif occursin(r"[0-9]{1}[.]{1}[0-9]{1}[.]{1}weight", k)
            setproperty!(model.chain.layers[feature_idx].chain.layers[layer_idx].bn, :bnparams, convert_params("bn_params", [t_dict[k], t_dict[k[1:end-6] * "bias"]]))
        
        elseif occursin(r"[a-z]{1}[.]{1}[0-9]{1}[.]{1}weight", k)
            if length(size(t_dict[k][:data].tolist())) > 2
                setproperty!(model.chain.layers[feature_idx].chain.layers[layer_idx], :w, convert_params("conv_weight", t_dict[k]))
            else
                setproperty!(model.chain.layers[feature_idx].chain.layers[layer_idx], :bnparams, convert_params("bn_params", [t_dict[k], t_dict[k[1:end-6] * "bias"]]))
            end
        
        elseif occursin(r"[A-Za-z]{1}[.]{1}[0-9]{1}[.]{1}running_mean", k)
            setproperty!(model.chain.layers[feature_idx].chain.layers[layer_idx], :bnmoments, convert_params("bn_moments", [t_dict[k], t_dict[k[1:end-4] * "var"]]))
            
        elseif occursin("running_mean", k)
            setproperty!(model.chain.layers[feature_idx].chain.layers[layer_idx].bn, :bnmoments, convert_params("bn_moments", [t_dict[k], t_dict[k[1:end-4] * "var"]]))
        end   
    end
    return model
end

model = MobileNetV2()
model = load_mobilenet_from_pytorch(model, d)
model = to_gpu(model)
model = set_eval_mode(model)
print("Model loaded!")