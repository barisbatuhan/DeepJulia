using Images, Random

"""
Image data augmentor object. 

# Keywords:

- processes(Array of augmenation structs): add the augmentation modes as an array. Please put the
augmentation modes in the order you want to be processed. If only reading the image is aimed, then 
an empty array "[]" can be passed for this parameter.
- img_paths(Array{String}): paths of all of the images in the dataset to iterate. Cannot be empty or nothing.
- labels(Array{Any}, optional): all the labels of the images with the same order in the image_paths array. No changes
are made to these values but the correct orderis returned in batches. Can be set to nothing for not returning the labels.
Default: "nothing"
- batch_size(Int, optional): the batch size of the images to iterate. Default: "1"
- img_size(Tuple or Int, optional): the common sizes of the images in a batch. To read only image in its default size,
please set batch_size=1 and img_size=-1. Default: "224"
- means(Tuple with size 3, optional): The mean to substract from the image. Range is [0, 1].
Default is set to ImageNet mean: "(0.406, 0.456, 0.485)"
- stds(Tuple with size 3, optional): The stds to divide the image. Range is [0, 1].
Default is set to ImageNet std: "(0.224, 0.225, 0.229)"
- shuffle(Bool, optional): set true to shuffle the data. Default: "true"
- return_changes(Bool, optional): if you want to return all the values applied during augmentation,
then set it to true. Default: "false"

Currently, the augmentation processes below are implemented for this process scope:

- RandomCrop: crops the image randomly or from the center
- Squaritize: fills the short side with a given fill value and makesthe image square
- Flip: flips the image horizontally or-and vertically
- Resize: resizes the image to the given shape
- DistortColor: distorts brighness, contrast, saturation and hue values of the image

For detailed information, please check the documentation of the individual pieces separately.
"""
mutable struct Transforms
    processes::Array{Any}; img_paths::Array{String}; labels; state::Array{Int}; img_size; 
    means; stds; batch_size::Int; shuffle::Bool; return_changes::Bool;

    function Transforms(
        processes::Array{Any}, 
        img_paths::Array{String}; 
        labels=nothing,
        batch_size::Int=1,
        img_size::Union{Tuple, Int}=224,
        means::Tuple=(0.485, 0.456, 0.406),
        stds::Tuple=(0.229, 0.224, 0.225),
        shuffle::Bool=true,
        return_changes::Bool=false)

        data_size = length(img_paths); 
        state = shuffle ? randperm(data_size) : collect(1:data_size)

        if labels !== nothing && data_size != size(labels, 1)
            println("[ERROR] Size of images and labels are not the same!")
            return nothing
        elseif img_paths === nothing
            println("[ERROR] No image path is given!")
            return nothing
        elseif img_size == -1 && batch_size != 1
            println("[ERROR] If no image size is set, batch size should be 1!")
            return nothing
        elseif sum(max.(1, means) .> 1) > 1 || sum(min.(0, means) .< 0) > 1
            println("[ERROR] Means must be between 0 and 1!")
            return nothing
        elseif sum(max.(1, stds) .> 1) > 1 || sum(min.(0, stds) .< 0) > 1
            println("[ERROR] Stds must be between 0 and 1!")
            return nothing
        end

        if img_size != -1 && typeof(img_size) <: Int 
            img_size = (img_size, img_size)
        end
        
        return new(
            processes, img_paths, labels, state, img_size, means, stds,
            batch_size, shuffle, return_changes
            )
    end
end

"""
Applies all the transformation processes given in the Transforms object and returns:
transformed image, transformed labels (or nothing if none given), changes_dict (or nothing)

# Keywords:

- tr(Transforms object)
- restart(Bool, optional): set true if you want to start a new epoch with all the data present.
"""
function get_batch(tr::Transforms; restart::Bool=false)
    
    if restart # to reset the batch information
        data_size = length(tr.img_paths)
        tr.state = tr.shuffle ? randperm(data_size) : collect(1:data_size)
    end

    if tr.state === nothing || length(tr.state) < tr.batch_size
        # if no complete batch is remaining
        return nothing, nothing, nothing
    end

    w = nothing; h = nothing; img = nothing;
    if tr.batch_size == 1 && tr.img_size == -1
        img = convert(Array{Float32}, channelview(RGB.(load(tr.img_paths[tr.state[1]]))))
        w = size(img)[end]; h = size(img)[end-1];
    else
        w = tr.img_size[1]; h = tr.img_size[2];
    end

    result = zeros(w, h, 3, tr.batch_size); ctr = 1; changes = [];

    for idx in tr.state[1:tr.batch_size]
        change = Dict()

        if img === nothing
            img = convert(Array{Float32}, channelview(RGB.(load(tr.img_paths[idx]))))
        end

        for process in tr.processes
            img, k, val = process(img)
            if typeof(process) <: Squaritize && tr.batch_size == 1
                changed_h = size(img, 2); changed_w = size(img, 3)
                result = zeros(changed_w, changed_h, 3, tr.batch_size)
            end
            if tr.return_changes; change[k] = val; end;
        end

        img .-= tr.means; img ./= tr.stds; 
        img = permutedims(reverse(img, dims=1), (3, 2, 1))
        result[:,:,:,ctr] .= img
        push!(changes, change)

        ctr += 1
        img = nothing
    end

    sub_labels = tr.labels !== nothing ? tr.labels[tr.state[1:tr.batch_size]] : nothing
    tr.state = tr.state[tr.batch_size+1:end]
    changes = tr.return_changes ? changes : nothing
    
    return result, sub_labels, changes
end

"""
Crops the given image in CxHxW format. Returns the cropped image, key string "crop" to add the 
region of interest [x1, y1, x2, y2] in changes dict and the region of interest value.

# Keywords:

- min_ratio(Float, optional): minimum width ratio to crop from the whole width. Default: "0.3"
- max ratio(Float, optional): maximum width ratio to crop from the whole width (maximum value is 1). Default: "1.0"
- center(Bool, optional): if given true, the cropping is made from the center. Default: "false"
- width_height_ratio(Float, optional): ratio of width/height in the cropped image. Default: "1"
"""
mutable struct RandomCrop
    min_ratio; max_ratio; center; width_height_ratio;

    function RandomCrop(; min_ratio=0.3, max_ratio=1.0, center::Bool=false, width_height_ratio=1)
        if max_ratio > 1 || min_ratio <= 0
            println("[ERROR] Max Ratio cannot be higher than 1 and Min ratio must be bigger than 0.")
            return nothing
        elseif width_height_ratio <= 0
            println("[ERROR] Width / height ratio must be a positive value.")
            return nothing
        end
        return new(min_ratio, max_ratio, center, width_height_ratio)
    end
end

function (rc::RandomCrop)(x)
    c, h, w = size(x)
    height_ratio = rand(rc.min_ratio:0.01:rc.max_ratio)
    weight_ratio = rc.width_height_ratio * height_ratio
    
    if weight_ratio > 1
        println("[ERROR] Weight ratio is more than 1. Please set a more accurate weigh / height ratio!")
        return nothing
    end
    
    w_len = floor(Int, w * weight_ratio); h_len = floor(Int, h * height_ratio);
    
    x1 = rc.center ? max(1, floor(Int, w/2) - floor(Int, w_len/2)) : rand(1:1:max(1,w-w_len))
    y1 = rc.center ? max(1, floor(Int, h/2) - floor(Int, h_len/2)) : rand(1:1:max(1, h-h_len))
    roi = [x1, y1, x1+w_len-1, y1+h_len-1] # x1, y1, x2, y2

    return x[:, roi[2]:roi[4], roi[1]:roi[3]], "crop", roi
end


"""
Resizes a given image to the given width and height integer values. Returns the resized image, 
key name for keeping changes in a dict, and the ratio of changes for both width and height.
The ratio is calculated with: initial length / resize length
"""
mutable struct Resize
    w; h;

    function Resize(width::Int, height::Int)
        return new(width, height)
    end
end

function (rs::Resize)(x)
    c, h, w = size(x)
    ratio_h = h / rs.h; ratio_w = w / rs.w;
    return imresize(x, (c, rs.h, rs.w)), "resize", [ratio_w, ratio_h]
end


"""
If an image is not in square size, then it fills the minimum side with the enterd fill value
and centers the actual image.

# Keywords:

- fill_value(Float, optional): The value to fill the short side, between [0, 1]. Default: "0"

Returns the squared image, key name "squaritize" for keeping changes in a dict, and the upleft 
and downright coordinates of the actual image in the whole image.
"""
mutable struct Squaritize
    fill_value::Union{Float32, Float64};

    function Squaritize(;fill_value::Union{Float32, Float64}=0.0)
        if fill_value > 1 || fill_value < 0
            println("[ERROR] The fill value in squaritize is not in the range [0, 1]")
            return nothing
        end

        return new(fill_value)
    end
end

function (sq::Squaritize)(x)
    c, h, w = size(x)
    maxlen = max(h, w); minlen = min(h, w)
    if h != w 
        full_x = fill(sq.fill_value, (3, maxlen, maxlen))
        diff = maxlen - minlen
        pads = [floor(Int, diff/2), floor(Int, diff/2)]
        # complete the length for an odd difference size
        if mod(diff, 2) == 1 pads[1] += 1 end
        if minlen == w
            full_x[:,:,pads[1]+1:maxlen-pads[2]] = x
            return full_x, "squaritize", [pads[1]+1, 0, maxlen-pads[2], h]
        else
            full_x[:,pads[1]+1:maxlen-pads[2],:] = x
            return full_x, "squaritize", [0, pads[1]+1, w, maxlen-pads[2]]
        end
    else
        return x, "squaritize", [0, 0, w, h]
    end
end


"""
Flips the image horizontally and/or vertically. 

# Keywords

- horizontal(Bool, optional): set true to flip horizontal. Default: "true"
- vertical(Bool, optional): set true to flip vertical. Default: "false"
- probs(Float or Tuple of Floats, optional): probability to apply a specific flip. The order is 
(horizontal_prob, vertical_prob). Default: "0.5"

Returns the flipped image, key name "flip" for keeping changes in a dict, and the boolean flipping information.
"""
mutable struct Flip
    horizontal::Bool; vertical::Bool; probs::Tuple;

    function Flip(; horizontal::Bool=true, vertical::Bool=false, probs::Union{Float32, Float64, Tuple}=0.5)
        if typeof(probs) <: Union{Float32, Float64}
            probs = (probs, probs)
        end
        return new(horizontal, vertical, probs)
    end
end

function (fl::Flip)(x)
    fh = false
    if fl.horizontal && rand() < fl.probs[1]
        fh = true
        x = reverse(x, dims=3)
    end
    fv = false
    if fl.vertical && rand() < fl.probs[2]
        fv = true
        x = reverse(x, dims=2)
    end
    return x, "flip", [fh, fv]
end


"""
Distorts the color of the image. 

# Keywords:
- probs(Float or Tuple of Floats, optional): probability to apply a specific distortion. The order is 
(brightness_prob, contrast_prob, saturation_prob, hue_prob). Default: "0.5"
- brightness_range(Tuple of 2 Floats, optional): the range for brightness distortion. Default: "(-0.125,0.125)"
- contrast_range(Tuple of 2 Floats, optional): the range for contrast distortion. Default: "(0.5,1.5)"
- saturation_range(Tuple of 2 Floats, optional): the range for saturation distortion. Default: "(0.5,1.5)"
- hue_range(Tuple of 2 Floats, optional): the range for hue distortion. Default: "(-(18/256),(18/256))"

Returns the distorted image, key name "distortion" for keeping changes in a dict, and the distortion values of all.
"""
mutable struct DistortColor
    probs::Union{Float32, Float64, Tuple}; brightness_range::Tuple; contrast_range::Tuple;
    saturation_range::Tuple; hue_range::Tuple;

    function DistortColor(;
        probs::Union{Float32,  Float64, Tuple}=0.5,
        brightness_range::Tuple=(-0.125,0.125),
        contrast_range::Tuple=(0.5,1.5),
        saturation_range::Tuple=(0.5,1.5),
        hue_range::Tuple=(-(18/256),(18/256))
        )

        if typeof(probs) <: Union{Float32, Float64}
            probs = (probs, probs, probs, probs)
        end
        return new(probs, brightness_range, contrast_range, saturation_range, hue_range)
    end
end

function (dc::DistortColor)(img)
    br = rand(dc.brightness_range[1]:0.001:dc.brightness_range[2])
    con = rand(dc.contrast_range[1]:0.1:dc.contrast_range[2])
    sat = rand(dc.saturation_range[1]:0.1:dc.saturation_range[2])
    hue = rand(dc.hue_range[1]:0.001:dc.hue_range[2])
    # brightness distortion
    if rand() < dc.probs[1]; img = _convert(img, beta=br); end; 
    # contrast distortion
    if rand() < dc.probs[2]; img = _convert(img, alpha=con); end;
    
    img = channelview(colorview(HSV, img))
    
    # saturation distortion
    if rand() < dc.probs[3]; img[2,:,:] = _convert(img[2,:,:], alpha=sat); end;
    # hue distortion
    if rand() < dc.probs[4]; img[1,:,:] = _convert(img[1,:,:], beta=hue); end;
    
    img = channelview(colorview(RGB, colorview(HSV, img)))
    
    return img, "distortion", [br, con, sat, hue]
end

function _convert(image; alpha=1, beta=0)
    image = image .* alpha .+ beta
    image[image .< 0] .= 0
    image[image .> 1] .= 1
    return image
end
