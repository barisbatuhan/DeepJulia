"""
Kaiming initialization.

# Keywords:

- dims (Array): an array having the initialization parameter's shape
- mode (Int, optional): if set to 1, then initialization division is set to 
prod(dims) / dims[end], else prod(dims) / dims[end - 1]. Default: "1"
- gain (Int, optional): the ratio to multipy the initialization. Default: "1"
"""

function kaiming(dims...; mode=1, gain=1)
    w = rand(dims...)
    if ndims(w) == 1
        fanout = 1
        fanin = length(w)
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanin = div(length(w),  dims[end])
        fanout = div(length(w), dims[end-1])
    end
    
    fan = fanin
    if mode != 1 fan = fanout end
    
    s = convert(eltype(w), gain*sqrt(3 / (fan)))
    return 2s .* w .- s
end