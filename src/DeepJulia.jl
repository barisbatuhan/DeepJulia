module DeepJulia

using Knet
using MAT
using Images, FileIO
import Knet: atype

dir() = abspath(@__DIR__)
dir(args...) = abspath(joinpath(dir(), args...))

include("core/layers.jl")
include("core/blocks.jl")
include("core/initializers.jl")
include("models/mobilenet.jl")
include("models/resnet.jl")
include("operations.jl")
include("preprocess/transforms.jl")

end # module