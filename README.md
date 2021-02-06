# DeepJulia

## Contact

- **Email:** barisbatuhantopal@gmail.com / baristopal20@ku.edu.tr 

**Note:** For bug reports and requests, please use the GitHub issues.

## Description

DeepJulia is a Deep Learning library implemented in Julia Programming Language using Knet Deep Learning Framework. Knet includes low level operations but not high-level layers and classes for constructing more complex deep learning structures. This library provides some initial layer structures and most commonly used backbone models such as ResNet, VGG, MobileNet out-of-the-box with their pretrained weights.

## Included

- **Basic Layers:** Linear, Convolution, Batch Normalization, ReLU, Pooling, Global Pooling, Dropout, etc.

- **GPU/CPU Devices:** With only one function call, your models can be moved to a GPU or CPU device.

- **Kaiming Initializer:** Initialization with Kaiming is added especially for Convolution operations.

- **ResNet & MobilenetV2:** All basic ResNet structures & MobileNetV2 with their pretrained weights are included in the library.

- **Preprocessing Methods:** For image data, preprocessing methods are added, which include random & center crops, horizontal flip, square-image conversion and color distortion.

## Links

- The pretrained weights can be found in [this link](https://drive.google.com/drive/folders/1YRi2S-IA_Ekz7R9Ey2hf1OqyHo4PAsxv?usp=sharing). Please download the required ones and place them to `./weights` directory.

## Sample Code

```julia

model = MobileNetV2(pretrained=true)
model = to_gpu(model)
model = set_eval_mode(model)

tr = Transforms(
    [
        RandomCrop(min_ratio=0.6),
        Resize(224, 224),
        Flip(horizontal=true),
        DistortColor(probs=0.5)
    ],
    ["cat.jpeg"],
    img_size=224,
    return_changes=true,
    batch_size=1
)

imgs, _, vals = get_batch(tr)
imgs = convert(KnetArray{Float32}, imgs)
preds = model(imgs)
print(findmax(preds), dims=1)
```
**Output:** You will see that the index output will be 286, which is the correct label to the cat image.

## To-Do:

- **Backbone Addition:** VGG and other most-used structures will be added to the library with pretrained weights.

- **Test-cases Addition:** Tests will be included for checking everything is functioning well even after upgrading a version.

## Final Notes

- Since this library is built on top of the Knet framework, please check [Knet Documentation](https://denizyuret.github.io/Knet.jl/latest/reference/) for non-existing features. Please keep in mind that this library only supports some basic operations & models additional to Knet and currently, it is only developed by a single person.