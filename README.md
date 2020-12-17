# Semantic Segmentation With Sentinel-2 Satellite Imagery
This repository is a complete walkthrough to download the "landcovernet" dataset, clean and preprocess it, and train a Deep Learning U-Net model with it.

# Environment
```
conda create -n segmentation python=3.8
activate segmentation
pip install -r requirements.txt
```

# Data
[LandCoverNet](https://registry.mlhub.earth/10.34911/rdnt.d2ce8i/) dataset was released in July 2020. It is a global annual land cover classification training dataset with labels for the multi-spectral satellite imagery from Sentinel-2 mission in 2018. The documentation can be found [here](https://radiant-mlhub.s3-us-west-2.amazonaws.com/landcovernet/Documentation.pdf). There are total of 1980 image chips of 256x256 pixel in the current V1.0 version spanning 66 tiles of Sentinel-2. Each image chip contains temporal observations from Sentinel-2 surface reflectance product (L2A) at 10m spatial resolution and an annual class label, all stored in raster format (GeoTIFF files).

![image](/docs/landcovernet.jpg)

The following schema is used for classification:

![image](/docs/annotations.jpg)

# Downloading data
To download the data, you first need to register on the [Radiant MLHub website](http://www.mlhub.earth/) to get the API token. MLHub offers its API ([documentation](https://docs.mlhub.earth/)) to download the full dataset but one can also use a more raw Python script with the `urllib` library to download the data, using instructions [provided here](https://github.com/radiantearth/mlhub-tutorials/blob/master/notebooks/radiant-mlhub-landcovernet.ipynb). By the time you read this tutorial, I hope the MLHub team has completed the API development and it works flawlessly. Unfortunately, in Dec 2020 it is still at a very early stage of development and downloading the data fails after getting around 400 items.

To overcome this issue, I had to play around with the links generated with the approach involving `urllib` library and found a way to "crack" it to download only the images that I needed. The full dataset is 250GB and downloading it completely would take ages to complete. Using my custom approach, I managed to reduce the size of the data to only 323 MB.

To do this, I firstly downloaded only the image bands that contain cloud coverage assessment values that say how probable it is that a particular pixel contains clouds in it (clouds are a frequent problem while working with Sentinel-2 images). For each of 1980 different locations where images were taken, I took 4 cloudless images for each quarter of the year (Jan-Mar; Apr-Jun; Jul-Sep; Oct-Dec), as well as the label masks. This is how an image and its corresponding mask looks like:

![image](/docs/label_preview.jpg)

# Cleaning the data
The data resulted to be very dirty. Apart from clouds which couldn't be removed completely using cloud coverage assessment (it is only estimation, so some cloudy images can still be present), many labels are of insufficient quality. But that's not a problem, let's get our hands dirty!

These are the steps that I went through to clean the data:
1. Manually delete the labels that were clearly erroneous and leave only high-quality masks, discarding over 50% of the data already! 
2. Delete the satellite images that now had no labels. 
3. Go through the remaining Sentinel-2 images and manually delete those that still contain clouds.

In the end, we are left with 2512 cloudless RGB images (inputs) as well as 920 label masks (targets), both in .png format. The reason why we have more inputs than targets is that for each target, several input images are used for different time of the year. The data is finally prepared for training.

# Training
I used fastai API on top of PyTorch and went for U-Net convolutional neural net with a ResNet34 encoder to perform semantic segmentation. To make the training most efficient, I used a model that was pretrained on Imagenet. This implies that our data has to be maximally similar to Imagenet data. Following preprocessing steps were used to do this:
- `RandomResizedCrop` to crop images at random center points from 256x256 px to 224x224 px
- Normalize the images to Imagenet stats
- Use augmentation techniques: horizontal and vertical flips

Also, I had to implement a custom accuracy metric since current `foreground_acc()` of fastai seems to have a bug (probably, due to new recent release of PyTorch). Now, let's train our model! I followed a standard fastai approach of:
1. Training several epochs on only the last layer, keeping the rest of the model frozen
2. Finding new optimal learning rate
3. Unfreezing the model and training for more epochs with a *discriminative learning rate* starting small in the first layers and gradually increasing up to the last ones.

![image](/docs/training.jpg)

After 15 epochs in total, an accuracy of 74% could be achieved.

Here is a preview of how a prediction on one of the validation samples looked like:

![image](/docs/preds.jpg)

# Further improvements
As you can see, it's not perfect yet. However, taking into account the complexity of the data and the number of classes, the results are pretty decent and our prediction looks pretty close to actual landcover mask! Ideas on how to improve the model performance:
- Include more bands into training. For now, I only used red, green and blue bands from the satellite images. Every image actually contains **19** bands. My guess is that including B8 (near-infrared) band or constructing a custom NDVI band that evaluates the vegetation density would increase the model's ability to more accurately recognize vegetation and label classes as "cropland", "(semi) natural vegetation" and "woody vegetation" more correctly. 
- The drawback to this approach is that the pretrained Imagenet model couldn't be used anymore since we would have 4-channel images instead of 3-channel ones. To compensate for this, another model could be trained on satellite image dataset (for example, as [Eurosat](https://arxiv.org/abs/1709.00029)) solely to use as the pretraining model for this task.
