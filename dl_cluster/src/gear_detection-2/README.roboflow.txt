
gear_detection - v2 cluster_lab_gear_data
==============================

This dataset was exported via roboflow.com on December 9, 2025 at 2:56 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 76 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 21 percent of the image
* Random shear of between -5° to +5° horizontally and -5° to +5° vertically
* Random brigthness adjustment of between -24 and +24 percent
* Random exposure adjustment of between -14 and +14 percent
* Random Gaussian blur of between 0 and 1.1 pixels
* Salt and pepper noise was applied to 1.68 percent of pixels


