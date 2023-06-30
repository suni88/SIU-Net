# SIU-Net
Skip-Inception U-Net for Ultrasound Lateral Bony Feature Segmentation

## SIU-Net Paper
SIU-Net was published in Biocybernetics and Biomedical Engineering Journal under the following citation:
>S. Banerjee et al. “Ultrasound spine image segmentation using multi-scale feature fusion Skip-Inception U-Net (SIU-Net)”. In: Biocybernetics and Biomedical Engineering (2022-01), pp. 341–361. issn: 02085216. doi: 10.1016/j.bbe.2022.02.011 (cit. on pp. 10, 12, 13, 15).

## Dataset
This model was originally designed to perform the segmentation of ultrasound spine images but can be used with other datasets as well. It is recommended to create a folder named "data", which has the folders "train" and "test" inside. The folder "train" would have the images to train the model, while the folder "test" would have the images to test the model. The predicted images from the model will be saved in the "test" folder with the suffix "_predict".

## SIU-Net Model
Skip-Inception U-Net, or SIU-Net, is developed to suitably segment the Thoracic Bony Features and Lumbar Bony Features in the ultrasound spine image dataset. The standard U-Net is adopted as the main network architecture, and the simple convolutional layers are replaced with Inception blocks. The encoders-decoders are bridged using newly designed decoder side skip pathways. The architecture of the proposed network of SIU-Net is shown below.

![alt text](/images/SIU-Net.png)
