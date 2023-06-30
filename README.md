# SIU-Net
Skip-Inception U-Net for Ultrasound Lateral Bony Feature Segmentation

## SIU-Net Paper
SIU-Net was published in Biocybernetics and Biomedical Engineering Journal under the following citation:
>S. Banerjee et al. “Ultrasound spine image segmentation using multi-scale feature fusion Skip-Inception U-Net (SIU-Net)”. In: Biocybernetics and Biomedical Engineering (2022-01), pp. 341–361. ISSN: 02085216. doi: 10.1016/j.bbe.2022.02.011 (cit. on pp. 10, 12, 13, 15).

## Dataset
This model was originally designed to perform the segmentation of ultrasound spine images but can also be used with other datasets. It is recommended to create a folder named "data", which has the folders "train" and "test" inside. The folder "train" would have the images to train the model, while the folder "test" would have the images to test the model. The predicted images from the model will be saved in the "test" folder with the suffix "_predict".

## SIU-Net Model
Skip-Inception U-Net, or SIU-Net, is developed to suitably segment the Thoracic Bony Features and Lumbar Bony Features in the ultrasound spine image dataset. The standard U-Net is adopted as the main network architecture, and the simple convolutional layers are replaced with Inception blocks. The encoders-decoders are bridged using newly designed decoder side skip pathways. The architecture of the proposed network of SIU-Net is shown below.

![alt text](/images/SIU-Net.png)

In SIU-Net, the output of the previous IB of the decoder of the same dense block is merged with the corresponding up-sampled output of the lower dense block through dense skip connections (DSC) (Fig. 3). Through a dense convolution operation, each node in a decoder is presented with a final aggregated-fused feature map containing: a) the feature from the previous decoder, b) intermediate block combined feature maps and c) the same-scale feature from the corresponding encoder, as it is possible to see in the SIU-Net architecture.

To solve the issue of choosing an appropriate kernel size to handle large variability in the spine image dataset, the concept of Inception block (IB) is adopted to develop a high-performance segmentation model. A modified IB of Inception V2 architecture is introduced to replace the traditional convolutional layers of basic U-Net. IB has two advantages: (a) it increases the depth and width of the model without any increase in the computational requirement, and (b) it allows the flexibility of using multiple filter sizes within the same level. The architecture of the designed Inception Block is shown below.

![alt text](/images/InceptionBlock.png)

For more information on this model, read the [paper.](https://www.sciencedirect.com/science/article/abs/pii/S0208521622000146)https://www.sciencedirect.com/science/article/abs/pii/S0208521622000146)






