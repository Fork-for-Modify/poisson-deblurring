This branch of the repository provides a pretrained model of P4IP trained on a new low-light dataset employing a physical-process-based noise model. Please refer to the following paper ([arXiv](https://arxiv.org/abs/2207.08201)) for details.
> Zhang, Zhihong, Yuxiao Cheng, Jinli Suo, Liheng Bian, and Qionghai Dai. “INFWIDE: Image and Feature Space Wiener Deconvolution Network for Non-Blind Image Deblurring in Low-Light Conditions.” arXiv, July 17, 2022. https://doi.org/10.48550/arXiv.2207.08201.


In this implementation, we format the original RAW image into two 'fake' RGB images (i.e. rg1b & rg2b) to fit our code framework's data interface. After deblurring these two images separately, we merge them together to get a RAW image and conduct demosaicing afterwards to get the final deblurred RGB image.

## How to run
1. run `test.py` to deblur the two 'fake' RGB images extracted from the RAW image.
2. run `rggb2rgb.py` to merge the deblurring results and get the final deblurred RGB image