In this implementation, we format the original RAW image into two 'fake' RGB image (i.e. rg1b & rg2b) to fit our pretrained model. After deblur these two images separately, we merge them back to a RAW image and conduct demasaic.

## How to run
1. run `test.py` to deblur the two 'fake' RGB image extracted from the RAW image.
2. run `rggb2rgb.py` to merge the deblurring results and get the final deblurred RGB image