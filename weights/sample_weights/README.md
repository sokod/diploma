Pre-trained networks are available directly when creating the model object.

Currently 4 models are available:
  - RDN: psnr-large, psnr-small, noise-cancel

Example usage:

  ``` model = RDN(weights='noise-cancel')```
  
The network parameters will be automatically chosen.
