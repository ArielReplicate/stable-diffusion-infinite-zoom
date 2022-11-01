## Stable Diffusion Infinite Zoom

Run it on Replicate: 
<a href="https://replicate.com/arielreplicate/stable_diffusion_infinite_zoom"><img src="https://replicate.com/arielreplicate/stable_diffusion_infinite_zoom/badge"></a>


<img src="assets/inf_zooms/infinite_zoom_1.gif" width="200" height="200" /> <img src="assets/inf_zooms/infinite_zoom_2.gif" width="200" height="200" />
<img src="assets/inf_zooms/infinite_zoom_4.gif" width="200" height="200" /> <img src="assets/inf_zooms/infinite_zoom_3.gif" width="200" height="200" />


This repo is based on [Stable Diffusion](https://github.com/compvis/stable-diffusion) by [CompVis group:](https://ommer-lab.com/) 
and [Stable Inpainting](https://github.com/runwayml/stable-diffusion) by [Runway](https://runwayml.com/)

The idea is based on this [tweet](https://twitter.com/matthen2/status/1564608773485895692) by [Matt Henderson](https://twitter.com/matthen2)

## Model description
Given a prompt I run txt2img,py with [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
Then I paste a downscaled version of the image into it's center and inpaint around the center using inpaint.py using this [sd-v1-5-inpainting.ckpt from](https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main) 
I repeat the inpainting step twice.

Then zoom in by upscaling the image and cuting it to the original size  while pasting the "center" image in its due area.


# How to run
## Download text-2-image and inpainting weights
hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.ckpt", cache_dir=".", use_auth_token=<HuggingFace token>)
hf_hub_download(repo_id="runwayml/stable-diffusion-inpainting", filename="sd-v1-5-inpainting.ckpt", cache_dir=".", use_auth_token=<HuggingFace token>)

## create video
`python3 scripts/inf_zoom.py <your prompt>`

# Credits
```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
