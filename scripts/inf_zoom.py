import argparse
from math import log

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange, repeat
from torch import autocast
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from scripts.txt2img import check_safety, put_watermark, WatermarkEncoder
from imwatermark import WatermarkEncoder

wm = "StableDiffusionV1"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

device = torch.device("cuda")


def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def text2img(opt):
    txt2img_ckpt = "weights_and_confs/txt2img.ckpt"
    txt2img_conf = "weights_and_confs/v1-inference.yaml"
    config = OmegaConf.load(txt2img_conf)
    model = load_model_from_config(config, txt2img_ckpt).to(device)

    sampler = PLMSSampler(model)

    start_code = None
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""])
                c = model.get_learned_conditioning([opt.prompt])
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=start_code)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                # TODO check saftey
                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                # x_checked_image = x_samples_ddim

                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                x_sample = x_checked_image_torch[0]
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                # TODO set watermark
                img = put_watermark(img, wm_encoder)
                return img


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            result, has_nsfw_concept = check_safety(result)
            result = result*255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    # result = [put_watermark(img) for img in result]
    return result


def inf_inpaint(image, opt):
    inpaint_ckpt = "weights_and_confs/inpaint.ckpt"
    inpaint_conf = "weights_and_confs/v1-inpainting-inference.yaml"
    config = OmegaConf.load(inpaint_conf)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(inpaint_ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)


    downscaled_image = image.resize((128, 128))
    image = np.array(image)
    downscaled_image = np.array(downscaled_image)
    image[256-64:256+64, 256-64:256+64] = downscaled_image
    image = Image.fromarray(image)

    mask = np.zeros_like(np.array(image), dtype=np.uint8)
    mask[256-128:256+128, 256-128:256+128] = 255
    mask[256-64:256+64, 256-64:256+64] = 0
    mask = Image.fromarray(mask)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=opt.prompt,
        seed=0,
        scale=opt.scale,
        ddim_steps=opt.ddim_steps,
        num_samples=1,
        h=opt.H, w=opt.W
    )
    return result[0], mask


def resize_img(img, size):
    return np.array(Image.fromarray(img).resize((size, size)))


def zoom_in(img, n_frames=120):
    frames = []
    size = img.shape[0]
    factor = 4
    middle = size//2

    for small_size in [size // factor**(i+1) for i in range(3)]:
        hs = small_size // 2
        img[middle - hs:middle + hs, middle - hs:middle + hs] = resize_img(img, small_size)

    log_base = factor ** (-1/n_frames)
    scale_factors = [1] + [factor * log_base**i for i in range(int(log(1 / factor, log_base)), -1, -1)]
    for scale_factor in scale_factors:
        print(scale_factor)
        new_size = int(size * scale_factor)
        new_image = resize_img(img, new_size)
        new_image = new_image[new_size//2-size//2:new_size//2+size//2, new_size//2-size//2:new_size//2+size//2]
        d = int(size / factor * scale_factor)
        x = new_image[middle-d//2:middle+d//2, middle-d//2: middle+d//2].shape[0]
        new_image[middle-d//2:middle+d//2, middle-d//2: middle+d//2] = resize_img(img, x)
        frame = resize_img(new_image, size)
        frames.append(frame)
    return frames


def write_frames(frames, fname="infinite_zoom.gif", fps=30):
    # height, width = frames[0].shape[:2]
    # video = cv2.VideoWriter(fname, 0, fps, (width, height))
    #
    # for image in frames:
    #     video.write(image)
    #
    # video.release()

    import imageio
    imageio.mimsave(fname, [Image.fromarray(frame) for frame in frames], duration=1/fps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--scale", default=7.5, type=float)
    parser.add_argument("--H", default=512, type=int)
    parser.add_argument("--W", default=512, type=int)
    parser.add_argument("--f", default=8, type=int)
    parser.add_argument("--C", default=4, type=int)
    parser.add_argument("--ddim_steps", default=50, type=int)
    parser.add_argument("--ddim_eta", default=0, type=int)
    opt = parser.parse_args()

    img = text2img(opt)
    img.save("output-1.png")
    img, mask = inf_inpaint(img, opt)
    mask.save("mask.png")
    img.save("output-2.png")
    img, mask = inf_inpaint(img, opt)
    img.save("output-3.png")

    frames = zoom_in(np.array(img)[...,::-1], n_frames=60)

    write_frames(frames, fname="infinite_zoom.gif", fps=30)

if __name__ == "__main__":
    main()
