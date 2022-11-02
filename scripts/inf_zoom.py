import argparse
import os
from math import log
import subprocess
import imageio

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


def text2img(txt2img_sampler, opt):
    with torch.no_grad():
        # stable diffusion constants
        C = 4
        F = 8

        start_code = None
        with torch.no_grad():
            with autocast("cuda"):
                with txt2img_sampler.model.ema_scope():
                    uc = None
                    if opt.scale != 1.0:
                        uc = txt2img_sampler.model.get_learned_conditioning([""])
                    c = txt2img_sampler.model.get_learned_conditioning([opt.prompt])
                    shape = [C, opt.dim // F, opt.dim // F]
                    samples_ddim, _ = txt2img_sampler.sample(S=opt.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=1,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x_T=start_code)

                    x_samples_ddim = txt2img_sampler.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    x_sample = x_checked_image_torch[0]
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    return img


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
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
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
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

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1)
        result, has_nsfw_concept = check_safety(result)
        result = result * 255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    # result = [put_watermark(img) for img in result]
    return result


def self_inpaint(inpaint_sampler, image, opt):
    with torch.no_grad():

        HS = opt.dim // 2
        SS = opt.dim // opt.zoom_factor
        HSS = SS // 2

        # Prepare mask surrounding middle
        mask = np.zeros((opt.dim, opt.dim), dtype=np.uint8)
        mask[HS - SS:HS + SS, HS - SS:HS + SS] = 255
        mask[HS - HSS:HS + HSS, HS - HSS:HS + HSS] = 0
        mask = Image.fromarray(mask)

        image.save(f"debug-0.png")
        for i in range(opt.inpaint_iter):
            # Paste image inside itself
            downscaled_image = np.array(image.resize((SS, SS)))
            image = np.array(image)
            image[HS - HSS:HS + HSS, HS - HSS:HS + HSS] = downscaled_image
            image = Image.fromarray(image)

            image = inpaint(
                sampler=inpaint_sampler,
                image=image,
                mask=mask,
                prompt=opt.prompt,
                seed=0,
                scale=opt.scale,
                ddim_steps=opt.ddim_steps,
                num_samples=1,
                h=opt.dim, w=opt.dim
            )[0]
            image.save(f"debug-{i + 1}.png")

        return image


def resize_img(img, size):
    return np.array(Image.fromarray(img).resize((size, size)))


def zoom_in(img, factor, n_self_paste=0, n_frames=120):
    frames = []
    size = img.shape[0]
    middle = size // 2
    center_size = size / factor

    # Paste the image inside itself in decreasing sizes
    for small_size in [size // factor ** (i + 1) for i in range(n_self_paste)]:
        hs = small_size // 2
        img[middle - hs:middle + hs, middle - hs:middle + hs] = resize_img(img, small_size)

    # Compute a serie of scale up factor that dictate a steady zoom-in speed
    log_base = factor ** (-1 / n_frames)
    scale_factors = [1] + [factor * log_base ** i for i in range(int(log(1 / factor, log_base)), -1, -1)]

    for scale_factor in scale_factors:
        # zoom in
        new_size = int(size * scale_factor)
        new_image = resize_img(img, new_size)
        slc = slice(new_size // 2 - size // 2, new_size // 2 + size // 2)
        new_image = new_image[slc, slc]

        # Compute center position and paste it in higher rsolution
        center_new_half_size = int(center_size * scale_factor / 2)
        slc = slice(middle - center_new_half_size, middle + center_new_half_size)
        x = new_image[slc, slc].shape[0]
        new_image[slc, slc] = resize_img(img, x)

        frame = resize_img(new_image, size)
        frames.append(frame)

    return frames


def write_frames(frames, fname, fps=30):
    if fname.endswith('gif'):
        imageio.mimsave(fname, [Image.fromarray(frame) for frame in frames], duration=1 / fps)
    elif fname.endswith('mp4'):
        os.makedirs("tmp", exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(f"tmp/{str(i).zfill(5)}.png", frame)

        cmd = [
            "ffmpeg", "-y", "-vcodec", "png",
            "-r", str(fps),
            "-start_number", str(0),
            "-i", f"tmp/%05d.png",
            "-c:v", "libx264", "-vf", f"fps={fps}", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "veryfast", fname,
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
    else:
        raise ValueError("Can only produce mp4 or gif files")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")

    parser.add_argument("--dim", default=512, type=int, help="size of image")
    parser.add_argument("--zoom_factor", default=4, type=int, help="Size of the replaced image")
    parser.add_argument("--inpaint_iter", default=2, type=int,
                        help="How many time to fill image with itself and inpaint the boarders")

    parser.add_argument("--scale", default=7.5, type=float)
    parser.add_argument("--ddim_steps", default=50, type=int)
    parser.add_argument("--ddim_eta", default=0, type=int)
    opt = parser.parse_args()

    # load models
    txt2img_conf = 'configs/stable-diffusion/v1-inference.yaml'
    txt2img_ckpt = 'models--runwayml--stable-diffusion-v1-5/snapshots/51b78538d58bd5526f1cf8e7c03c36e2799e0178/v1-5-pruned-emaonly.ckpt'
    config = OmegaConf.load(txt2img_conf)
    txt2img_model = load_model_from_config(config, txt2img_ckpt).to(device)
    txt2img_sampler = PLMSSampler(txt2img_model)

    inpaint_conf = 'configs/stable-diffusion/v1-inpainting-inference.yaml'
    inpaint_ckpt = 'models--runwayml--stable-diffusion-inpainting/snapshots/e5fae204b8239db36f2cdc2c8b773d10c457a567/sd-v1-5-inpainting.ckpt'
    config = OmegaConf.load(inpaint_conf)
    inpaint_model = instantiate_from_config(config.model)
    inpaint_model.load_state_dict(torch.load(inpaint_ckpt)["state_dict"], strict=False)
    inpaint_model = inpaint_model.to(device)
    inpaint_sampler = DDIMSampler(inpaint_model)


    img = text2img(txt2img_sampler, opt)
    img = self_inpaint(inpaint_sampler, img, opt)

    frames = zoom_in(np.array(img), opt.zoom_factor, n_frames=60)

    write_frames(frames, fname="infinite_zoom.mp4", fps=10)


if __name__ == "__main__":
    main()
