import argparse
import subprocess
import torch
import numpy as np
from scripts.inf_zoom import write_frames, zoom_in, text2img, self_inpaint, OmegaConf, load_model_from_config, PLMSSampler, instantiate_from_config, DDIMSampler
from cog import BasePredictor, Path, Input, BaseModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Output(BaseModel):
    mp4: Path = None
    gif: Path = None


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["pip3", "install", "-e", "."])

        txt2img_conf = 'configs/stable-diffusion/v1-inference.yaml'
        txt2img_ckpt = 'models--runwayml--stable-diffusion-v1-5/snapshots/51b78538d58bd5526f1cf8e7c03c36e2799e0178/v1-5-pruned-emaonly.ckpt'
        config = OmegaConf.load(txt2img_conf)
        txt2img_model = load_model_from_config(config, txt2img_ckpt).to(device)
        self.txt2img_sampler = PLMSSampler(txt2img_model)

        inpaint_conf = 'configs/stable-diffusion/v1-inpainting-inference.yaml'
        inpaint_ckpt = 'models--runwayml--stable-diffusion-inpainting/snapshots/e5fae204b8239db36f2cdc2c8b773d10c457a567/sd-v1-5-inpainting.ckpt'
        config = OmegaConf.load(inpaint_conf)
        inpaint_model = instantiate_from_config(config.model)
        inpaint_model.load_state_dict(torch.load(inpaint_ckpt)["state_dict"], strict=False)
        inpaint_model = inpaint_model.to(device)
        self.inpaint_sampler = DDIMSampler(inpaint_model)

    def predict(
            self,
            prompt: str = Input(description="Prompt"),
            output_format: str = Input(description="infinite loop gif or mp4 video", default="mp4", choices=["mp4", "gif"]),
            inpaint_iter: int = Input(description="Number of iterations of pasting the image in it's center and inpainting the boarders", default=2),

    ) -> Output:
        opt = argparse.Namespace()
        opt.dim = 512
        opt.zoom_factor = 4
        opt.inpaint_iter = int(inpaint_iter)
        opt.scale = 7.5
        opt.ddim_steps = 50
        opt.ddim_eta = 0
        opt.prompt = prompt

        img = text2img(self.txt2img_sampler, opt)
        img = self_inpaint(self.inpaint_sampler, img, opt)
        frames = zoom_in(np.array(img)[..., ::-1], factor=opt.zoom_factor, n_self_paste=3, n_frames=60)

        output_format = str(output_format)
        ext = output_format.split(".")[-1]
        fname = f"infinit_zoom.{ext}"
        write_frames(frames, fname=fname, fps=20)

        if ext == 'gif':
            return Output(gif=Path(fname))
        else:
            return Output(mp4=Path(fname))


def write_web_mp4(frames, fps, fname):
    os.makedirs("tmp", exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"tmp/{str(i).zfill(5)}.png", frame)

    # make video

    # make video
    cmd = [
        "ffmpeg",
        "-y",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-start_number",
        str(0),
        "-i",
        f"tmp/%05d.png",
        "-c:v",
        "libx264",
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "17",
        "-preset",
        "veryfast",
        fname,
    ]
    import subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)