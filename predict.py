import argparse
import subprocess
import torch
import numpy as np
from cog import BasePredictor, Path, Input
from scripts.inf_zoom import write_frames, zoom_in, inf_inpaint, text2img

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["pip3", "install", "-e", "."])

    def predict(
            self,
            prompt: str = Input(description="Prompt"),

    ) -> Path:
        opt = argparse.Namespace()
        opt.H = opt.W = 512
        opt.f = 8
        opt.C = 4
        opt.scale = 7.5
        opt.ddim_steps = 50
        opt.ddim_eta = 0
        opt.prompt = prompt

        output_path = "infinite_zoom.gif"

        img = text2img(opt)
        img, mask = inf_inpaint(img, opt)
        img, mask = inf_inpaint(img, opt)
        frames = zoom_in(np.array(img)[..., ::-1], n_frames=60)

        write_frames(frames, fname=output_path, fps=30)

        return Path(output_path)



