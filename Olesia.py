import torch


class Olesia:
    def __init__(self, model_path, accepted_audio_length=24575):
        self.model = torch.jit.load(model_path)
        self.accepted_audio_length=accepted_audio_length

    def __call__(self, x):
        if x.shape[1] > self.accepted_audio_length:
            # too long, chop off
            x = x[:, :self.accepted_audio_length]
        elif x.shape[1] < self.accepted_audio_length:
            # too short, pad
            zeros = torch.zeros(x.shape[0], (self.accepted_audio_length - x.shape[1]))
            x = torch.cat([x, zeros], 1)
        z = self.model(x)
        return z
