import os
import soundfile as sf
import torch.jit
from torch.utils.data import DataLoader

from Olesia import Olesia
from data import prepare_fn_groups, ContrastLoader
from utils import get_audio_length, nr_samples


def filter1(fn):
    try:
        if get_audio_length(fn) < (44100 / 10):
            # print(f"too short: {fn}")
            return False
    except RuntimeError:
        # print(f"exception: {fn}")
        return False
    # print(f"accept {fn}")
    return (fn.endswith("wav") or fn.endswith(
        "WAV")) and "MASTER" not in os.path.basename(
        fn) and "mix" not in os.path.basename(fn)


def filter2(fn):
    return filter1(fn)


def my_torch_istft(spec, n_fft=2048):
    zeros = torch.zeros(spec.shape[0], 1, 48, 2).to(spec.device)
    # Transpose the spectrogram to fit the dimensions required by torch.istft
    spec = spec.permute(0, 2, 3, 1)
    # Append the fundamental to the spectrogram
    spec = torch.cat([zeros, spec], 1).contiguous()

    spec = torch.view_as_complex(spec)
    sig = torch.istft(spec, n_fft=n_fft)
    return sig


if __name__ == '__main__':
    batch_size = 60
    groups = prepare_fn_groups(
        "/media/data/stefan/CURATED TRAP MASTER DATASET",
        filter_fun_level1=filter1, filter_fun_level2=filter2)

    groups_mix = [entry[0] for entry in groups]
    groups_single = [entry[1] for entry in groups]
    eval_split = int(len(groups_mix) * 0.1)
    print(f"Size train: {len(groups)-eval_split}, eval: {eval_split}")
    train_loader = DataLoader(ContrastLoader(groups_mix[eval_split:],
                                             groups_single[eval_split:],
                                             nr_samples=nr_samples),
                                             shuffle=True, batch_size=batch_size,
                                             num_workers=5, drop_last=True)
    eval_loader = DataLoader(ContrastLoader(groups_mix[:eval_split],
                                            groups_single[:eval_split],
                                            nr_samples=nr_samples),
                                            shuffle=True, batch_size=batch_size,
                                            num_workers=5, drop_last=True)
    olesia = Olesia("./encoderOlesia15.pt")
    drumgan = torch.jit.load("./TrainedDrumGAN.pt", map_location='cpu')
    for entry in enumerate(train_loader):
        z = olesia(entry[1][0])
        sound = drumgan(z)[0]
        sound = my_torch_istft(sound)
        print("hh")

