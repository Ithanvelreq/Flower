import soundfile as sf
import torch

perc_kws = ['Kick', 'KICK', 'KD', 'kick', 'BD', 'Snare', 'SNARE', 'snare',
            'SD', 'HH', 'HiHat', 'HIHAT', 'hihat', 'hat', 'HAT', 'Hat',
            'Crash', 'CRASH', 'crash', 'Toms', 'TOM', 'Tom', 'tom', 'TT',
            'Ride', 'ride', 'RIDE', 'RD', 'cym', 'Cym', 'Clap', 'CLAP', 'CY',
            'CYM', 'Perc', 'PERC', 'Drum', 'DRUM', 'OH']

perc_types = {"Kick": ['Kick', 'KICK', 'KD', 'kick', 'BD'],
              "Snare": ['Snare', 'SNARE', 'snare', 'SD', 'Clap', 'CLAP'],
              "HiHat": ['HH', 'HiHat', 'HIHAT', 'hihat', 'hat', 'HAT', 'Hat'],
              "Crash": ['Crash', 'CRASH', 'crash'],
              "Toms": ['Toms', 'TOM', 'Tom', 'tom', 'TT'],
              "Ride": ['Ride', 'ride', 'RIDE', 'RD', 'cym', 'Cym', 'CY', 'CYM']
             }

# length of input to contrastive models, sample rate times seconds we want
nr_samples = 44100 * 1


# This can be bogus, ask what it actually returns
def get_audio_length(fn):
    with sf.SoundFile(fn, 'r') as f:
        return len(f)


def my_torch_istft(spec, n_fft=2048):
    zeros = torch.zeros(spec.shape[0], 1, 48, 2).to(spec.device)
    # Transpose the spectrogram to fit the dimensions required by torch.istft
    spec = spec.permute(0, 2, 3, 1)
    # Append the fundamental to the spectrogram
    spec = torch.cat([zeros, spec], 1).contiguous()

    spec = torch.view_as_complex(spec)
    sig = torch.istft(spec, n_fft=n_fft)
    return sig

