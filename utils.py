import soundfile as sf

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
