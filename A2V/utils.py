import torch

def create_overlapping_samples(sound):
    sound_avg = sound.mean(1)
    audio_len = sound_avg.size()[0]
    sample_window = 8000
    overlapping = 2000
    left = 0
    right = 8000
    audio_data = torch.Tensor()
    while left < audio_len:
        sample = sound_avg[left:right]
        if sample.size()[0] != 8000:
            break
        sample = sample.view(1, 1, sample.size()[0])
        audio_data = torch.cat((audio_data, sample))
        # print(audio_data.size())
    #     print(sample.size())
        left = right - overlapping
        right = left + sample_window

    return audio_data
