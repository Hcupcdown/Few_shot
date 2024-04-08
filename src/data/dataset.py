import json
import os
import random

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F


def random_mix_audio(audio1, audio2):
    audio1 = audio1 / (torch.max(audio1) + 1e-8)
    audio2 = audio2 / (torch.max(audio2) + 1e-8)
    random_ratio = (random.random()+0.1)*3
    mix_audio = audio1 + audio2*random_ratio
    mix_audio = mix_audio / (torch.max(mix_audio) + 1e-8)
    return mix_audio

class SeparDataset:


    def __init__(self,
                 dataset_dir,
                 only_clean = False,
                 *args,
                 **kwargs):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
        """
        
        self.dataset_dir = dataset_dir
        self.audio_data_map = {}
        self.radar_data_map = {}
        self.sample_per_person = 100
        self.only_clean = only_clean
        for persoin_id in os.listdir(self.dataset_dir):
            for sample_name in os.listdir(os.path.join(self.dataset_dir, persoin_id, "audio")):
                temp_audio = os.path.join(self.dataset_dir, persoin_id, "audio", sample_name)
                temp_radar = os.path.join(self.dataset_dir, persoin_id, "radar", sample_name.replace("wav", "npy"))
                self.audio_data_map[int(persoin_id)] = self.audio_data_map.get(int(persoin_id), []) + [temp_audio]
                self.radar_data_map[int(persoin_id)] = self.radar_data_map.get(int(persoin_id), []) + [temp_radar]
        self.person_num = len(self.audio_data_map)


    def __getitem__(self, index):
        sample_id = index % self.sample_per_person
        person_id = index // self.sample_per_person
        gt_audio_path = self.audio_data_map[person_id][sample_id]
        gt_radar_path = self.radar_data_map[person_id][sample_id]
        gt_audio, _ = torchaudio.load(gt_audio_path)
        if self.only_clean:
            return gt_audio

        # 随机选择一个不同人的噪声
        noisy_person_id = random.randint(0, self.person_num-2)
        noisy_person_id = noisy_person_id if noisy_person_id < person_id\
                        else noisy_person_id + 1
        noisy_sample_id = random.randint(0, self.sample_per_person-1)
        noisy_audio_path = self.audio_data_map[noisy_person_id][noisy_sample_id]
        
        noisy_audio, _ = torchaudio.load(noisy_audio_path)
        gt_radar = torch.tensor(np.load(gt_radar_path),
                                dtype = torch.float32)
        mix_audio = random_mix_audio(gt_audio, noisy_audio)
        return gt_radar, gt_audio, mix_audio, torch.tensor(person_id)

    def __len__(self):
        return self.sample_per_person * self.person_num

class IGNLibriDataset:


    def __init__(self,
                 dataset_dir,
                 only_clean = False,
                 *args,
                 **kwargs):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
        """
        
        self.dataset_dir = dataset_dir
        self.sample_list = []
        self.id_dict = {}
        self.only_clean = only_clean
        for sample_name in os.listdir(os.path.join(self.dataset_dir, "s1")):
            self.sample_list.append(sample_name)
            temp_id = sample_name.split("-")[0]
            if temp_id in self.id_dict:
                continue
            else:
                self.id_dict[temp_id] = len(self.id_dict)
        print(len(self.id_dict))

    def __getitem__(self, index):
        gt_sample = self.sample_list[index]
        gt_audio_path = os.path.join(self.dataset_dir, "s1", gt_sample)
        gt_radar_path = os.path.join(self.dataset_dir, "s1_radar", gt_sample.replace("wav", "npy"))
        gt_audio, _ = torchaudio.load(gt_audio_path)
        if self.only_clean:
            return gt_audio

        # 随机选择一个不同人的噪声
        noisy_person_id = random.randint(0, len(self.sample_list)-1)
        noisy_audio_sample = self.sample_list[noisy_person_id]
        noisy_audio_path = os.path.join(self.dataset_dir, "s2", noisy_audio_sample)
        noisy_audio, _ = torchaudio.load(noisy_audio_path)
        gt_radar = torch.tensor(np.load(gt_radar_path),
                                dtype = torch.float32)
        gt_radar = gt_radar.squeeze(0)
        mix_audio = random_mix_audio(gt_audio, noisy_audio)
        person_id = self.id_dict[gt_sample.split("-")[0]]
        return gt_radar, gt_audio, mix_audio, torch.tensor(person_id)

    def __len__(self):
        return len(self.sample_list)


class LibriDataset:


    def __init__(self,
                 dataset_dir,
                 LibriSpeech_dir,
                 only_clean = False,
                 *args,
                 **kwargs):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
        """
        
        self.dataset_dir = dataset_dir
        self.audio_data_map = {}
        self.radar_data_map = {}
        self.sample_per_person = 100
        self.only_clean = only_clean
        for persoin_id in os.listdir(self.dataset_dir):
            for sample_name in os.listdir(os.path.join(self.dataset_dir, persoin_id, "audio")):
                temp_audio = os.path.join(self.dataset_dir, persoin_id, "audio", sample_name)
                temp_radar = os.path.join(self.dataset_dir, persoin_id, "radar", sample_name.replace("wav", "npy"))
                self.audio_data_map[int(persoin_id)] = self.audio_data_map.get(int(persoin_id), []) + [temp_audio]
                self.radar_data_map[int(persoin_id)] = self.radar_data_map.get(int(persoin_id), []) + [temp_radar]
        self.person_num = len(self.audio_data_map)
        self.LibriSpeech_dir = LibriSpeech_dir
        self.noise_list = []
        for noise_file in os.listdir(LibriSpeech_dir):
            self.noise_list.append(noise_file)
        self.noise_len = len(self.noise_list)

    def __getitem__(self, index):
        sample_id = index % self.sample_per_person
        person_id = index // self.sample_per_person
        gt_audio_path = self.audio_data_map[person_id][sample_id]
        gt_radar_path = self.radar_data_map[person_id][sample_id]
        gt_audio, _ = torchaudio.load(gt_audio_path)
        if self.only_clean:
            return gt_audio

        # 随机选择一个不同人的噪声
        noisy_sample = random.randint(0, self.noise_len-1)
        noisy_path = os.path.join(self.LibriSpeech_dir, self.noise_list[noisy_sample])
        noisy_audio, _ = torchaudio.load(noisy_path)

        # align the length of noisy_audio and gt_audio
        noise_l = noisy_audio.shape[-1]
        gt_audio_l = gt_audio.shape[-1]
        if noise_l > gt_audio_l:
            start = random.randint(0, noise_l-gt_audio_l)
            noisy_audio = noisy_audio[..., start : start+gt_audio.shape[1]]
        else:
            noisy_audio = F.pad(noisy_audio, (0, gt_audio_l-noise_l))
        
        mix_audio = random_mix_audio(gt_audio, noisy_audio)

        gt_radar = torch.tensor(np.load(gt_radar_path),
                                dtype = torch.float32)
        return gt_radar, gt_audio, mix_audio, torch.tensor(person_id)

    def __len__(self):
        return self.sample_per_person * self.person_num

class FewShotDataset:

    def __init__(self,
                 trainset_dir,
                 few_shot_dir,
                 num_shot,
                 *args,
                 **kwargs):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
        """
        self.trainset_dir = trainset_dir
        self.few_shot_dir = few_shot_dir
        self.num_shot = num_shot
        self.noisy_dataset = SeparDataset(trainset_dir, only_clean = True)
        self.few_shot_data_list = os.listdir(os.path.join(few_shot_dir, "audio"))[:num_shot]
    
    def __getitem__(self, index):

        # 随机选择一个few_shot数据
        few_shot_index = random.randint(0, self.num_shot-1)
        few_shot_data_file = self.few_shot_data_list[few_shot_index]
        noisy = self.noisy_dataset[index]

        clean_audio, _ = torchaudio.load(os.path.join(self.few_shot_dir,
                                                      "audio",
                                                      few_shot_data_file))

        radar_file = os.path.join(self.few_shot_dir,
                                  "radar",
                                  few_shot_data_file.replace("wav", "npy"))

        mix_audio = random_mix_audio(clean_audio, noisy)
        label = int(few_shot_data_file.split("_")[0])

        radar = torch.tensor(np.load(radar_file),
                             dtype = torch.float32)
        return radar, clean_audio, mix_audio, torch.tensor(label)

    def __len__(self):
        return self.noisy_dataset.__len__()


class FewShotTestDataset:


    def __init__(self,
                 dataset_dir,
                 *args,
                 **kwargs):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
        """
        
        self.dataset_dir = dataset_dir
        self.sample_list = os.listdir(os.path.join(self.dataset_dir, "s1_radioses"))

    def __getitem__(self, index):
        sample_name = self.sample_list[index]

        radar_file = os.path.join(self.dataset_dir, f"s{1}_radioses", sample_name)
        radar = torch.tensor(np.load(radar_file),
                             dtype = torch.float32)
        label = sample_name.split("-")[0]
        label = int(label.split("_")[0])
        clean_audio_file = os.path.join(self.dataset_dir,
                                        f"s{1}",
                                        sample_name.replace("npy", "wav"))
        mix_audio_file = os.path.join(self.dataset_dir,
                                      f"mix_clean",
                                      sample_name.replace("npy", "wav"))
        clean_audio, _ = torchaudio.load(clean_audio_file)
        mix_audio, _ = torchaudio.load(mix_audio_file)
        return radar, clean_audio, mix_audio, torch.tensor(label)

    def __len__(self):
        return len(self.sample_list)


# class FewShotTestDataset:

#     def __init__(self,
#                  trainset_dir,
#                  few_shot_dir,
#                  num_shot,
#                  *args,
#                  **kwargs):
#         """
#         Initialize the Dataset object.

#         Args:
#             dataset_dir (str): The directory path of the dataset.
#         """
#         self.trainset_dir = trainset_dir
#         self.few_shot_dir = few_shot_dir
#         self.num_shot = num_shot
#         self.noisy_dataset = SeparDataset(trainset_dir, only_clean = True)
#         self.few_shot_data_list = os.listdir(os.path.join(few_shot_dir, "audio"))
    
#     def __getitem__(self, index):

#         # 随机选择一个few_shot数据
#         few_shot_index = index
#         few_shot_data_file = self.few_shot_data_list[few_shot_index]
#         noisy = self.noisy_dataset[index]

#         clean_audio, _ = torchaudio.load(os.path.join(self.few_shot_dir,
#                                                       "audio",
#                                                       few_shot_data_file))

#         radar_file = os.path.join(self.few_shot_dir,
#                                   "radar",
#                                   few_shot_data_file.replace("wav", "npy"))

#         mix_audio = random_mix_audio(clean_audio, noisy)
#         label = int(few_shot_data_file.split("_")[0])

#         radar = torch.tensor(np.load(radar_file),
#                              dtype = torch.float32)
#         return radar, clean_audio, mix_audio, torch.tensor(label)

#     def __len__(self):
#         return len(self.few_shot_data_list)

class FewShotInitDataset:

    def __init__(self,
                 few_shot_dir,
                 num_shot,
                 *args,
                 **kwargs):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
        """
        self.few_shot_dir = few_shot_dir
        self.num_shot = num_shot
        self.few_shot_data_list = os.listdir(os.path.join(few_shot_dir, "radar"))[:num_shot]
    
    def __getitem__(self, index):

        few_shot_data_file = self.few_shot_data_list[index]
        radar_file = os.path.join(self.few_shot_dir,
                                  "radar",
                                  few_shot_data_file)

        label = int(few_shot_data_file.split("_")[0])

        radar = torch.tensor(np.load(radar_file),
                             dtype = torch.float32)
        return radar, torch.tensor(label)

    def __len__(self):
        return len(self.few_shot_data_list)
