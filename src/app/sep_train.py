

import os

import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from utils.metric import sisnr

from .train import Trainer


class SepTrainer(Trainer):

    def __init__(self, model, data, args):
        super().__init__(model, data, args)

    def process_data(self, batch_data):
        radar = batch_data["radar"].to(self.args.device)
        label = batch_data["label"].to(self.args.device)
        mix_audio = batch_data["mix"].to(self.args.device)
        clean_audio = batch_data["clean"].to(self.args.device)

        radar = radar/(torch.std(radar)+1e-8)
        mix_audio = mix_audio/(torch.std(mix_audio)+1e-8)
        clean_audio = clean_audio/(torch.std(clean_audio)+1e-8)

        return {
            "radar": radar,
            "mix_audio": mix_audio,
            "clean_audio":clean_audio,
            "label": label
        }
       
    def run_batch(self, batch_data):
        batch_data = self.process_data(batch_data)

        est_audio, embedding_loss = self.model(batch_data["mix_audio"],
                                               batch_data["radar"],
                                               batch_data["label"])
        est_loss = torch.mean(-sisnr(est_audio, batch_data["clean_audio"]))
        ori_sisnr = torch.mean(sisnr(batch_data["mix_audio"], batch_data["clean_audio"]))
        if embedding_loss:
            loss = est_loss + embedding_loss *0.1
        else:
            loss = est_loss
            embedding_loss = torch.tensor(0)
        loss = {
            "loss":loss,
            "est_loss":est_loss,
            "embedding_loss":embedding_loss,
            "ori_sisnr":ori_sisnr
        }
        return loss, est_audio / (torch.max(est_audio) + 1e-8)


class SepTester(SepTrainer):
    def __init__(self, model, data, args):
        args.checkpoint = True
        super().__init__(model, data, args)
    
    def save_wav(self, sample_id, **kwargs):
        for key, audio in kwargs.items():
            save_path = f"{sample_id}/{key}"
            audio = audio.squeeze(0)
            audio = audio.to(torch.float)
            os.makedirs(os.path.join("result", save_path), exist_ok=True)
            for i in range(audio.shape[0]):
                temp_audio = audio[i].unsqueeze(0).cpu().detach()
                torchaudio.save(os.path.join("result", save_path,f"{i}.wav"), temp_audio, 8000)
    
    def test(self):
        self.model.eval()
        data_loader = self.val_loader
        total_loss = 0
        total_snr = 0
        ori_sisnr = 0
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(data_loader)):
                loss, est_audio = self.run_batch(batch_data)
                total_loss += loss["loss"].item()
                temp_snr = -loss["est_loss"].item()
                ori_sisnr += loss["ori_sisnr"].item()
                total_snr += temp_snr
                self.save_wav(i, est_audio=est_audio,
                              clean_audio=batch_data["clean"],
                              noise_audio=batch_data["mix"])
        print(f"ori_snr: {ori_sisnr/(i+1)}")
        print(f"snr: {total_snr/(i+1)}")