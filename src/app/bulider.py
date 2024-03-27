import torch
from torch.utils.data import DataLoader

from data import FewShotDataset, FewShotInitDataset, FewShotTestDataset, SeparDataset
from model import RadarMossFormer

from .sep_train import SepTester, SepTrainer


def collate_fn(batch):
    batch = [x for x in zip(*batch)]
    radar, clean_audio, mix_audio, label = batch

    return {
        "radar":torch.stack(radar,0),
        "clean":torch.stack(clean_audio,0),
        "mix":torch.stack(mix_audio,0),
        "label":torch.stack(label,0)
        }

def build_dataloader(args):
    val_loader = None
    if args.few_shot:
        val_dataset = FewShotTestDataset(args.few_shot_dataset['valset_dir'])
        val_loader    = DataLoader(val_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=args.num_worker,
                               collate_fn=collate_fn)
   
    dataloader = {"val":val_loader}
    if args.action == "train":
        if args.few_shot:
            train_dataset = FewShotDataset(**args.few_shot_dataset)
        else:
            train_dataset = SeparDataset(args.dataset_dir['train'])
        train_loader  = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_worker,
                                   collate_fn=collate_fn)
        dataloader = {"train":train_loader, "val":val_loader}
    
    return dataloader

def build_trainer(args, model, data):
    return SepTrainer(model, data, args)

def build_tester(args, model, data):
    return SepTester(model, data, args)

def bulid_model(args):
    model = RadarMossFormer(**args.model_config)
    if args.checkpoint or args.few_shot:
        print("load model from:", args.model_path)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])

    if args.action == "train" and args.few_shot:
        # freeze_model_parameters
        for p_name, param in model.named_parameters():
            if "person_embedding" in p_name or "adpter" in p_name:
                print("unfreeze:", p_name)
                continue
            param.requires_grad = False
            print("freeze:", p_name)
        model = model.to(args.device)
        init_datset = FewShotInitDataset(args.few_shot_dataset['few_shot_dir'],
                                         args.few_shot_dataset['num_shot'])
        radar_loader = DataLoader(init_datset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)
        new_embedding = []
        for batch_data in radar_loader:
            radar, label = batch_data
            radar = radar.to(args.device)
            label = label.to(args.device)
            radar = radar/(torch.std(radar)+1e-8)
            embedding, _ = model.radar_net.extract_radar_feature(radar)
            new_embedding.append(embedding)
        new_embedding = torch.cat(new_embedding,0)
        init_embedding = torch.mean(new_embedding,0)
        model.radar_net.init_embedding(init_embedding, label)

    model.to(args.device)
    return model