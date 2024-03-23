import argparse


def get_config():

    parser = argparse.ArgumentParser()

    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')

    # dataset
    parser.add_argument('--train', type=str, default=r'G:\my_radar_sound\few_shot_data\train', help='Train path')
    parser.add_argument('--few_shot_dir', type=str, default=r'G:\my_radar_sound\few_shot_data\few_shot\16', help='few_shot data path')
    parser.add_argument('--few_shot_val', type=str, default=r"G:\my_radar_sound\few_shot_data\test\16", help='few_shot test data')
    parser.add_argument('--num_shot', type=int, default=1, help='few_shot num')

    #basic 
    parser.add_argument('--model_path', type=str, default='log/24-03-23-20-02-58/model/best_train.pth', help='Model path')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=300, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Checkpoint')

    # device
    parser.add_argument('--device', type=str, default='cuda:0', help='Gpu device')
    parser.add_argument('--env', type=str, default='local', help='Enviornment')
    parser.add_argument('--num_worker', type=int, default=4, help='Num workers')

    parser.add_argument("--val_per_epoch", type=int, default=1, help="")
    arguments = parser.parse_args()

    return arguments