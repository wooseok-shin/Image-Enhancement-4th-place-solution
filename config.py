import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default='1', type=str, help='experiment_number')
    parser.add_argument('--tag', default='Default', type=str, help='tag')

    # Path settings
    parser.add_argument('--pretrained_path', type=str, default='./pretrained_weight/HINet-GoPro.pth')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='10Folds', help='Folds folder')
    parser.add_argument('--fold', type=str, default='0', help='Validation Fold')
    parser.add_argument('--model_path', type=str, default='results/')

    # Model parameter settings
    parser.add_argument('--arch', type=str, default='HINet')

    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pred_batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--initial_lr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--clipping', type=float, default=1, help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=30, help="Scheduler ReduceLROnPlateau's parameter")

    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = getConfig()
    args = vars(args)
    print(args)
