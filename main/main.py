# TODO: tensorboard

import torch

from utils import prepare_args, data_loader, get_optimizer, get_loss_fn
from networks import get_model
from train import train_model


def main():
    args = prepare_args()

    model = get_model(args)

    if args.mode == 'train':
        train_data, val_data = data_loader(args.dist, args.fidelity)
    elif args.mode == 'tl':
        model.load_state_dict(torch.load(f'{args.load_path}model.pth'))
        train_data, val_data = data_loader(args.dist, args.fidelity, tl=True)
    else:
        raise ValueError("Invalid mode. Please choose either 'train' or 'tl'.")
    
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    loss_fn = get_loss_fn(args.loss_fn)

    train_model(model, train_data, val_data, optimizer, loss_fn, 
                args.device, args.epochs, args.val_interval, args.save_path)


if __name__ == "__main__":
    main()
