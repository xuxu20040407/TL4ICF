import torch
import utils, networks, train


def main():
    args = utils.prepare_args()
    model = networks.get_model(args)

    # Load the lower fidelity model for transfer learning
    if args.mode == 'tl': 
        tl_load_path = utils.get_tl_load_path(args)
        model.load_state_dict(torch.load(tl_load_path)) 

    # The loading method varies depending on args.mode
    train_data, val_data = utils.get_data(args) 

    optimizer = utils.get_optimizer(args.optimizer, model, args.lr)
    loss_fn = utils.get_loss_fn(args.loss_fn)
    device = utils.get_device(args.device)
    save_path, log_path = utils.get_save_log_path(args)

    train.train_model(model, train_data, val_data, optimizer, loss_fn, device, 
                      args.epochs, args.val_interval, save_path, log_path)


if __name__ == "__main__":
    main()
