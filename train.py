import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results, interval_time, get_parameter_number
from models.RoCAF import build_model
from core.metric import MetricsTop
from tqdm import tqdm
import time


start = time.time()
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='')
parser.add_argument('--seed', type=int, default=-1)
opt = parser.parse_args()
print(opt)

def main():
    best_valid_results, best_test_results = {}, {}

    config_file = 'configs/train_mosi.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("ckpt root :", ckpt_root)

    model = build_model(args).to(device)

    print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))
    dataLoader = MMDataLoader(args)

    params = []
    for name, param in model.named_parameters():
        lr = args['base']['lr']
        if 'bertmodel' in name:
            lr = 0.000005
        params.append({'params': param, 'lr': lr})
    optimizer = torch.optim.AdamW(params, weight_decay=args['base']['weight_decay'])

    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = MultimodalLoss(args)

    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    for epoch in range(1, args['base']['n_epochs'] + 1):
        print(f'Training Epoch: {epoch}')
        start_time = time.time()
        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))
        train(model, train_loader, optimizer, loss_fn, epoch, metrics)
        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root,
                                                  seed, 'valid',save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed,'test',
                                             save_best_model=True)

        end_time = time.time()
        epoch_mins, epoch_secs = interval_time(start_time, end_time)
        print("Epoch: {}/{} | Current Best Test Results: {} | \n Time: {}m {}s".format(epoch, args['base']['n_epochs'],
                                                                                    best_test_results, epoch_mins,
                                                                                    epoch_secs))
        scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}

        out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item()
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {key: value / (cur_iter + 1) for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')


def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()

    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}
        with torch.no_grad():
            out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item()
                except:
                    loss_dict[key] += value

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    return results


if __name__ == '__main__':
    main()
print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))