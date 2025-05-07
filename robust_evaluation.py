import os
import torch
import yaml
import argparse
from core.dataset import MMDataEvaluationLoader
from models.RoAF import build_model
from core.metric import MetricsTop
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--key_eval', type=str, default='') 
parser.add_argument('--type', type=str, default='')
opt = parser.parse_args()
print(opt)


def main():
    config_file = 'configs/eval_sims.yaml' if opt.config_file == '' else opt.config_file
    
    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    dataset_name = args['dataset']['datasetName']
    key_eval = args['base']['key_eval'] if opt.key_eval == '' else opt.key_eval
    type = args['base']['type'] if opt.type == '' else opt.type

    model = build_model(args).to(device)
    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(dataset_name)

    missing_rate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    result_avg1 =0
    result_arg2 = 0
    for cur_r in tqdm(missing_rate_list, desc="Missing Rates", unit="rate"):
        test_results_list = []
        if dataset_name == 'sims':
            for _, cur_seed  in enumerate([1111, 1112, 1113]):
                best_ckpt = os.path.join(f'ckpt/{dataset_name}/best_{key_eval}_{cur_seed}.pth')

                model.load_state_dict(torch.load(best_ckpt)['state_dict'])

                args['base']['missing_rate_eval_test'] = cur_r # Set missing rate
                dataLoader = MMDataEvaluationLoader(args)
                test_results_cur_seed = evaluate(model, dataLoader, metrics)
                # print(f'{cur_seed}: {test_results_cur_seed}'
                test_results_list.append(test_results_cur_seed)

            if key_eval == 'Mult_acc_2':
                Mult_acc_2_avg = (test_results_list[0]['Mult_acc_2'] + test_results_list[1]['Mult_acc_2'] + test_results_list[2]['Mult_acc_2']) / 3
                F1_score_avg = (test_results_list[0]['F1_score'] + test_results_list[1]['F1_score'] + test_results_list[2]['F1_score']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {round(Mult_acc_2_avg,4)}, F1_score_avg: {round(F1_score_avg,4)}')
                result_avg1 += Mult_acc_2_avg
                result_arg2 += F1_score_avg
            elif key_eval == 'Mult_acc_3':
                Mult_acc_3_avg = (test_results_list[0]['Mult_acc_3'] + test_results_list[1]['Mult_acc_3'] + test_results_list[2]['Mult_acc_3']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_3_avg: {round(Mult_acc_3_avg,4)}')
                result_avg1 += Mult_acc_3_avg
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'] + test_results_list[1]['Mult_acc_5'] + test_results_list[2]['Mult_acc_5']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {round(Mult_acc_5_avg,4)}')
                result_avg1 += Mult_acc_5_avg
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] + test_results_list[1]['MAE'] + test_results_list[2]['MAE']) / 3
                Corr_avg = (test_results_list[0]['Corr'] + test_results_list[1]['Corr'] + test_results_list[2]['Corr']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {round(MAE_avg,4)}, Corr_avg: {round(Corr_avg,4)}')
                result_avg1 += MAE_avg
                result_arg2 += Corr_avg

        else:
            seed_list = [1111]
            # for _, cur_seed  in enumerate([1111, 1112, 1113]):
            for _, cur_seed  in enumerate(seed_list):
                best_ckpt = os.path.join(f'ckpt/{dataset_name}/best_{type}_{key_eval}_{cur_seed}.pth')
                model.load_state_dict(torch.load(best_ckpt)['state_dict'])
                args['base']['missing_rate_eval_test'] = cur_r # Set missing rate
                dataLoader = MMDataEvaluationLoader(args)
                test_results_cur_seed = evaluate(model, dataLoader, metrics)
                test_results_list.append(test_results_cur_seed)
            # if len(seed_list) == 1:
            #     print(f'key_eval: {key_eval}, missing rate: {cur_r}')
            #     print(test_results_list)
            #     continue
            if key_eval == 'Has0_acc_2':
                Has0_acc_2_avg = (test_results_list[0]['Has0_acc_2'])
                Has0_F1_score_avg = (test_results_list[0]['Has0_F1_score'])
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {round(Has0_acc_2_avg,4)}, F1_score_avg: {round(Has0_F1_score_avg,4)}')
                result_avg1 += Has0_acc_2_avg
                result_arg2 += Has0_F1_score_avg
            elif key_eval == 'Non0_acc_2':
                Non0_acc_2_avg = (test_results_list[0]['Non0_acc_2'])
                Non0_F1_score_avg = (test_results_list[0]['Non0_F1_score'])
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Non0_acc_2_avg: {round(Non0_acc_2_avg,4)}, Non0_F1_score_avg: {round(Non0_F1_score_avg,4)}')
                result_avg1 += Non0_acc_2_avg
                result_arg2 += Non0_F1_score_avg
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'])
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {round(Mult_acc_5_avg,4)}')
                result_avg1 += Mult_acc_5_avg
            elif key_eval == 'Mult_acc_7':
                Mult_acc_7_avg = (test_results_list[0]['Mult_acc_7'] + test_results_list[1]['Mult_acc_7'] + test_results_list[2]['Mult_acc_7']) / len(seed_list)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_7_avg: {round(Mult_acc_7_avg,4)}')
                result_avg1 += Mult_acc_7_avg
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] + test_results_list[1]['MAE'] + test_results_list[2]['MAE']) / len(seed_list)
                Corr_avg = (test_results_list[0]['Corr'] + test_results_list[1]['Corr'] + test_results_list[2]['Corr']) / len(seed_list)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {round(MAE_avg,4)}, Corr_avg: {round(Corr_avg,4)}')
                result_avg1 += MAE_avg
                result_arg2 += Corr_avg
    print(f'result_avg1: {round(result_avg1/10, 4)}, result_arg2: {round(result_arg2/10,4)}')
    # print(f'result_avg1: {result_avg1/10}, result_arg2: {result_arg2/10}')

def evaluate(model, eval_loader, metrics):
    y_pred, y_true = [], []

    model.eval()
    for cur_iter, data in enumerate(eval_loader):
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))
        sentiment_labels = data['labels']['M'].to(device)
        
        with torch.no_grad():
            out = model((None, None, None), incomplete_input)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(sentiment_labels.cpu())
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    return results



if __name__ == '__main__':
    main()
