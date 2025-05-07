#mosi
for key in Has0_acc_2 Non0_acc_2 Mult_acc_5 Mult_acc_7 MAE
do
  CUDA_VISIBLE_DEVICES=0 python -u robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval $key | tee -a ./log/eval_mosi.txt 2>&1 &
  wait
done

# mosei
for key in Has0_acc_2 Non0_acc_2 Mult_acc_5 Mult_acc_7 MAE
do
  CUDA_VISIBLE_DEVICES=0 python -u robust_evaluation.py --config_file configs/eval_mosei.yaml --key_eval $key | tee -a ./log/eval_mosei.txt 2>&1 &
  wait
done

#### sims
for key in Mult_acc_2 Mult_acc_3 Mult_acc_5 MAE
do
  CUDA_VISIBLE_DEVICES=0 python -u robust_evaluation.py --config_file configs/eval_sims.yaml --key_eval $key | tee -a ./log/eval_sims.txt 2>&1 &
  wait
done

