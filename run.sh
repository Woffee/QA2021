now_time=`date +"%Y-%m-%d-%H-%M"`
log_file="${now_time}.log"

data_type="stackoverflow4"
test_num="v4"

qa_file="stackoverflow4/QA_list.txt"
doc_file="stackoverflow4/Doc_list.txt"

#weight_path="models/nn_weights_${data_type}_${test_num}.h5"
weight_path="ckpt/best_model.hdf5"

# == args for main model ==
input_dim=300
output_dim=300
hidden_dim=64
ns_amount=10
learning_rate=0.001
drop_rate=0.2
batch_size=32
epochs=200
output_length=1000

args="--data_type ${data_type} --qa_file ${qa_file} --doc_file ${doc_file} --weight_path ${weight_path} --input_dim ${input_dim} --output_dim ${output_dim} --hidden_dim ${hidden_dim} --ns_amount ${ns_amount} --learning_rate ${learning_rate} --drop_rate ${drop_rate} --batch_size ${batch_size} --epochs ${epochs} --output_length ${output_length} --log_file ${log_file} --doc_mode 1"
echo $args

# == args for learning to rank ==
ltr_train_file="for_ltr/QA2021_ltr_${data_type}_train.txt"
ltr_vali_file="for_ltr/QA2021_ltr_${data_type}_vali.txt"
ltr_eval_file="for_ltr/QA2021_ltr_${data_type}_eval.txt"

ltr_qrel_file="data/QA2021_${data_type}_qrel.txt"
ltr_pred_file="data/QA2021_${data_type}_${test_num}_ltr_pred.txt"

#echo "=== Step 1 train NN model"
python main.py $args

#echo "=== Step 2 Training Learning to rank"
python learning2rank.py $args --ltr_train_file $ltr_train_file --ltr_vali_file $ltr_vali_file --ltr_eval_file $ltr_eval_file --ltr_qrel_file $ltr_qrel_file --ltr_pred_file $ltr_pred_file

#echo "=== Step 3 Evaluation"
python evaluation.py --qrel $ltr_qrel_file --run $ltr_pred_file
