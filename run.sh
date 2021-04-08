now_time=`date +"%Y-%m-%d-%H-%M"`
log_file="${now_time}.log"

data_type="stackoverflow3"
test_num="v3"

qa_file="/Users/woffee/www/emse-apiqa/our_code/stackoverflow3/QA_list.txt"
doc_file="/Users/woffee/www/emse-apiqa/our_code/stackoverflow3/Doc_list.txt"

model_path="models/nn_${data_type}_${test_num}.bin"
weight_path="models/nn_weights_${data_type}_${test_num}.h5"

input_dim=300
output_dim=300
hidden_dim=64
ns_amount=10
learning_rate=0.001
drop_rate=0.01
batch_size=32
epochs=200
output_length=1000

args="--data_type ${data_type} --qa_file ${qa_file} --doc_file ${doc_file} --model_path ${model_path} --weight_path ${weight_path} --input_dim ${input_dim} --output_dim ${output_dim} --hidden_dim ${hidden_dim} --ns_amount ${ns_amount} --learning_rate ${learning_rate} --drop_rate ${drop_rate} --batch_size ${batch_size} --epochs ${epochs} --output_length ${output_length} --log_file ${log_file}"
echo $args

#echo "=== Step 1 word2vec"
python word2vec.py --data_type $data_type --input_dim $input_dim --qa_file ${qa_file} --doc_file ${doc_file}

#echo "=== Step 2 nn_model"
python nn_model.py $args


to_train_file="for_ltr/ltr_${data_type}_${test_num}.txt"
to_test_file="for_ltr/ltr_${data_type}_${test_num}.txt"


# echo "=== Step 3 prepare_learning2rank_data"
#python prepare_learning2rank_data.py $args --to_train_file $to_train_file --to_test_file $to_test_file


model_type="lambdaMART"
ranklib_path="/Users/woffee/www/gis_technical_support/gis_qa/ltrdemo2wenbo/utils/bin/RankLib.jar"

#echo "=== Step 4 Training Learning to rank"
#python learning2rank.py --data_type $data_type --model_type $model_type --train_file ${to_train_file} --ranklib_path $ranklib_path --train
#python learning2rank.py --data_type $data_type --model_type $model_type --test_file ${to_test_file} --ranklib_path $ranklib_path --pred

# Evaluation

# AUC:
#python evaluation.py --data_type $data_type --model_type $model_type

#echo "=== Step 5 Evaluation"
#pred_file="ltr/predicts/stackoverflow3_lambdaMART_MAP_pred.txt"
#echo --data_type $data_type --qa_file ${qa_file} --doc_file ${doc_file} --pred_file ${pred_file} --qrel_file ${to_qrel_file}
#python map_mrr.py --data_type $data_type --qa_file ${qa_file} --doc_file ${doc_file} --pred_file ${pred_file} --qrel_file ${to_qrel_file}



