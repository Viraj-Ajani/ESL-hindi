CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \

python3 train.py \
  --data_path Flickr30K/ \
  --data_name f30k_precomp \
  --logger_name murillog \
  --model_name murilcheckpoint \
  --batch_size 32 \
  --num_epochs 30 \
  --lr_update 15 \
  --learning_rate 0.0005 \
  --precomp_enc_type basic \
  --workers 10 \
  --log_step 200 \
  --embed_size 512 \
  --vse_mean_warmup_epochs 1 \
  --kernel_size 2 \
  --token "google/muril-base-cased"
  
python3 test.py "./murilcheckpoint/model_best.pth" "google/muril-base-cased"


python3 train.py \
  --data_path Flickr30K/ \
  --data_name f30k_precomp \
  --logger_name mBERTlog \
  --model_name mBERTcheckpoint \
  --batch_size 32 \
  --num_epochs 30 \
  --lr_update 15 \
  --learning_rate 0.0005 \
  --precomp_enc_type basic \
  --workers 10 \
  --log_step 200 \
  --embed_size 512 \
  --vse_mean_warmup_epochs 1 \
  --kernel_size 2 \
  --token "google-bert/bert-base-multilingual-cased"
  
python3 test.py "./mBERTcheckpoint/model_best.pth" "google-bert/bert-base-multilingual-cased"


python3 train.py \
  --data_path Flickr30K/ \
  --data_name f30k_precomp \
  --logger_name mmBERTlog \
  --model_name mmBERTcheckpoint \
  --batch_size 32 \
  --num_epochs 30 \
  --lr_update 15 \
  --learning_rate 0.0005 \
  --precomp_enc_type basic \
  --workers 10 \
  --log_step 200 \
  --embed_size 512 \
  --vse_mean_warmup_epochs 1 \
  --kernel_size 2 \
  --token "jhu-clsp/mmBERT-base"
  
python3 test.py "./mmBERTcheckpoint/model_best.pth" "jhu-clsp/mmBERT-base"
