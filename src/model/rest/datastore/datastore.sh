Vicuna_PATH=/cpfs02/llm/shared/public/lvkai/workspace/sd/data/vicuna-7b-v1.5

# python3 get_datastore_chat.py --model-path $Vicuna_PATH # get datastore_chat_small.idx in this folder

python3 get_datastore_chat.py --model-path $Vicuna_PATH --large-datastore True # get datastore_chat_large.idx in  this folder