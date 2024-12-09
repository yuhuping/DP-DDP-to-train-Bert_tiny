# DP-DDP-to-train-Bert_tiny
使用Pytorch的DP和DDP来训练推荐系统文本模型的应用，使用bert文本模型搭建新闻推荐MIND数据推荐系统
本代码参考出处：https://github.com/westlake-repl/IDvs.MoRec

运行环境：
- torch == 1.7.1+cu110
- torchvision==0.8.2+cu110
- transformers==4.20.1

！！！ 运行前请讲main.py中bert_model_load路径设置为你的模型文件夹绝对路径 ！！！


如果你想运行在：
单机单卡：
cd ./bce_text/main-end2end
python main.py

DP单机双卡：
cd ./bce_text/DP_2gpu
torch.distributed.launch --nproc_per_node=2 main.py

DDP单机双卡：
cd ./bce_text/DDP_2gpu
torch.distributed.launch --nproc_per_node=2 main.py
