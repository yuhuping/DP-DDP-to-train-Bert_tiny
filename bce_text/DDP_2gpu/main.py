import torch
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import random
import os
from pathlib import Path
import sys
from parameters import parse_args
from model import Model
from data_utils import read_news,read_news_bert, read_behaviors, get_doc_input_bert, BuildTrainDataset, eval_model, get_item_embeddings
from data_utils.utils import *
# ======== DDP ========== # 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import warnings
warnings.filterwarnings('ignore')

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
    else:
        print("Only one GPU available.")

    Log_file.info(('load bert model...'))
    # Load pre-trained BERT model and tokenizer
    bert_model_load = 'E:\IDvs.MoRec-main\pretrained_models\bert_tiny' # bert_model_load预训练模型的文件夹绝对路径
    tokenizer = BertTokenizer.from_pretrained(bert_model_load)
    config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(bert_model_load, config=config)
    if 'base' in args.bert_model_load:
        pooler_para = [197, 198]
        args.word_embedding_dim = 768
    if 'tiny' in args.bert_model_load:
        pooler_para = [37, 38]          # pooler_para：这可能是用于特定层或参数的索引，用于后续的冻结或其他操作
        args.word_embedding_dim = 128   #args.word_embedding_dim：设置词嵌入的维度，这将影响模型的输入和输出维度
    # Freeze some parameters if needed
    for index, (name, param) in enumerate(bert_model.named_parameters()):
        if index < args.freeze_paras_before or index in pooler_para:    # index最大:199
            param.requires_grad = False                                 # 意味着在训练过程中这些参数将不会更新

    print('read news and behaviors')
    before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news_bert(
        os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)
    # 查看数据

    item_num, item_id_to_dic, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_dic,
                          before_item_name_to_id, before_item_id_to_name, args.max_seq_len, args.min_seq_len, Log_file)

    # Combine news information
    news_title, news_title_attmask, news_abstract, news_abstract_attmask, news_body, news_body_attmask = get_doc_input_bert(
        item_id_to_dic, args)               # news_title and news_title_attmask shape:[item_num+1, 30]

    item_content = np.concatenate([news_title, news_title_attmask], axis=1) # item_content shape:[item_num+1, 60]

    Log_file.info('build dataset...')
    # Build dataset and dataloader
    train_dataset = BuildTrainDataset(u2seq=users_train, item_content=item_content, item_num=item_num,
                                      max_seq_len=args.max_seq_len, use_modal=True)
    Log_file.info('build DDP sampler...')
    train_sampler = DistributedSampler(train_dataset)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=1, pin_memory=True, sampler=train_sampler)

    Log_file.info('build model...')
    # 使用Bert建立模型
    model = Model(args, item_num, True, bert_model).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if True:     # 打印需要训练的参数
        # 假设 model 是一个 DataParallel 对象
        original_model = model.module
        bert_params = []
        recsys_params = []
        for index, (name, param) in enumerate(original_model.named_parameters()):         #  遍历模型的所有参数
            if param.requires_grad:
                if 'bert_model' in name:
                    bert_params.append(param)
                else:
                    recsys_params.append(param)
        optimizer = optim.AdamW([
            {'params': bert_params, 'lr': args.fine_tune_lr, 'weight_decay': args.l2_weight},
            {'params': recsys_params, 'lr': args.lr, 'weight_decay': args.l2_weight}
        ])
        # 记录和报告模型中BERT部分的参数数量以及整个模型的参数数量--输出 199 parameters in bert, 228 parameters in model
        Log_file.info("***** {} parameters in bert, {} parameters in model *****".format(
            len(list(original_model.bert_encoder.text_encoders.title.bert_model.parameters())),
            len(list(original_model.parameters()))))
        # 'param_groups'是一个列表，包含了优化器管理的所有参数组
        for children_model in optimizer.state_dict()['param_groups']:
            Log_file.info("***** {} parameters have learning rate {}, weight_decay {} *****".format(
                len(children_model['params']), children_model['lr'], children_model['weight_decay']))

        model_params_require_grad = []
        model_params_freeze = []
        bert_params_require_grad = []
        bert_params_freeze = []
        for param_name, param_tensor in original_model.named_parameters():
            if param_tensor.requires_grad:                                  # 训练阶段Bert参数都要更新
                model_params_require_grad.append(param_name)
                if 'bert_model' in param_name:
                    bert_params_require_grad.append(param_name)
            else:
                model_params_freeze.append(param_name)
                if 'bert_model' in param_name:
                    bert_params_freeze.append(param_name)
        Log_file.info("***** freeze parameters before {} in bert *****".format(args.freeze_paras_before))
        Log_file.info("***** model: {} parameters require grad, {} parameters freeze *****".format(
            len(model_params_require_grad), len(model_params_freeze)))
        Log_file.info("***** bert: {} parameters require grad, {} parameters freeze *****".format(
            len(bert_params_require_grad), len(bert_params_freeze)))

    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0
    scaler = torch.cuda.amp.GradScaler()
    # Training loop
    model.train()
    for ep in range(args.epoch):
        Log_file.info("##### Epoch:{} #####".format(ep))
        loss, batch_index, need_break = 0.0, 1, False
        model.train()
        train_dl.sampler.set_epoch(ep)

        # train_epoch开始
        train_start_time = time.time()
        for data in train_dl:
            sample_items, log_mask = data
            sample_items, log_mask = sample_items.to(local_rank), log_mask.to(local_rank)
            sample_items = sample_items.view(-1, sample_items.size(-1))

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bz_loss = model(sample_items, log_mask, local_rank)         # 将数据通过模型进行前向传播，计算当前批次的损失bz_loss。
                bz_loss = bz_loss.mean()  # 确保 bz_loss 是标量
                loss += bz_loss.item()

            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(bz_loss):
                print("NaN loss encountered, stopping training.")
                need_break = True
                break
            if batch_index % 100 == 0: 
                Log_file.info('batch_index:{}/{}, batch loss: {:.5f}, sum loss: {:.5f}'.format(batch_index, len(train_dl), loss / batch_index, loss))
            batch_index += 1
        # train_epoch结束
        end_time = time.time()
        hour, minu, secon = get_time(train_start_time, end_time)
        Log_file.info("##### (time) train: {} hours {} minutes {} seconds #####".format(hour, minu, secon))

        # 保存模型，可以选择在每个epoch结束后保存或者根据需要调整
        if ep % 1 == 0:
            save_model(ep, model, model_dir, optimizer,
                   torch.get_rng_state(), torch.cuda.get_rng_state(), scaler, Log_file)
        if need_break:
            break

    # 最后保存一次模型
    save_model(ep, model, model_dir, optimizer,
                   torch.get_rng_state(), torch.cuda.get_rng_state(), scaler, Log_file)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数
    dist.init_process_group(backend='nccl')     # 初始化分布式训练环境，使用NCCL后端
    rank = dist.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    local_rank = int(local_rank)
    torch.cuda.set_device(local_rank)           # 设置当前使用的GPU
    setup_seed(12345)

     # 构建模型保存的目录路径
    model_dir = os.path.join('./checkpoint_bert')
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    # 更新标签屏幕参数，包含时间戳
    args.label_screen = args.label_screen + time_run
    # 构建日志和模型保存的标签
    dir_label = args.behaviors + ' ' + str(args.item_tower) + f'_Bert_freeze_{args.freeze_paras_before}'
    log_paras = f'Bert'
    # 设置日志记录器，用于文件和屏幕输出
    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, 0, args.behaviors) # setuplogger来自data_utils
    Log_file.info(args)

    train(args)