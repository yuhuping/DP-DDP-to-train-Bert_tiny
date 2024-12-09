import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Log_file.info(('load bert model...'))
    # Load pre-trained BERT model and tokenizer
    bert_model_load = r"E:\IDvs.MoRec-main\pretrained_models\bert_tiny" # bert_model_load预训练模型的文件夹绝对路径
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
    print(f'len:{len(list(bert_model.parameters()))}')
    total_params = sum(p.numel() for p in bert_model.parameters())
    print(f"Total number of parameters: {total_params}")
    for index, (name, param) in enumerate(bert_model.named_parameters()):
        # print(f'index:{index}_name:{name}')
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
    # print(users_train[0])
    # Combine news information
    news_title, news_title_attmask, news_abstract, news_abstract_attmask, news_body, news_body_attmask = get_doc_input_bert(
        item_id_to_dic, args)               # news_title and news_title_attmask shape:[item_num+1, 30]

    item_content = np.concatenate([news_title, news_title_attmask], axis=1) # item_content shape:[item_num+1, 60]

    # Build dataset and dataloader
    train_dataset = BuildTrainDataset(u2seq=users_train, item_content=item_content, item_num=item_num,
                                      max_seq_len=args.max_seq_len, use_modal=True)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    # 使用Bert建立模型
    model = Model(args, item_num, True, bert_model).to(device)

    if True:
        bert_params = []
        recsys_params = []
        for index, (name, param) in enumerate(model.named_parameters()):         #  遍历模型的所有参数
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
            len(list(model.bert_encoder.text_encoders.title.bert_model.parameters())),
            len(list(model.parameters()))))
        # 'param_groups'是一个列表，包含了优化器管理的所有参数组
        for children_model in optimizer.state_dict()['param_groups']:
            Log_file.info("***** {} parameters have learning rate {}, weight_decay {} *****".format(
                len(children_model['params']), children_model['lr'], children_model['weight_decay']))

        model_params_require_grad = []
        model_params_freeze = []
        bert_params_require_grad = []
        bert_params_freeze = []
        
        for param_name, param_tensor in model.named_parameters():
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
        # 初始化 total_loss 为一个张量
        total_loss = torch.tensor(0.0, device=device)
        max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save = \
                run_eval(ep, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                         model, item_content, users_history_for_valid, users_valid, 512, item_num, True,
                         args.mode, False, device)

        for data in train_dl:
            sample_items, log_mask = data
            sample_items, log_mask = sample_items.to(device), log_mask.to(device)
            # print(sample_items.shape)  --> torch.Size([32, 21, 2, 60])  2为正负样本，60为30+30
            sample_items = sample_items.view(-1, sample_items.size(-1))
            # print(sample_items.shape)      # torch.Size([1344, 60]) 1344= 32 * 21 *2

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bz_loss = model(sample_items, log_mask, device)         # 将数据通过模型进行前向传播，计算当前批次的损失bz_loss。
                total_loss += bz_loss.to(device)

            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(total_loss):
                print("NaN loss encountered, stopping training.")
                break
        Log_file.info("Epoch {}/{}, Loss: {}".format(ep, args.epoch, bz_loss.item()))

        if not need_break and ep % 1 == 0:
            Log_file.info('')
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save = \
                run_eval(ep, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                         model, item_content, users_history_for_valid, users_valid, 512, item_num, True,
                         args.mode, False, device)
            model.train()
        
        # 保存模型，可以选择在每个epoch结束后保存或者根据需要调整
        if ep % 1 == 0:
            save_model(ep, model, model_dir, optimizer, args)

    # 最后保存一次模型
    save_model(ep, model, model_dir, optimizer, args)

def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
             model, item_content, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    item_embeddings = get_item_embeddings(model, item_content, batch_size, args, use_modal, local_rank)
    print(item_embeddings.shape)

    valid_Hit10 = eval_model(model, user_history, users_eval, item_embeddings, batch_size, args,
                             item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    need_break = False
    need_save = False
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
        need_save = True
    else:
        early_stop_count += 1
        if early_stop_count > 6:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save

if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数
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