import numpy as np
import torch


def read_behaviors(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    Total_news_item_num = len(before_item_name_to_id)       # 获取新闻条目的总数
    before_item_counts = [0] * (Total_news_item_num + 1)    # 用于存储每个新闻条目有多少用户与其交互（新闻流行指数），初始值都为0
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0                                       # 初始化用户-新闻对计数器为0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]                      # 用户名称ID
            history_item_name = splited[1].split(' ')   # 获取用户的交互历史行为（新闻条目名称）
            if len(history_item_name) < min_seq_len:    # 如果历史行为的长度小于最小序列长度 min_seq_len，则跳过当前用户
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name] # 用字典将新闻条目ID名称转换为顺序id数
            user_seq_dic[user_name] = item_ids_sub_seq  # 用户的行为序列存储到 user_seq_dic 字典中。
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1  # 样本数据交互次数
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))
    # 处理新闻条目的ID映射和统计信息
    item_id = 1
    item_id_to_dic = {}         # 用新闻id数得到 新闻名称ID、标题、摘要
    item_name_to_id = {}
    item_id_before_to_now = {}  # 用来清洗没有被用户交互过的新闻
    for before_item_id in range(1, Total_news_item_num + 1):                    # 遍历总新闻条目id数
        if before_item_counts[before_item_id] != 0:                             # 该新闻条目的被交互次数不为0
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))
    # 处理用户行为数据 将用户的历史行为序列分割为训练集、验证集和测试集
    users_train = {}            # 用于训练的用户及其交互的新闻键值对
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    for user_name, item_seqs in user_seq_dic.items():               # 遍历每个用户名称ID（user_name）及对应交互新闻（item_seqs）
        user_seq = [item_id_before_to_now[i] for i in item_seqs]    # 将用户的行为序列中的新闻条目ID转换为顺序id数
        train = user_seq[:-2]
        valid = user_seq[-(max_seq_len+2):-1]
        test = user_seq[-(max_seq_len+1):]
        users_train[user_id] = train
        users_valid[user_id] = valid
        users_test[user_id] = test

        users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_news(news_path):
    item_id_to_dic = {}
    item_id_to_name = {}
    item_name_to_id = {}
    item_id = 1
    with open(news_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            doc_name, _, _ = splited
            item_name_to_id[doc_name] = item_id
            item_id_to_dic[item_id] = doc_name
            item_id_to_name[item_id] = doc_name
            item_id += 1
    return item_id_to_dic, item_name_to_id, item_id_to_name


def read_news_bert(news_path, args, tokenizer): # 从新闻数据文件中读取新闻条目，并将它们转换为BERT模型可以理解的格式(mind_60w_items.tsv有三列)
    item_id_to_dic = {}     # 用于存储每个新闻条目的顺序id和对应的文本表示（新闻ID、Title标题、abstarct摘要）
    item_id_to_name = {}
    item_name_to_id = {}    # 用于存储每个新闻条目的名称和对应的顺序id
    item_id = 1
    print(news_path)
    with open(news_path, "r", encoding='utf-8') as f:
        for line in f:
            splited = line.strip('\n').split('\t')  # 去除每行末尾的换行符 \n, 使用制表符 \t（水平制表）分割处理后的字符串
            doc_name, title, abstract = splited     # 新闻ID | 新闻标题 | 新闻摘要
            if 'title' in args.news_attributes:
                # 标题文本转换为BERT模型的输入格式
                title = tokenizer(title.lower(), max_length=args.num_words_title, padding='max_length', truncation=True) 
            else:
                title = []

            if 'abstract' in args.news_attributes:
                abstract = tokenizer(abstract.lower(), max_length=args.num_words_abstract, padding='max_length', truncation=True)
            else:
                abstract = []

            if 'body' in args.news_attributes:
                body = tokenizer(body.lower()[:2000], max_length=args.num_words_body, padding='max_length', truncation=True)
            else:
                body = []
            item_name_to_id[doc_name] = item_id
            item_id_to_name[item_id] = doc_name
            item_id_to_dic[item_id] = [title, abstract, body]
            item_id += 1
    return item_id_to_dic, item_name_to_id, item_id_to_name


def get_doc_input_bert(item_id_to_content, args):       # item_id_to_content映射新闻条目的ID到其内容
    item_num = len(item_id_to_content) + 1

    if 'title' in args.news_attributes:
        news_title = np.zeros((item_num, args.num_words_title), dtype='int32') # 数组用于存储新闻标题的BERT嵌入ID
        news_title_attmask = np.zeros((item_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((item_num, args.num_words_abstract), dtype='int32')
        news_abstract_attmask = np.zeros((item_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((item_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((item_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_attmask = None

    for item_id in range(1, item_num):
        title, abstract, body = item_id_to_content[item_id]

        if 'title' in args.news_attributes:
            news_title[item_id] = title['input_ids']
            news_title_attmask[item_id] = title['attention_mask']

        if 'abstract' in args.news_attributes:
            news_abstract[item_id] = abstract['input_ids']
            news_abstract_attmask[item_id] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[item_id] = body['input_ids']
            news_body_attmask[item_id] = body['attention_mask']

    return news_title, news_title_attmask, \
        news_abstract, news_abstract_attmask, \
        news_body, news_body_attmask


