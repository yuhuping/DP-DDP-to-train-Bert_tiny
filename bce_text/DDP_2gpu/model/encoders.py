import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from .modules import TransformerEncoder


class User_Encoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(User_Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)


class Text_Encoder_mean(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder_mean, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        input_mask_expanded = text_attmask.unsqueeze(-1).expand(hidden_states.size()).float()
        mean_output = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_output = self.fc(mean_output)
        return self.activate(mean_output)


class Text_Encoder(torch.nn.Module): # 使用预训练的 BERT 模型来编码文本数据，并将其转换为项目（item）嵌入
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder, self).__init__()
        self.bert_model = bert_model
        # 线性层用于将 BERT 模型的输出-词嵌入维度 word_embedding_dim转换为项目嵌入item_embedding_dim
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim) 
        self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        cls = self.fc(hidden_states[:, 0])
        return self.activate(cls)


class Bert_Encoder(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(Bert_Encoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 2,
            'abstract': args.num_words_abstract * 2,
            'body': args.num_words_body * 2
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())               # 将 attributes2length 的值转换为列表
                [:list(self.attributes2length.keys()).index(key)]   # 获取当前键之前的所有键的索引
            )
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'abstract', 'body']

        if 'opt' in args.bert_model_load:
            self.text_encoders = nn.ModuleDict({
                'title': Text_Encoder_mean(bert_model, args.embedding_dim, args.word_embedding_dim)
            })
        else:
            self.text_encoders = nn.ModuleDict({
                    'title': Text_Encoder(bert_model, args.embedding_dim, args.word_embedding_dim)
            })

        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]

    def forward(self, news):
        text_vectors = [
            self.text_encoders['title'](
                # torch.narrow 的第二个参数是维度，第三个参数是起始位置，第四个参数是长度。
                torch.narrow(news, 1, 0, 60)) 
            for name in self.newsname
        ]
        if len(text_vectors) == 1:
            final_news_vector = text_vectors[0]
        else:
            final_news_vector = torch.mean(torch.stack(text_vectors, dim=1), dim=1)
        return final_news_vector