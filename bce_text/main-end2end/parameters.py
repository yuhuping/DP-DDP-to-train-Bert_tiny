from data_utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--item_tower", type=str, default='modal')
    parser.add_argument("--root_data_dir", type=str, default='../../',)
    parser.add_argument("--dataset", type=str, default='dataset/MIND/')
    parser.add_argument("--behaviors", type=str, default='mind_60w_users.tsv')
    parser.add_argument("--cold_file", type=str, default='None')
    parser.add_argument("--new_file", type=str, default='None')
    parser.add_argument("--news", type=str, default='mind_60w_items.tsv')

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=5e-5)
    parser.add_argument("--l2_weight", type=float, default=0.01)
    parser.add_argument("--drop_rate", type=float, default=0.1)

    # ============== model parameters ==============
    parser.add_argument("--bert_model_load", type=str, default='bert_tiny')
    parser.add_argument("--freeze_paras_before", type=int, default=0)
    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_seq_len", type=int, default=5)

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--load_ckpt_name", type=str, default='None')
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=4)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", type=str, default='cuda')

    # ============== news information==============
    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--news_attributes", type=str, default='title')

    args = parser.parse_args()
    args.news_attributes = args.news_attributes.split(',')

    return args


if __name__ == "__main__":
    args = parse_args()
