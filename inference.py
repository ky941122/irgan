#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import jieba
import tensorflow as tf
import numpy as np
import Generator
import Discriminator


# Data
tf.flags.DEFINE_string("train_data", "data/id_data_sort", "train data (id)")
tf.flags.DEFINE_string("dev_data", "data/dev/id_dev_2w", "dev data (id)")
tf.flags.DEFINE_integer("vocab_size", 22511, "word vocab size")
# Model Hyperparameters
tf.flags.DEFINE_integer("data_every_round", 20000, "every round update Dis and Gen with how many data.")
tf.flags.DEFINE_integer("max_sequence_length", 16, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5,7,9", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.05, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 500, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("pools_size", 500, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("gen_pools_size", 20, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("g_epochs_num", 1, " the num_epochs of generator per epoch")
tf.flags.DEFINE_integer("d_epochs_num", 1, " the num_epochs of discriminator per epoch")
tf.flags.DEFINE_integer("sampled_size", 500, " the real selectd set from the The sampled pools")
tf.flags.DEFINE_integer("sampled_temperature", 20, " the temperature of sampling")
tf.flags.DEFINE_integer("gan_k", 5, "he number of samples of gan")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Save Model
tf.flags.DEFINE_string("model_name", "model", "model name")
tf.flags.DEFINE_integer("num_checkpoints", 2000, "checkpoints number to save")
tf.flags.DEFINE_boolean("restore_model", False, "Whether restore model or create new parameters")
tf.flags.DEFINE_string("model_path", "runs", "Restore which model")
tf.flags.DEFINE_integer("init_loop", 0, "restore model from checkpoint and continue traing with former loop.")

FLAGS = tf.flags.FLAGS

def build_vocab(word_list):
    vocab = dict()
    id2tok = dict()
    f = open(word_list, 'r')
    for line in f.readlines():
        line = line.strip("\n")
        token, id = line.split("\t#\t")
        token = token.strip()
        token = token.decode("utf-8")
        id = id.strip()
        id = int(id)
        if token not in vocab:
            vocab[token] = id

        if id not in id2tok:
            id2tok[id] = token.encode("utf-8")
    print "build vocab done"
    return vocab, id2tok


def read_ans(file_name, seq_len):
    ans = []
    f = open(file_name, 'r')
    for line in f.readlines():
        line = line.strip()
        _, stdq = line.split("\t")
        stdq = stdq.strip().split()
        stdq = stdq[:seq_len]
        stdq = stdq + [1] * (seq_len - len(stdq))

        if stdq not in ans:
            ans.append(stdq)

    return ans


def tok2id(string, seq_len, vocab):
    ids = []
    string = string.strip().strip("_").strip()
    toks = string.split("_")
    for tok in toks:
        id = vocab.get(tok, 0)   #0是<unk>，1是<pad>
        ids.append(id)
    ids = ids[:seq_len]
    ids = ids + [1] * (seq_len - len(ids))

    return ids


def de_id(ids, id2tok):
    toks = []
    ids = [int(id) for id in ids]
    for id in ids:
        tok = id2tok[id]
        toks.append(tok)
    line = " ".join(toks)
    return line


def inference(word_list, user_dict, train, ckpt_path, k=10, model_type="Dis"):
    k = int(k)
    tokenizer = jieba.Tokenizer()
    tokenizer.load_userdict(user_dict)
    vocab, id2tok = build_vocab(word_list)   #vocab里token是unicode， id2tok里tok是str， 两个里面id都是int
    alist = read_ans(train, FLAGS.max_sequence_length)  #是个二维list

    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)

            with sess.as_default():

                param = None
                loss_type = "pair"
                discriminator = Discriminator.Discriminator(
                    sequence_length=FLAGS.max_sequence_length,
                    batch_size=FLAGS.batch_size,
                    vocab_size=FLAGS.vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    learning_rate=FLAGS.learning_rate,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    # embeddings=embeddings,
                    embeddings=None,
                    paras=param,
                    loss=loss_type)

                generator = Generator.Generator(
                    sequence_length=FLAGS.max_sequence_length,
                    batch_size=FLAGS.batch_size,
                    vocab_size=FLAGS.vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    learning_rate=FLAGS.learning_rate * 0.1,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    # embeddings=embeddings,
                    embeddings=None,
                    paras=param,
                    loss=loss_type)

                saver = tf.train.Saver(tf.global_variables())
                sess.run(tf.global_variables_initializer())
                saver.restore(sess=sess, save_path=ckpt_path)

                if model_type == "Dis":
                    model = discriminator
                else:
                    model = generator

                while True:
                    print "Please input query:\n"
                    line = sys.stdin.readline().strip()
                    if not line:
                        line = "小米蓝牙手柄能连接手机玩吗"
                    ws = tokenizer.cut(line)  #切出来每个tok是unicode。
                    ws = list(ws)
                    q = "_".join(ws)

                    ws_enc = [tok.encode("utf-8") for tok in ws]
                    q_enc = "_".join(ws_enc)

                    print "tokenized query is:", q_enc

                    q = tok2id(q, FLAGS.max_sequence_length, vocab)  #是个list
                    print "id q is:", q

                    qs = []
                    for a in alist: #每个a是个list
                        qs.append(q)

                    feed_dict = {
                        model.input_x_1: qs,
                        model.input_x_2: alist,
                        model.input_x_3: alist
                    }

                    scorel2 = tf.reshape(model.score12, [-1])
                    topk = tf.nn.top_k(scorel2, k)

                    index = sess.run(topk, feed_dict)[1]

                    recalls = np.array(alist)[index]

                    print "Recall results are: \n"
                    for recall in recalls:
                        line = de_id(recall, id2tok)
                        print line, "\n"


if __name__ == "__main__":
    word_list = "data/word_list"
    user_dict = "data/userterms.dic"
    train = 'data/id_data_sort'
    args = sys.argv
    ckpt_path = args[1]
    if len(args) > 2:
        k = args[2]
        if len(args) > 3:
            model_type = args[3]
            inference(word_list, user_dict, train, ckpt_path, k, model_type)
        else:
            inference(word_list, user_dict, train, ckpt_path, k)
    else:
        inference(word_list, user_dict, train, ckpt_path)

