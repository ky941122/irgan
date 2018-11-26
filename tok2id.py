def tok2id(tok, id_data, word_list):
    vocab = dict()
    f1 = open(word_list, 'r')
    for line in f1.readlines():
        line = line.strip()
        token, id = line.split("\t#\t")
        token = token.strip()
        id = id.strip()
        id = int(id)
        if token not in vocab:
            vocab[token] = id
    print("build vocab done")
    f1.close()

    f2 = open(tok, 'r')
    f3 = open(id_data, "w")
    for line in f2.readlines():
        line = line.strip()
        pid, levels, userq, stdq = line.split("#\t#")
        userq = userq.strip().strip("_").strip()
        userq = userq.split("_")
        userq_id = []
        for q in userq:
            q = q.strip()
            q_id = vocab.get(q, 0)
            q_id = str(q_id)
            userq_id.append(q_id)
        userq_id = " ".join(userq_id)

        stdq = stdq.strip().strip("_").strip()
        stdq = stdq.split("_")
        stdq_id = []
        for q in stdq:
            q = q.strip()
            q_id = vocab.get(q, 0)
            q_id = str(q_id)
            stdq_id.append(q_id)
        stdq_id = " ".join(stdq_id)

        writeLine = userq_id + "\t" + stdq_id
        f3.write(writeLine + "\n")

    f2.close()
    f3.close()


if __name__ == "__main__":
    tok = "/home/ky/work/irgan/Question-Answer/data/tokenized_data_sort"
    id_data = "data/id_data_sort"
    word_list = "/home/ky/work/irgan/Question-Answer/data/word_list"
    tok2id(tok, id_data, word_list)



