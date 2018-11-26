#coding=utf-8
import numpy as np


def read_raw(filename, seq_len):
    raws = []
    f = open(filename, 'r')
    for line in f.readlines():
        raw = []
        line = line.strip()
        userq, stdq = line.split("\t")
        userq, stdq = userq.strip(), stdq.strip()
        userq = userq.strip().split()
        userq = userq[:seq_len]
        userq = userq + [1] * (seq_len - len(userq))
        raw.append(userq)

        stdq = stdq.strip().split()
        stdq = stdq[:seq_len]
        stdq = stdq + [1] * (seq_len - len(stdq))
        raw.append(stdq)

        raws.append(raw)

    print("read raw data done")
    return raws



def read_alist(filename, seq_len):
    alist = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip()
        _, stdq = line.split("\t")
        stdq = stdq.strip().split()
        stdq = stdq[:seq_len]
        stdq = stdq + [1] * (seq_len - len(stdq))
        alist.append(stdq)

    print("read alist done")
    return alist


def read_dev(file_name, seq_len):
    dev_data = dict()
    f = open(file_name, 'r')
    for line in f.readlines():
        line = line.strip()
        userq, stdq = line.split("\t")

        userq = userq.strip()

        stdq = stdq.strip().split()
        stdq = stdq[:seq_len]
        stdq = stdq + [1] * (seq_len - len(stdq))

        if userq not in dev_data:   # dict的key不能是list，因此userq保留string的形式
            dev_data[userq] = [stdq]
        else:
            dev_data[userq].append(stdq)

    return dev_data


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

