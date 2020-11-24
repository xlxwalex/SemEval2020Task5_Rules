import re
import csv

def load_idx(file_path):
    idxs =[]
    with open(file_path, 'r+', encoding='utf-8') as f:
        idxline = f.readlines()
    for ele in idxline:
        idxs.append(eval(ele.strip()))
    return idxs

spec_token = ['�', '聳', '≥', '®', '́', '°']
spec_punc = ['$', '¥', '#', '@', '&', '%', '*', '//', '|', '`', '<', '>']
spec_link = ['-', '_']
spec_abri = {"'re": " are", "'ve": " have", "'ll": " will", "'m":" am", "n't": "not"}

def text_wash(sentence):
    sent = []
    for stoken in spec_token:
        sentence = sentence.replace(stoken, '')
    sentence.lower()
    for spunc in spec_punc:
        sentence = sentence.replace(spunc, '')
    for slink in spec_link:
        sentence = sentence.replace(slink, ' ')
    for sabri in spec_abri.keys():
        sentence = sentence.replace(sabri, spec_abri[sabri])
    find_float = "\d+(\.\d+)?"
    find_time  = "\d+(\:\d+)?"
    find_price = "(\d+(\,\d+)+)|((\d)+(bn|m|th|st|nd|rd))"
    find_tel   = "\+(\d)+"
    find_year  = "(\d)+s"
    find_multi = "(\!|\?|\.)+"
    sentence = re.sub(find_float, '', sentence)
    sentence = re.sub(find_time, '', sentence)
    sentence = re.sub(find_price, '', sentence)
    sentence = re.sub(find_tel, '', sentence)
    sentence = re.sub(find_year, '', sentence)
    sentence = re.sub(find_multi, '.', sentence)
    find_ie   = "(\(.+\))|(\[.+\])|(\<.+\>)"
    sentence = re.sub(find_ie, '.', sentence)
    find_all_num = "[0-9]*"
    sentence = re.sub(find_all_num, '', sentence)
    sentence = sentence.replace('''"''', '')
    sentence = sentence.replace("'", '')
    sentence = sentence.lstrip()
    return sentence

def read_file(file_path):
    label, sentence = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    file.close()
    head = True
    for item in data:
        if head:
            head = False
            continue
        elemets = item.split(',', maxsplit=2)
        label.append(eval(elemets[1]))
        sentence.append(elemets[-1].strip())
    return label, sentence

if __name__ == '__main__':
    print(read_file('./data/subtask1_train.txt')[1][:5])

