import numpy as np
import pandas as pd
import csv, copy, argparse, os, re
from gensim.summarization import keywords
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tqdm import tqdm
import math


class Wiki103():
    '''
    wikitext-103 csv file contents example

    = Robert Boulter =
    "Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as "" Craig "" in the episode "" Teddy 's Story "" of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the <unk> Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall ."
    "In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris <unk> , and Donkey Punch directed by Olly Blackburn . In May 2008 , Boulter made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , followed by an appearance on the television series Survivors in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as "" Kieron Fletcher "" . Boulter starred in the 2011 film Mercenaries directed by Paris <unk> ."
    = = Career = =
    = = = 2000 – 2005 = = =

    nltk stopword list
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


    '''

    def __init__(self, args):
        self.data_dir = args.wiki_path
        self.text_max_length = args.text_max_length
        self.text_min_length = args.text_min_length
        self.out_path = args.out_path
        self.is_one = args.is_one_sentence
        self.output_format = args.output_format
        self.except_stopword_list = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
                                      'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                                      'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
                                      'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'once',
                                      'all']
    # csv file read
    def read_csv(self, flag):
        if flag == 'train':
            file_name = 'wikitext-103-train.csv'
        elif flag == 'test':
            file_name = 'wikitext-103-test.csv'
        elif flag == 'valid':
            file_name = 'wikitext-103-valid.csv'
        else:
            raise AssertionError('Incorrect Flag value')

        path = self.data_dir + file_name
        with open(path, encoding='UTF-8') as f:
            data_reader = csv.reader(f)
            csv_data = [row for row in data_reader]
        return self.raw_csv_to_list(csv_data)

    # csv to list
    # 미리 설정한 minimum length 이하 text 제외
    def raw_csv_to_list(self, csv_data):
        para_list = []
        for lines in csv_data:
            paragraph = ''
            for line in lines:
                paragraph += line
            if len(paragraph) >= self.text_min_length:
                para_list.append(paragraph)
        return para_list

    # csv 파일은 단락으로 data가 나눠져 있음. 전처리를 두 가지 버전으로 진행. 단락 & 한 문장
    # 단락 -> 한 문장으로 변경하기
    def one_sentence(self, para_list):
        li = []
        print('making sentence....')
        for para in tqdm(para_list):
            sentences = para.split('.')
            for sentence in sentences:
                if sentence and len(sentence) >= 50:
                    li.append(sentence.strip() + '.')
        return li

    # 단락 중 max_length를 넘는 단락은 max_length 안에 있는 제일 마지막 마침표(.)를 찾아서, 그 이후의 text 삭제.
    def sentences(self, para_list):
        li = []
        print('makeing sentence....')
        for para in tqdm(para_list):
            if len(para) >= self.text_max_length:
                para = para[:self.text_max_length]
                index = -1
                idx = len(para) - 1
                while True:
                    index = para.find('.', index + 1)
                    if index == -1:
                        break
                    idx = index
                para = para[:idx + 1]
            li.append(para)
        return li

    def csv_to_list(self, flag):
        para_list = self.read_csv(flag)
        return self.one_sentence(para_list)

    # 전처리 된 데이터 파일로 저장
    def pickle_save(self, file_name, file):
        with open(os.path.join(self.out_path, f'{file_name}.txt'), 'wb') as f:
            pickle.dump(file, f)

    '''
    keyword 뽑아내고 pickle로 파일 저장하기
    하나의 csv 파일에서 단락 & 단락 키워드 & 문장 & 문장 키워드, 4개 list 파일 pickle로 저장
    train 시에는  전처리된 text-keyword pickle 파일을 불러와서, dataset을 만들어 사용

    preprocessing : 핵심 keyword
    preprocessing_stopword : stopword 제외 모든 keyword

    미리 만들어져 있는 파일이 있는지 check 후 필요한 파일을 생성.
    단락(문장) 파일 먼저 체크, 그 다음 키워드 파일 체크.

    단락(문장)과 키워드의 list 개수가 같아야 한다.
    키워드 생성 후 빈 키워드의 경우가 있으므로 체크한다.
    빈 키워드와 매칭되는 단락(문장)이라면 빈 키워드와 단락(문장)을 삭제한다.
    check_is_empty 함수 
    '''

    def preprocessing(self):
        '''
            one sentence txt only generate
        '''
        flags = ['train', 'test', 'valid']
        out_path = self.out_path
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        output_format = self.output_format
        for flag in flags:
            if not os.path.isfile(os.path.join(out_path, f'{flag}_one_sentence.txt')):
                para_list = self.csv_to_list(flag)
            else:
                with open(f'{out_path}{flag}_one_sentence.txt', 'rb') as f:
                    para_list = pickle.load(f)

            if not os.path.isfile(os.path.join(out_path, f'{flag}_one_sentence_keyword_{output_format}.txt')):
                if output_format == 'default':
                    keyword_list = self.gen_keyword_list_basic(para_list)
                elif output_format == 'stopword':
                    keyword_list = self.gen_keyword_list_stopword(para_list)
                elif output_format == 'edit_stopword':
                    keyword_list = self.gen_keyword_list_edit_stopword(para_list)
                elif output_format == 'noun':
                    keyword_list = self.gen_keyword_list_noun(para_list)
                else:
                    raise ValueError('unexpected output_format value!!!')

                para_list, keyword_list = self.check_is_empty(para_list, keyword_list)
                self.pickle_save(f'{flag}_one_sentence', para_list)
                print(f'making {flag}_one_sentence.txt complete!!!')
                self.pickle_save(f'{flag}_one_sentence_keyword_{output_format}', keyword_list)
                print(f'making keyword file from {flag}_one_sentence.txt complete!!!')
            else:
                print(f"{flag}_one_sentence_keyword_{output_format}.txt file exist !!!")
                exit()

    def check_is_empty(self, para_list, keyword_list):
        para_li = []
        keyword_li = []
        print('checking is_empty...')
        for i in range(len(para_list)):
            if keyword_list[i]:
                para_li.append(para_list[i])
                keyword_li.append(keyword_list[i])
        return para_li, keyword_li

    # text에 <unk>이 있는 경우가 있어서, 미리 제거 해 줌.
    def gen_keyword_list(self, para_list, ratio=1):
        keyword_list = []
        keyword_ratio = ratio
        print('generating default keyword...')
        for para in tqdm(para_list):
            line = re.sub("<unk>\s?", '', para)
            keyword_list.append(keywords(line, ratio=keyword_ratio, split=True))
        return keyword_list

    '''
    stopword(unk 추가) list 준비
    text에 문자나 숫자를 제외한 것들 제거
    text를 단어로 나누고, stopword와 비교하여 아니면 keyword에 추가
    '''

    def gen_keyword_list_basic(self, para_list):
        keyword_list = []
        print('generating default keyword...')
        for para in tqdm(para_list):
            line = " ".join(re.findall("[a-zA-Z0-9]+", para))
            word_tokens = word_tokenize(line)
            keyword_list.append(word_tokens)
        return keyword_list

    def gen_keyword_list_edit_stopword(self, para_list):
        stop_words = set(stopwords.words('english'))
        # add stop word
        stopword = ["unk"]
        for ele in self.except_stopword_list:
            stop_words.remove(ele)
        for i in stopword:
            stop_words.add(i)
        keyword_list = []
        print('generating edit stopword keyword...')
        for para in tqdm(para_list):
            line = " ".join(re.findall("[a-zA-Z0-9]+", para))
            word_tokens = word_tokenize(line)
            result = []
            for w in word_tokens:
                if w not in stop_words:
                    result.append(w)
            keyword_list.append(result)
        return keyword_list

    def gen_keyword_list_stopword(self, para_list):
        stop_words = set(stopwords.words('english'))
        # add stop word
        stopword = ["unk"]
        for i in stopword:
            stop_words.add(i)
        keyword_list = []
        print('generating stopword keyword...')
        for para in tqdm(para_list):
            line = " ".join(re.findall("[a-zA-Z0-9]+", para))
            word_tokens = word_tokenize(line)
            result = []
            for w in word_tokens:
                if w not in stop_words:
                    result.append(w)
            keyword_list.append(result)
        return keyword_list

    def gen_keyword_list_noun(self, para_list):
        noun_list = []
        _noun_list = []
        del_words = ['unk', 'III', 'I', 'II', 'VII', 'VI', 'AFI']
        print('generating noun keyword...')
        for para in tqdm(para_list):
            tagged_list = pos_tag(word_tokenize(para))
            noun_list_ = [t[0] for t in tagged_list if t[1] == 'NN' or t[1] == 'NNP']
            for text in noun_list_:
                if re.match(r"[^0-9]*\d", text):
                    # print(f'delete element! : {text}')
                    pass
                elif re.match(r"[^0-9]*\W", text):
                    # print(f'delete element! : {text}')
                    pass
                elif len(text) <= 2:
                    # print(f'delete element! : {text}')
                    pass
                elif text in del_words:
                    # print(f'delete element! : {text}')
                    pass
                else:
                    if text not in _noun_list:
                        _noun_list.append(text)
            noun_list.append(_noun_list)
            # print(f'after processing : {_noun_list}')
            _noun_list = []
        return noun_list


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    '''
        python make_wiki_dataset.py --output_format edit_stopword --out_path data/wiki103/preprocessing/edit_stopword/
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_shuffle', type=str2bool, default=True, required=False,
                        help="determine inp shuffle")
    parser.add_argument('--wiki_path', type=str, default='data/wiki103/csv/', required=False,
                        help="wiki data path")
    parser.add_argument('--text_max_length', type=int, default=300, required=False,
                        help="determine maximum sentence length")
    parser.add_argument('--text_min_length', type=int, default=50, required=False,
                        help="determine minimum sentence length")
    parser.add_argument('--out_path', type=str, default='data/wiki103/preprocessing/', required=True,
                        help="output path processed data")
    parser.add_argument('--output_format', type=str, default='edit_stopword', required=True,
                        help="(default / edit_stopword / stopword / noun)")
    parser.add_argument('--is_one_sentence', type=str2bool, default=True, required=False,
                        help="determine making one sentence or not")
    args = parser.parse_args()

    wiki103 = Wiki103(args)
    wiki103.preprocessing()


    # para_list = wiki103.csv_to_list(flag='test')
    # # keywords = wiki103.gen_keyword_list(para_list)
    # # keywords_basic = wiki103.gen_keyword_list_basic(para_list)
    # # keywords_stop = wiki103.gen_keyword_list_stopword(para_list)
    # # keywords_noun = wiki103.gen_keyword_list_noun(para_list)
    #
    # keywords_stop_edit = wiki103.gen_keyword_list_edit_stopword(para_list)
    #
    # for i in range(10):
    #     print(para_list[i])
    #     print(keywords_stop_edit[i])
    #     # print(keywords[i])
    #     # print(keywords_basic[i])
    #     # print(keywords_stop[i])
    #     # print(keywords_noun[i])
    #     print('=' * 20)




