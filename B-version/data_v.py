from konlpy.tag import Okt
import pandas as pd
import tensorflow as tf
import enum
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
from configs import DEFINES

from tqdm import tqdm

PAD_MASK = 0
NON_PAD_MASK = 1

FILTERS = "([~.,!?\"':;)(])"

PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)


def load_data():
    # 판다스를 통해 데이터를 불러옴
    data_df = pd.read.csv(DEFINES.data_path, header=0)
    # 질문과 답변 열을 가져와 question과 answer에 넣음
    question, answer = list(data_df['Q'], list(data_df['A'])
    # 사이킷런에서 지원하는 함수를 통해 학습 셋과 테스트 셋을 나눔
    train_input, eval_input, train_label, eval_label = train_test_split(question, answer,
    test_size = 0.33, random_state = 42)
    # 그 값을 리턴
    return train_input, train_label, eval_input, eval_label

def prepro_like_morphlized(data):
    # 형태소 분석 모듈 객체를 생성
    morph_analyzer = Okt()
    # 형태소 토크나이즈 결과 문장을 받을 리스트를 생성
    result_data = list()
    # 데이터에 있는 각 문장에 대해 토크나이징할 수 있게 반복문을 선언
    for seq in tqdm(data):
        # Twitter.morphs 함수를 통해 토크나이징된
        # 리스트 객체를 받고 다시 공백 문자를 기준으로
        # 문자열로 재구성
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data

# 인덱스화할 value, 그리고 키가 워드이고
# 값이 인덱스인 딕셔너리를 받음

def enc_processing(value, dictionary):
    # 인덱스 값을 가지고 있는 배열(누적됨)
    sequences_input_index = []
    # 하나의 인코딩되는 문장의 길이를 가지고 있음(누적됨)
    sequences_length = []
    # 형태소 토크나이징 사용 여부
    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphilized(value)

    # 한 줄씩 불러옴
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 통해 필터에 들어 있는 값을 ""으로 치환
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 하니의 문장을 인코딩할 때 가지고 있기 위하 배열
        sequence_index = []
        # 문장을 공백 문자 단위로 자름
        for word in sequence.split():
            # 잘려진 단어들이 딕셔너리에 존재하는지 보고
            # 그 값을 가져와 sequence_index에 추가
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            # 잘려진 단어가 딕셔너리에 존재하지 않는
            # 경우이므로 UNK(2)를 넣음
            else:
                sequence_index.extend([dictionary[UNK]])
        # 문장 제한 길이보다 길어질 경우 뒤에 있는 토큰을 자름
        if len(sequence_index) > DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length]
        # 하나의 문장에 길이를 넣음
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣음
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) *
        [dictionary[PAD]]
        # 뒤로 넣음
        sequence_index.reverse()
        # 인덱스화돼 있는 값을
        # sequece_input_index에 넣음
        sequences_input_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경
    # (텐서플로 dataset에 넣어 주기 위한 사전 작업)
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 전달
    return np.asarray(sequences_input_index), sequences_length

# 인덱스화할 value와 키가 워드이고
# 값이 인덱스인 딕셔너리를 받음
def dec_target_processing(value, dictionary):
    # 인덱스 값을 가지고 있는 배열(누적됨)
    sequences_target_index = []
    sequences_length = []
    # 형태소 토크나이징 사용 여부
    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # 한 줄씩 불러옴
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 통해 필터에 들어 있는 값을 ""을 치환
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 문장에서 공백 문자 단위별로 단어를 가져와
        # 딕셔너리의 값인 인덱스를 넣음
        # 디코딩 출력의 마지막에 END를 넣음
        sequence_index = [dictionary[word] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤에 있는 토큰을 자름
        # 그리고 END 토큰을 넣음
        if len(sequence_index) > DEFINES.max_sequence_length :
            sequence_index = sequence_index[:DEFINES.max_sequence_length-1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]
        # 학습 시 PAD 마스크를 위한 벡터를 구성
        sequences_length.append([PAD_MASK if num > len(sequence_index) else NON_PAD_MASK
        for num in range(DEFINES.max_sequence_length)])
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣음
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index))*
        [dictionary[PAD]]
        # 인덱스화돼 있는 값을 sequences_target_index에 넣음
        sequences_target_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경
    # (텐서플로 데이터셋에 넣기 위한 사전 작업)
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 전달
    return np.asarray(sequences_target_index), np.asarray(sequence_index)


# 인덱스를 문자열로 변경하는 함수
# 바꾸고자 하는 인덱스 value와 인덱스를
# 키로 가지고 있고 값으로 단어를 가지고 있는
# 딕셔너리를 받음
def pred2string(value, dictionary):
    # 텍스트 문장을 보관할 배열을 선언
    sentence_string = []
    # 인덱스 배열 하나를 꺼내서 v에 전달
    if DEFINES.serving == True:
        for v in value['output']:
            sentence_string = [dictionary[index] for index in v]
    else:
        for v in value:
            # 딕셔너리에 있는 단어로 변경해서 배열에 담음
            sentence_string = [dictionary[index] for index in v['indexs']]

    print(sentence_string)
    answer = ""
    # 패딩값도 담겨 있으므로 패딩은 모두 공백으로 처리
    for word in sentence_string:
        if word not in PAD and word not in END:
            answer += word
            answer += " "
    # 결과물 출력
    print(answer)
    return(answer)

def rearrange(input, target):
    features = {"input": input}
    return features, target

def train_rearrange(input, length, target):
    features = {"input": input, "length": length}
    return features, target

# 학습에 들어가 배치 데이터를 만드는 함수
def train_input_fn(train_input_enc, train_target_dec, train_target_dec_length,
batch_size):
    # 테이터셋을 생성하는 부분으로서 from_tensor_slices 부분은
    # 각각 한 문장으로 자른다고 보면 됨
    # train_input_enc, train_target_dec_length, train_target_dec_length
    # 3개를 각각 한 문장으로 나눔
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc,
    train_target_dec_length, train_target_dec))
    # 전체 데이터를 섞음
    dataset = dataset.shuffle(buffer_size = len(train_input_enc))
    # 배치 인자 값이 없다면 에러를 발생시킴
    assert batch_size is not None, "train batchSize must not be None"
    # from_tensor_slices를 통해 나눈 것을
    # 배치 크기만큼 묶음
    dataset = dataset.batch(batch_size)
    # 데이터의 각 요소에 대해 train_rearrange 함수를
    # 통해 요소를 변환해 맵으로 구성
    dataset = dataset.map(train_rearrange)
    # repeat() 함수에 원하는 에폭 수를 넣을 수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터됨
    dataset = dataset.repeat()
    # make_one_shot_iterator를 통해 이터레이터를 만듦
    iterator = dataset.make_one_shot_iterator()
    # 이터레이터를 통해 다음 항목의 텐서 개체를 전달
    return iterator.get_next()

# 평가에 들어가 배치 데이터를 만드는 함수
def eval_input_fn(eval_input_enc, eval_target_dec, batch_size):
    # 데이터셋을 생성하는 부분으로서 from_tensor_slices 부분은
    # 각각 한 문장으로 자른다고 보면 됨
    # eval_input_enc, eval_target_dec, batch_size
    # 3개를 각각 한 문장으로 나눔
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_target_dec))
    # 전체 데이터를 섞음
    dataset = dataset.shuffle(buffer_size = len(eval_input_enc))
    # 배치 인자 값이 없다면 에러를 발생시킴
    assert batch_size is not None, "eval batchSize must not be None"
    # from_tensor_slices를 통해 나눈 것을
    # 배치 크기만큼 묶음
    dataset = dataset.batch(batch_size)
    # 데이터의 각 요소에 대해 rearrange 함수를
    # 통해 요소를 변환해 맵으로 구성
    dataset = dataset.map(rearrange)
    # repeat() 함수에 원하는 에폭 수를 넣을 수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터됨
    # 평가이므로 1회만 동작시킴
    dataset = dataset.repeat(1)
    # make_one_shot_iterator를 통해
    # 이터레이터를 만듦
    iterator = dataset.make_one_shot_iterator()
    # 이터레이터를 통해 다음 항목의 텐서 개체를 전달
    return iterator.get_next()

def data_tokenzier(data):
    # 토크나이징해서 담을 배열 생성
    words = []
    for sentence in data:
        # FILTERS = "([~.,!?\"':;)(])"
        # 위 필터와 같은 값들을 정규화 표현식을
        # 통래 모두 ""으로 변환
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    # 토크나이징과 정규표현식을 통해 만들어진 값들을 전달
    return [word for word in words if word]

def load_vocabulary():
    # 사전을 담을 배열을 준비
    vocabulary_list = []
    # 사전을 구성한 후 파일로 저장
    # 파일의 존재 여부를 확인
    if (not (os.path.exists(DEFINES.vocabulary_path))):
        # 이미 생성된 사전 파일이 존재하지 않으므로
        # 데이터를 가지고 만들어야 함
        # 그래서 데이터가 존재하면 사전을 만들기 위해
        # 데이터 파일릐 존재 여부를 확인
        if (os.path.exists(DEFINES.data_path)):
            # 데이터가 존재하니 판다스를 통해 데이터를 불러옴
            data_df = pd.read_csv(DEFINES.data_path, encoding = 'utf-8')
            # 판다스의 데이터 프레임을 통해
            # 질문과 답에 대한 열을 가져옴
            question, answer = list(data_df['Q']), list(data_df['A'])
            if DEFINES.tokenize_as_morph: # 형태소에 따른 토크나이저 처리
                quesiton = prepro_like_morphlized(question)
                answer = prepro_likeA_morphlized(answer)
            data = []
            # 질문과 답변을 extend를 통해 구조가 없는 배열로 만듦
            data.extend(question)
            data.extend(answer)
            # 토크나이저 처리하는 부분
            words = data_tokenizer(data)
            # 공통적인 단어에 대해서는 모두
            # 필요 없으므로 한 개로 만들기 위해
            # set으로 만들고 이를 리스트로 만듦
            words = list(set(words))
            # 데이터에 없는 내용 중에 MARKER를 사전에
            # 추가하기 위해 다음과 같이 처리
            # 다음은 MARKER 값이며 리스트의 첫 번째부터
            # 순서대로 넣기 위해 인덱스 0에 추가
            # PAD = "<PADDING>"
            # STD = "<START>"
            # END = "<END>"
            # UNK = "<UNKNOWN>"
            words[:0] = MARKER
        # 사전을 리스트로 만들었으니 이 내용을 사전 파일을 만들어 넣음

        with open(DEFINES.vocabulary_path, 'w', encoding = 'utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
    # 사전 파일이 존재하면 여기서 그 파일을 불러서 배열에 넣음
    with open(DEFINES.vocavulary_path, 'r', encoding = 'utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    # 배열의 내용을 키와 값이 있는 딕셔너리 구조로 만듦
    char2idx, idx2word = make_vocabulary(vocabulary_list)
    # 두 가지 형태의 키와 값이 있는 형태를 리턴
    # (예) 단어: 인덱스, 인덱스: 단어
    return char2idx, idx2word, len(char2idx)


def make_vocabulary(vocabulary_list):
    # 리스트를 키가 단어이고 값이 인덱스인 딕셔너리를 만듦
    char2idx = {char : idx for idx, char in enumerate(vocabulary_list)}
    # 리스트를 키가 인덱스이고 값이 단어인 딕셔너리를 만듦
    idx2word = {idx : char for idx, char in enumerate(vocabulary_list)}
    # 두 개의 딕셔너리를 전달
    return char2idx, idx2word


def main(self):
    char2idx, idx2word, vocabulary_length = load_vocabulary

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
