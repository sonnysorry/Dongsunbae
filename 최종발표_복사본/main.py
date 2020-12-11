#-*-coding:utf-8-*-
# 카카오톡에서 사용자가 딕셔너리에 나타나는 사람에게 존댓말을 쓰는지 여부를 파악합니다.
# 존대말을 쓰면 1 존대말을 안쓰면 0 입니다.
# 대화 파일명과 사용자의 이름을 입력하면 대화 상대에게 존댓말을 쓰는 지 여부를 파악합니다.
import pandas as pd
import re
import datetime as dt
from check import mecab_ch
import avgtime

#존대말 사용 여부를 1과 0으로 나타냅니다.
def drop_dupli():
    data = pd.read_csv(file, encoding = 'utf-8')
    kakaotalk_la = pd.DataFrame(data, columns=["User", "Date", "Message"])
    kakaotalk_la.drop_duplicates()
    kakaotalk_la2 = kakaotalk_la.reset_index(drop=True)
    return kakaotalk_la2

def read_kko_msg(filename):
    with open(filename, encoding = 'utf-8') as f:
        msg_list = f.readlines()
    return msg_list

def apply_kko_regex(msg_list):
    kko_pattern = re.compile("\[([\S\s]+)\] \[(오전|오후) ([0-9:\s]+)\] ([^\n]+)")
    kko_date_pattern = re.compile("--------------- ([0-9]+년 [0-9]+월 [0-9]+일) ")

    kko_parse_result = list()
    cur_date = ""

    for msg in msg_list:
        if len(kko_date_pattern.findall(msg)) > 0:
        # 패턴에 해당하는 것이 있을 경우, 아래의 코드를 실행한다.
            cur_date = dt.datetime.strptime(kko_date_pattern.findall(msg)[0], "%Y년 %m월 %d일")
            # finall() 정규식으로 찾으면, 결과 문자열의 리스트를 리턴
            cur_date = cur_date.strftime("%Y-%m-%d")
            # cur_date에 날짜를 넣는다.
        else:
            kko_pattern_result = kko_pattern.findall(msg)
            # kko_pattern_result에 findall()을 통해 정규식으로 문자열의 리스트를 리턴
            if len(kko_pattern_result) > 0:
            # 패턴에 해당하는 것이 있을 경우 아래의 코드를 실행한다.
                tokens = list(kko_pattern_result[0])
                # tokens에 패턴에 해당하는 것을 전부 저장한다.
                pattern = re.compile("[0-9]+")
                cur_hour = pattern.findall(tokens[2])[0]
                # 시간 반환
                cur_minute = pattern.findall(tokens[2])[1]
                # 분 반환
                if tokens[1] == '오전' and cur_hour == '12':
                    cur_hour = '0'
                    tokens[2] = ("%s:%s"%(cur_hour, cur_minute))
                    del tokens[1]
                elif tokens[1] == '오전':
                    del tokens[1]
                elif (tokens[1] == '오후' and cur_hour == '12'):
                    cur_hour = '12'
                    tokens[2] = ("%s:%s"%(cur_hour, cur_minute))
                    del tokens[1]
                elif tokens[1] == '오후':
                    tokens[2] =  ("%s:%s"%(str(int(cur_hour)+12), cur_minute))
                    del tokens[1]
                tokens.insert(1, cur_date)
                # cur_date를 인덱스 1에 저장
                tokens[1] = tokens[1] + " " + tokens[2]
                del tokens[2]
                kko_parse_result.append(tokens)

    kko_parse_result = pd.DataFrame(kko_parse_result, columns = ["User", "Date", "Message"])
    kko_parse_result.to_csv("/Users/brotoo/Desktop/study/3-2/oss/merge.csv", encoding='utf-8-sig', index = False)

    return kko_parse_result
def name():
    data1 = pd.read_csv(file, encoding = 'utf-8')
    name_list = list(data1["User"])
    count = {}
    for i in name_list:
        try:
            count[i] += 1
        except:
            count[i] = 1
    return max(count)


def output_csv():
    #데이터를 판다스 데이터프레임으로 불러오고 메시지만 추출합니다.
    data = pd.read_csv(file, encoding='utf-8')
    kakaotalk_label = pd.DataFrame(data, columns=["User", "Date", "Message"])
    text_sentences = list(data['Message'])
    #리스트를 초기화하여 추출한 메시지 데이터에서 각각 존대말 사용 여부를 넣습니다.
    c_list = []
    for line in text_sentences:
        # 추출한 메시지를 정규화하여 한글에서 실질적 의미를 가진 높임 종결어미를 찾습니다.
        parse = re.sub('[\'\"\-!-= .1234567890^#~/?:ㅋ$ㅜ}ㅠ)(*$%@]', '', str(line))
        #띄어쓰기를 제거합니다.
        eumjeol = [s.replace(' ', '') for s in parse]
        if not eumjeol:
            c_list.append(0)
        elif len(eumjeol) == 1:
            if eumjeol[0] == ('넵' or '네' or '넴' or '넨' or '옙' or '예' or '넷' or '옛'):
                c_list.append(1)
            else:
                c_list.append(0)
        elif eumjeol[-9:] == ['삭', '제', '된', '메', '시', '지', '입', '니', '다']:
            c_list.append(0)
        elif eumjeol[-3:] == ['아', '니', '다']:
            c_list.append(0)
        elif eumjeol[-2:] == (['니', '다'] or ['니', '까']) or eumjeol[-1] == ('요' or '용' or '욥' or "염"):
            c_list.append(1)
        else:
            c_list.append(0)
    #위에서 만든 존대말 사용 여부 리스트를(0,1만 들어가있음) 데이터프레임화 시킵니다.
    df2 = pd.DataFrame({'label': c_list})
    #데이터프레임을 합쳐줍니다.
    kakaotalk_label = kakaotalk_label.join(df2)
    return kakaotalk_label
#카카오톡 csv파일 데이터에서는 누구에게 보냈는 지 알 수 없으므로,
#받는 사람의 이름을 레이블링 해주는 코드입니다.
def labeling_data():
    #outputcsv에서 만든 데이터프레임을 엽니다.
    kakaotalk_label1 = output_csv()
    #데이터프레임에서 유저 데이터만 씁니다.
    kakaotalk_label2 = kakaotalk_label1.loc[:, ['User']]
    #사용자의 이름을 입력합니다.
    name = global_name
    #체크포인트 리스트는 카카오톡 데이터를 한줄 씩 읽으며 새로 나오는 이름을 넣는 리스트입니다.
    ckp_list = []
    #인덱스 데이터베이스는 인덱스를 표현하기 위해 데이터 프레임 각 줄의 인덱스를 넣습니다.
    idx_db = []
    #시간 별로 각각 받는 사람의 이름을 예측하 to list에 넣습니다.
    to_list = []
    for row_index, row in kakaotalk_label2.iterrows():
        try:
            #한줄 씩 보면 사용자의 이름이 아닐 때
            if not row[0] == name:
                #체크포인트 리스트에 들어있는 이름과 다른 새 이름이 나오면
                #체크포인트 리스트에 그 줄에 받은 사람의 이름을 넣고
                #인덱스에 추가해줍니다.
                #그리고 to list에 받는 사람 이름을 집어 넣습니다.
                if not row[0] == ckp_list[-1]:
                    ckp_list.append(row[0])
                    idx_db.append(row_index)
                    to_list.append(row[0])
                #체크포인트 리스트에 있는 사람과 같은 사람의 이름이 나오면
                #그 이름을 받는 사람인 to list에 넣습니다. (아래는 동일)
                else:
                    idx_db.append(row_index)
                    to_list.append(ckp_list[-1])
            else:
                idx_db.append(row_index)
                to_list.append(ckp_list[-1])
        except:
            if not row[0] == name:
                ckp_list.append(row[0])
                idx_db.append(row_index)
                to_list.append(ckp_list[-1])
            else:
                idx_db.append(row_index)
                to_list.append(name)
    #만들어진 to 리스트를 데이터프레임화 시키고 join으로 전 데이터프레임과 합칩니다.
    df2 = pd.DataFrame({'to': to_list})
    kakaotalk_label3 = kakaotalk_label1.join(df2)
    return kakaotalk_label3

#존대말을 쓰는 경우를 분석합니다.
def john():
    sentence = ("안녕하세요. 지금은 답장이 불가능합니다. 평균 " + str(t) + " 분 내 답장하는데 그 전에 필요하시면 전화 혹은 문자 부탁드립니다. - 이 메시지는 카카오톡 자동응답기가 자동으로 보낸 메시지입니다.")
    f = open("text1.txt", "w", encoding = "utf-8")
    f.write(sentence)
    f.close

def eorin_nom():
    kakaotalk_la3 = labeling_data()
    kakaotalk_la4 = kakaotalk_la3.loc[:, ['User','Message', 'label']]
    name = global_name
    is_user = kakaotalk_la4['User'] == name
    name_data = kakaotalk_la4[is_user]
    name_data1 = name_data.loc[:, ['Message', 'label']]
    is_up = name_data1['label'] == 0
    k_data = name_data1[is_up]
    text_sentences = list(k_data['Message'])
    keyword = []
    for line in text_sentences:
        word = mecab_ch(line)
        for idx, word in enumerate(word):
            if word.find('EF' or 'EC' or 'ETN') > 0 :
                keyword.append(word.split('/'))
    c = ""
    for i in range(len(keyword)):
        ans = str(keyword[i][0])
        if ans[-1] == ["엉"] or ["강"] or ["당"] or ["용"] or ["랭"] or ["강"]:
            c = "줭"
        else:
            c = "해"
    cnt = [0, 0]
    for j in range(len(keyword)):
        answ = str(keyword[j][1])
        if answ == 'ETN':
            cnt[0] += 1
        else:
            cnt[1] += 1
    if int(cnt[0]/cnt[1]) > 0.4:
        c = "ㄱㄱ"
        b = "음"
    else:
        b = "아"
    keyword_unform = []
    for line in text_sentences:
        word = mecab_ch(line)
        for idx, word in enumerate(word):
            if word.find('UNKNOWN') > 0 :
                keyword_unform.append(word.split('/'))
                break
    keyword_unform2 = []
    for i in range(len(keyword_unform)):
        keyword_unform2.append(keyword_unform[i][0])
    chk = [0, 0, 0]
    chk2 = ["ㅋㅋ", "ㅎㅎ", "ㅠㅠ"]
    for i in keyword_unform2:
        if i == "ㅋㅋ":
            chk[0] += 1
        elif i == "ㅎㅎ" or "ㅎ" or "ㅎㅎㅎ":
            chk[1] += 1
        elif i == "ㅠㅠ" or "ㅠ" or "ㅠㅠㅠ":
            chk[2] += 1
    for i in range(len(chk)):
        if max(chk) == chk[i]:
            d = chk2[i]
    sentence = (" 지금 카톡 못 받" + b + " 평균 답장 시간은 " + str(t) + "분 인 데 필요하면 전화나 문자 " + c + d + " - 이 메시지는 카카오톡 자동응답기가 자동으로 보낸 메시지입니다.")
    f = open("text2.txt", "w", encoding="utf-8")
    f.write(sentence)
    f.close
def johnbig():
    #labeling_data에서 만든 데이터프레임을 불러옵니다.
    kakaotalk_label4 = labeling_data()
    # 받는 사람 별 메시지 수를 세기 위해 카운트 열을 데이터프레임에 추가합니다.
    kakaotalk_label4['count'] = 1
    # 데이터프레임에서 아래 열만 가져옵니다.
    kakaotalk_label3_sample = kakaotalk_label4.loc[:, ['User', 'label', 'to', 'count']]
    # input한 사용의 이름과 동일한 user, 즉 사용자가 받는 사람에게 보낸 메시지만 추출합니다.
    name = global_name
    is_user = kakaotalk_label3_sample['User'] == name
    name_data = kakaotalk_label3_sample[is_user]
    # 레이블(존댓말 사용 여부), to (받는 사람), count(메시지 개수 세기 위한 1)을 추출합니다.
    name_data1 = name_data.loc[:, ['label', 'to', 'count']]
    # 받는 사람 이름 기준으로 정렬합니다.
    name_data1 = name_data1.sort_values(by='to', ascending=False)
    # 추출했기 때문에 인덱스가 섞여있으므로, 인덱스 사용을 위해 인덱스를 현재 값으로 초기화합니다.
    name_data2 = name_data1.reset_index(drop=True)
    # 받는 사람이름이 동일한 행을 모두 더합니다. (존대말 사용여부(0, 1)와 메시지 개수)
    # 데이터 프레임의 형태로 출력됩니다.
    name_data3 = name_data2.groupby(["to"]).sum().reset_index()
    # 받는 사람 이름과 확률 값을 을 넣기 위해 리스트를 만들어줍니다.
    nam = []
    avg_n = []
    # 각각의 데이터를 한 줄 씩 읽어줍니다. (한줄 당 받는 사람 이름 / 존대말 사용 여부 더한 값 / 전체 메시지 개수)
    for row_index, row in name_data3.iterrows():
        # 받는 사람 이름을 nam 리스트에 넣습니다.
        nam.append(row[0])
        # 존댓말 사용 메시지 수를 전체 메시지 수로 나눕니다. (확률 구하기)
        avg = int(row[1]) / int(row[2])
        # 평균 존대말 사용 확률을 리스트에 넣습니다.
        avg_n.append(avg)
    #확률 값을 넣기 위한 리스트를 만듭니다.
    final_idx = []
    # 확률이 0.4 이상이면 존대말을 쓰게 레이블링하고, 리스트에 넣습니다.
    for i in avg_n:
        if i > 0.4:
            final_idx.append(1)
        else:
            final_idx.append(0)
    # 받는 사람 이름과 확률 값을 통한 레이블링 값을 딕셔너리에 넣습니다.
    f = open('db.txt' , mode = 'at', encoding = 'utf-8-sig')
    for i in range(len(final_idx)):
        f.write(nam[i] + "\n");
        f.write(str(final_idx[i]) + "\n");
    f.close

if __name__ == "__main__":
    msg_list = read_kko_msg('merge.txt')
    apply_kko_regex(msg_list)
    file = 'merge.csv'
    global_name = name()
    data = pd.read_csv(file, encoding='utf-8')
    t = round(int(avgtime.reply_message_time(data)) / 60, 0)
    johnbig()
    john()
    eorin_nom()
    print("완료")
