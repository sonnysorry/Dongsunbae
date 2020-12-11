import pandas as pd
import re
from datetime import datetime, timedelta
import datetime
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

def reply_message_time(file):
    time_data =list(file['Date'])

    re_time_data = []
    for i in time_data:
        a = datetime.datetime.strptime(i, '%Y-%m-%d %H:%M')
        re_time_data.append(a)

    time_gap =[]
    for i in range(0, len(re_time_data)-1):
        delta = re_time_data[i+1] - re_time_data[i]
        gap = delta.seconds
        if gap > 60 and gap <= 1200: # 바로 연이어 메시지를 보내거나 대화가 끝나서 메시지를 보내지 않는 경우를
                                     # 제외시키기 위해 시간 차이는 60초이상이고 1200초이내
            time_gap.append(gap)

    sum = 0
    for i in time_gap:
        sum += i
    aver = sum/len(time_gap)
    aver = round(aver)

    return str(aver)
