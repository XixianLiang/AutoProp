import re
import os
from datetime import datetime

def sort_date(file_name):
    # 正则表达式
    pattern = r'(\d{4}-\d{2}-\d{2})_(\d{6})'

    # 匹配并提取日期和时间
    match = re.search(pattern, file_name)
    if match:
        date_str = match.group(1)  # 日期
        time_str = match.group(2)  # 时间
        
        # 将日期和时间字符串合并为一个字符串
        datetime_str = f"{date_str} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        
        # 转换为 datetime 对象
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        
        return datetime_obj

def get_filenames(events_path):
    file_names = os.listdir(events_path)
    file_names = sorted(file_names, key=sort_date)
    return file_names

if __name__ == "__main__":
    import json
    
    basepath = "output0808/events"
    filenames = get_filenames(basepath)
    input_events = []
    for i, filename in enumerate(filenames):
        with open(os.path.join(basepath, filename), "r") as fp:
            try:
                input_events.append(json.load(fp))
            except:
                print(f"failed when decoding file:{filename}")
    
    input_events 
