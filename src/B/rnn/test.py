def read_data(data_path):
    datas = []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            words = line.strip().split()
            datas.append(words)
    return datas


a = source_data = read_data("./data/ch_source_data_seg.txt")[:10]
print(a)
