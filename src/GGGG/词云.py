import jieba.analyse
import imageio
import jieba.posseg as pseg
def jieba_cut():
    #西游记停用词
    # fr = open('15677.txt', 'r')
    # stop_word_list = fr.readlines()
    # new_stop_word_list = []
    # for stop_word in stop_word_list:
    #     stop_word = stop_word.replace('\ufeef', '').strip()
    #     new_stop_word_list.append(stop_word)
    # print(stop_word_list)  #输出停用词
    #输出西游记 词语出现的次数
    fr_xyj=open('15677.txt','r',encoding='utf-8')
    s=fr_xyj.read()
    words=jieba.cut(s,cut_all=False)
    word_dict={}
    word_list=''
    for word in words:
        if (len(word) > 1 ):
            word_list = word_list + ' ' + word
            if (word_dict.get(word)):
                word_dict[word] = word_dict[word] + 1
            else:
                word_dict[word] = 1

    ##print(word_list)
    #print(word_dict) #输出西游记 词语出现的次数

    #按次数进行排序
    sort_words=sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
    print(sort_words[0:101])#输出前0-100的词

    from wordcloud import WordCloud
    color_mask =imageio.imread("1.jpg")
    wc = WordCloud(
            background_color="black",  # 背景颜色
            max_words=500,  # 显示最大词数
            # font_path="D:/软件（学习）/Python/PyCharm/font/simsun.ttc",  # 使用字体
            min_font_size=15,
            max_font_size=50,
            width=400,
            height=860,
            mask=color_mask) # 图幅宽度
    i=str('why')
    wc.generate(word_list)
    wc.to_file(str(i)+".png")
jieba_cut()
