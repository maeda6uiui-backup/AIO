import os
import MeCab

if __name__=="__main__":
    os.environ["MECABRC"]="/usr/local/etc/mecabrc"

    mecab=MeCab.Tagger("-d /usr/local/lib/mecab/dic/jumandic")
    result=MeCab.Tagger().parse("格闘家ボブ・サップの出身国はどこでしょう？")
    print(result)
