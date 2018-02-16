from collections import Counter
import pycld2
import unicodeblock.blocks
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('prefix', default='data/wmt17/de-en/')
args = parser.parse_args()

langs = ('de','en')
lang_fix = '.' + '-'.join(langs)
subsets = 'commoncrawl', 'europarl-v7', 'news-commentary-v12', 'rapid2016'
for x in subsets:
    path_prefix = args.prefix + x + lang_fix
    paths_in = [path_prefix+'.'+lang for lang in langs]
    paths_out = [path_prefix+'.clean.'+lang for lang in langs]
    latin = lambda s: all("LATIN" in b or "PUNCT" in b or "DIGIT" in b or "SPAC" in b for b in map(unicodeblock.blocks.of,s) if b is not None)
    good_src = lambda s: pycld2.detect(s)[2][0][1] in [langs[0],'un'] and latin(s.decode()) and len(s)>1
    good_trg = lambda s: pycld2.detect(s)[2][0][1] in [langs[1],'un'] and latin(s.decode()) and len(s)>1
    
    with open(paths_in[0],'rb') as src, open(paths_in[1],'rb') as trg, open(paths_out[0],'wb') as src_out, open(paths_out[1],'wb') as trg_out:
        for srcline,trgline in zip(src,trg):
            try:
                if good_src(srcline) and good_trg(trgline):
                    src_out.write(srcline)
                    trg_out.write(trgline)
            except:
                try:
                    srcline = srcline.decode("utf-8").encode("latin-1")
                    trgline = trgline.decode("utf-8").encode("latin-1")
                    try:
                        if good_src(srcline) and good_trg(trgline):
                            src_out.write(srcline)
                            trg_out.write(trgline)
                    except:
                        pass
                except:
                    pass
