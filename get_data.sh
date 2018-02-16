wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

mkdir -p data/wmt17
cd data/wmt17
wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz
wget http://data.statmt.org/wmt17/translation-task/rapid2016.tgz
wget http://data.statmt.org/wmt17/translation-task/dev.tgz
tar -xzf training-parallel-europarl-v7.tgz
tar -xzf training-parallel-commoncrawl.tgz
tar -xzf training-parallel-nc-v12.tgz
tar -xzf rapid2016.tgz
tar -xzf dev.tgz
mkdir de-en
mv *de-en* de-en
mv training/*de-en* de-en
mv dev/*deen* de-en
mv dev/*ende* de-en
mv dev/*.de de-en
mv dev/*.en de-en
mv dev/newstest2009*.en*
mv dev/news-test2008*.en*

python ../../wmt_clean.py de-en
for l in de; do for f in de-en/*.clean.$l; do perl ../../tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
for l in en; do for f in de-en/*.clean.$l; do perl ../../tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
for l in en de; do for f in de-en/*.clean.$l.tok; do perl ../../lowercase.perl < $f > $f.low; done; done
for l in en de; do perl ../../tokenizer.perl -no-escape -l $l -q  < de-en/newstest2013.$l > de-en/newstest2013.$l.tok; done
for l in en de; do perl ../../lowercase.perl  < de-en/newstest2013.$l.tok > de-en/newstest2013.$l.tok.low; done
for l in en de; do cat de-en/commoncraw*clean.$l.tok.low de-en/europarl*.clean.$l.tok.low de-en/news-commentary*.clean.$l.tok.low de-en/rapid*.clean.$l.tok.low > de-en/train.clean.$l.tok.low; done
