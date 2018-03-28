# OpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

# Requirements

=======
## Some useful tools:

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the Moses BLEU script for evaluation.

```bash
```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```
## WMT'16 Multimodal Translation: Multi30k (de-en)

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
```

The last line of the train and validation files is blank, so the last line of the bash above removes the non-compliant lines.

### 1) Preprocess the data.

Moses tokenization without html escaping (add the -a option after -no-escape for aggressive hypen splitting)

```bash
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
```

Typically, we lowercase this dataset, as the important comparisons are in uncased BLEU:

```bash
for f in data/multi30k/*.tok; do perl lowercase.perl < $f > $f.low; done # if you ran Moses
```

If you would like to use the Moses tokenization for source and target, prepare the data for the model as so:

```bash
python preprocess.py -train_src data/multi30k/train.en.tok.low -train_tgt data/multi30k/train.de.tok.low -valid_src data/multi30k/val.en.tok.low -valid_tgt data/multi30k/val.de.tok.low -save_data data/multi30k.tok.low -lower
```

```bash
```

The extra lower option in the line above will ensure that the vocabulary object converts all words to lowercase before lookup.

If you would like to use GloVe vectors and character embeddings, now's the time:

```bash
python get_embed_for_dict.py data/multi30k.tok.low.src.dict -glove -chargram -d_hid 400
python get_embed_for_dict.py data/multi30k.tok.low.src.dict -glove -d_hid 300
```

### 2) Train the model.

```bash
python train.py -data data/multi30k.tok.low.train.pt -save_model snapshots/multi30k.tok.low.600h.400d.2dp.brnn.2l.fixed_glove_char.model -brnn -pre_word_vecs_enc data/multi30k.tok.low.src.dict.glove.chargram -fix_embed

python train.py -data data/multi30k.tok.low.train.pt -save_model snapshots/multi30k.tok.low.600h.300d.2dp.brnn.2l.fixed_glove.model -brnn -rnn_size 600 -word_vec_size 300 -pre_word_vecs_enc data/multi30k.tok.low.src.dict.glove -fix_embed
```

### 3) Translate sentences.

```bash
python translate.py -gpu 0 -model model_name -src data/multi30k/test.en.tok.low -tgt data/multi30k/test.de.tok.low -replace_unk -verbose -output multi30k.tok.low.test.pred

```

### 4) Evaluate.

```bash
perl multi-bleu.perl data/multi30k/test.de.tok.low < multi30k.tok.low.test.pred
```

## IWSLT'16 (de-en)

### 0) Download the data.

```bash
mkdir -p data/iwslt16
wget https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz && tar -xf de-en.tgz -C data
```

### 1) Preprocess the data.

```bash
python iwslt_xml2txt.py data/de-en
python iwslt_xml2txt.py data/de-en -a

python preprocess.py -train_src data/de-en/train.de-en.en.tok -train_tgt data/de-en/train.de-en.de.tok -valid_src data/de-en/IWSLT16.TED.tst2013.de-en.en.tok -valid_tgt data/de-en/IWSLT16.TED.tst2013.de-en.de.tok -save_data data/iwslt16.tok.low -lower -src_vocab_size 22822 -tgt_vocab_size 32009

#Glove Vectors + CharNgrams
python get_embed_for_dict.py data/iwslt16.tok.low.src.dict -glove -chargrams
python get_embed_for_dict.py data/iwslt16.tok.low.src.dict -glove
```

### 2) Train the model.

```bash
python train.py -data data/iwslt16.tok.low.train.pt  -save_model snapshots/iwslt16.tok.low.600h.400d.2dp.brnn.2l.fixed_glove_char.model -gpus 0 -brnn -rnn_size 600 -fix_embed -pre_word_vecs_enc data/iwslt16.tok.low.src.dict.glove.chargram > iwslt16.clean.tok.low.600h.400d.2l.brnn.2dp.fixed_glove_char.log

python train.py -data data/iwslt16.tok.low.train.pt  -save_model snapshots/iwslt16.tok.low.600h.300d.2dp.brnn.2l.fixed_glove_char.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -fix_embed -pre_word_vecs_enc data/iwslt16.tok.low.src.dict.glove > iwslt16.tok.low.600h.300d.2dp.brnn.2l.fixed_glove.log
```

### 3) Translate sentences.

```bash
python translate.py -gpu 0 -model model_name -src data/de-en/IWSLT16.TED.tst2014.de-en.en.tok -tgt data/de-en/IWSLT16.TED.tst2014.de-en.de.tok -replace_unk -verbose -output iwslt.ted.tst2014.de-en.tok.low.pred
```

### 4) Evaluate.

```bash
perl multi-bleu.perl data/de-en/IWSLT16.TED.tst2014.de-en.de.tok < iwslt.ted.tst2014.de-en.tok.low.pred
```

## WMt'17 (de-en)

### 0) Download the data.

```bash
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
```

### 1) Preprocess the data.

```bash
# News Commentary
python preprocess.py -train_src data/wmt17/de-en/news-commentary-v12.de-en.clean.en.tok.low -train_tgt data/wmt17/de-en/news-commentary-v12.de-en.clean.de.tok.low -valid_src data/wmt17/de-en/newstest2013.en.tok.low -valid_tgt data/wmt17/de-en/newstest2013.de.tok.low -save_data data/news-commentary.clean.tok.low -lower -seq_length 75
python get_embed_for_dict.py data/news-commentary.clean.tok.low.src.dict -glove -d_hid 300
python get_embed_for_dict.py data/news-commentary.clean.tok.low.src.dict -glove -chargrams -d_hid 400

# Rapid Fire
python preprocess.py -train_src data/wmt17/de-en/rapid*.clean.en.tok.low -train_tgt data/wmt17/de-en/rapid*.clean.de.tok.low -valid_src data/wmt17/de-en/newstest2013.en.tok.low -valid_tgt data/wmt17/de-en/newstest2013.de.tok.low -save_data data/rapid.clean.tok.low -lower -seq_length 75
python get_embed_for_dict.py data/rapid.clean.tok.low.src.dict -glove -d_hid 300
python get_embed_for_dict.py data/rapid.clean.tok.low.src.dict -glove -chargrams -d_hid 400

# Europarl
python preprocess.py -train_src data/wmt17/de-en/europarl*.clean.en.tok.low -train_tgt data/wmt17/de-en/europarl*.clean.de.tok.low -valid_src data/wmt17/de-en/newstest2013.en.tok.low -valid_tgt data/wmt17/de-en/newstest2013.de.tok.low -save_data data/europarl.clean.tok.low -lower -seq_length 75
python get_embed_for_dict.py data/europarl.clean.tok.low.src.dict -glove -d_hid 300
python get_embed_for_dict.py data/europarl.clean.tok.low.src.dict -glove -chargrams -d_hid 400

# Common Crawl
python preprocess.py -train_src data/wmt17/de-en/commoncrawl*.clean.en.tok.low -train_tgt data/wmt17/de-en/commoncrawl*.clean.de.tok.low -valid_src data/wmt17/de-en/newstest2013.en.tok.low -valid_tgt data/wmt17/de-en/newstest2013.de.tok.low -save_data data/commoncrawl.clean.tok.low -lower -seq_length 75
python get_embed_for_dict.py data/commoncrawl.clean.tok.low.src.dict -glove -d_hid 300
python get_embed_for_dict.py data/commoncrawl.clean.tok.low.src.dict -glove -chargrams -d_hid 400

# WMT'17
python preprocess.py -train_src data/wmt17/de-en/train.clean.en.tok.low -train_tgt data/wmt17/de-en/train.clean.de.tok.low -valid_src data/wmt17/de-en/newstest2013.en.tok.low -valid_tgt data/wmt17/de-en/newstest2013.de.tok.low -save_data data/wmt17.clean.tok.low -lower -seq_length 75
python get_embed_for_dict.py data/wmt17.clean.tok.low.src.dict -glove -d_hid 300
python get_embed_for_dict.py data/wmt17.clean.tok.low.src.dict -glove -chargrams -d_hid 400
```

### 2) Train the model

```bash
# Train fixed glove+char models
for corpus in wmt17
do
python train.py -data data/${corpus}.clean.tok.low.train.pt  -save_model snapshots/${corpus}.clean.tok.low.600h.400d.2l.brnn.2dp.fixed_glove_char.model -gpus 0 -brnn -word_vec_size 400 -pre_word_vecs_enc data/${corpus}.clean.tok.low.src.dict.glove.chargram -fix_embed > logs/${corpus}.clean.tok.low.600h.400d.2l.brnn.2dp.fixed_glove_char.log
done

# Train fixed glove models
for corpus in wmt17
do
python train.py -data data/${corpus}.clean.tok.low.train.pt  -save_model snapshots/${corpus}.clean.tok.low.600h.300d.2l.brnn.2dp.fixed_glove.model -gpus 0 -brnn -word_vec_size 300 -pre_word_vecs_enc data/${corpus}.clean.tok.low.src.dict.glove -fix_embed > logs/${corpus}.clean.tok.low.600h.300d.2l.brnn.2dp.fixed_glove.log
done

# Train fixed glove+char models
for corpus in news-commentary rapid europarl commoncrawl
do
python train.py -data data/${corpus}.clean.tok.low.train.pt  -save_model snapshots/${corpus}.clean.tok.low.600h.400d.2l.brnn.2dp.fixed_glove_char.model -gpus 0 -brnn -word_vec_size 400 -pre_word_vecs_enc data/${corpus}.clean.tok.low.src.dict.glove.chargram -fix_embed > logs/${corpus}.clean.tok.low.600h.400d.2l.brnn.2dp.fixed_glove_char.log
done

# Train fixed glove models
for corpus in news-commentary rapid europarl commoncrawl
do
python train.py -data data/${corpus}.clean.tok.low.train.pt  -save_model snapshots/${corpus}.clean.tok.low.600h.300d.2l.brnn.2dp.fixed_glove.model -gpus 0 -brnn -word_vec_size 300 -pre_word_vecs_enc data/${corpus}.clean.tok.low.src.dict.glove -fix_embed > logs/${corpus}.clean.tok.low.600h.300d.2l.brnn.2dp.fixed_glove.log
done
```

### 3) Translate sentences.

### 4) Evaluate.
