import os
import glob
import subprocess
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

parser = ArgumentParser(description='sgm2txt')
parser.add_argument('path')
parser.add_argument('-t', '--tags', nargs='+', default=['seg']) 
parser.add_argument('-th', '--threads', default=8, type=int)
parser.add_argument('-a', '--aggressive', action='store_true')
args = parser.parse_args()

def tokenize(f_txt):
    lang = os.path.splitext(f_txt)[1][1:]
    f_tok = f_txt
    if args.aggressive:
        f_tok += '.atok'
    else:
        f_tok += '.tok'
    with open(f_tok, 'w') as fout, open(f_txt) as fin:
        if args.aggressive:
            pipe = subprocess.call(['perl', 'tokenizer.perl', '-a', '-q', '-threads', str(args.threads), '-no-escape', '-l', lang], stdin=fin, stdout=fout)
        else:
            pipe = subprocess.call(['perl', 'tokenizer.perl', '-q', '-threads', str(args.threads), '-no-escape', '-l', lang], stdin=fin, stdout=fout)

for f_xml in glob.iglob(os.path.join(args.path, '*.sgm')):
    print(f_xml)
    f_txt = os.path.splitext(f_xml)[0] 
    with open(f_txt, 'w') as fd_txt:
        root = ET.parse(f_xml).getroot()[0]
        for doc in root.findall('doc'):
            for tag in args.tags:
                for e in doc.findall(tag):
                    fd_txt.write(e.text.strip() + '\n')
    tokenize(f_txt)
