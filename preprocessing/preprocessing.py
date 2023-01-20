from os import path as px
from json import load
from string import punctuation, ascii_lowercase
from re import sub, match
from unicodedata import category
from random import choice
from sys import maxunicode
from helper import isWindows, trydel
from subprocess import check_output
from preprocessing.tokenizer import sr_tokenize


def get_preprocess(name):
    if 'srlat-sem' in name:
        return sem_preprocess
    elif 'srlat-synt' in name:
        return synt_preprocess
    else:
        return None


def sem_preprocess(text):
    if lex_dic and stopwords:
        text = sem_text(text, lex_dic)
        if text == "":
            text += " "
        return text
    else:
        return -1


def synt_preprocess(text):
    if synt_dic and par_path:
        text = synt_text(text)
        return text
    else:
        return -1


def load_lex():
    try:
        with open(lexpath, "r", encoding="utf-8") as j:
            lexdic = load(j)
        return lexdic
    except:
        return None


def load_stopwords():
    try:
        with open(swpath, "r", encoding="utf-8") as j:
            swords = load(j)
            return set(swords)
    except:
        return None


def load_syntmap():
    try:
        with open(syntmap, "r", encoding="utf-8") as j:
            syntdic = load(j)
        return syntdic
    except:
        return None


def load_ttmodel():
    if px.exists(tt_modelpath):
        return str(tt_modelpath)
    else:
        return None


def sem_text(tekst, dic, train=False, fill=""):
    save = []
    x = sr_tokenize(tekst)
    for y in x:
        y = y.lower()
        if y not in stopwords and set(y).isdisjoint(punctset):
            try:
                save.append(dic[y][1])
            except:
                save.append(y)
        else:
            save.append(fill)

    if not train or len(save) > 10:
        tekst = " ".join(save)

    if fill != "":
        tekst = tekst.replace("<e> ", "<e>")

    return tekst.rstrip()


def synt_text(tekst):
    tekst = sub(r" +", " ", tekst)
    tokeni = sr_tokenize(tekst, True)
    taggedlines = tokens2tags(tokeni)
    tekst = "$$".join(taggedlines)
    tekst = tekst.replace("$$ $$", "$$ ")
    for key in synt_dic.keys():
        tekst = tekst.replace(key, synt_dic[key])
    return tekst


def tokens2tags(tokens, keepspace=False):
    quotes = ''.join(c for c in (chr(i) for i in range(0x110000)) if category(c) in ('Pf', 'Pi')) + "„“"
    chrs = (chr(i) for i in range(maxunicode + 1))
    punctuation = set(c for c in chrs if category(c).startswith("P"))
    tempname = px.join(px.dirname(__file__), ''.join(choice(ascii_lowercase) for x in range(30)))

    if keepspace:
        tokens, _, exclusion = rem_xml(tokens)
    for i, token in enumerate(tokens):
        if token != "":
            if "?" in token:
                tokens[i] += "\t?"
            elif set(token).issubset(quotes):
                tokens[i] += "\t'"
            elif set(token).issubset(punctuation):
                tokens[i] += "\t."

    with open(tempname, "w", encoding="utf-8") as s:
        s.write("\n".join(tokens))
    tags = tag_treetagger(tempname)
    if keepspace:
        tags = insert_exc(tags, exclusion)

    trydel(tempname)
    return tags


def tag_treetagger(file_path, probability=False, lemmat=False):
    args = []
    tt_path = str(px.dirname(__file__))

    ext = ""
    if isWindows():
        ext = ".exe"

    args.append(tt_path + "/tree-tagger" + ext)
    args.append(par_path)
    args.append(file_path)

    if probability:
        args.append("-threshold")
        args.append("0.0001")
        args.append("-prob")

    if lemmat:
        args.append("-lemma")

    args.append("-quiet")
    args.append("-sgml")
    args.append("-no-unknown")

    r = check_output(args)
    r = r.decode('utf-8')
    return r.split("\r\n")


def rem_xml(lines):
    exclusion = {}
    noslines = list(line.rstrip('\n') for line in lines if line not in ['\n', ''])

    for idx, line in enumerate(noslines):
        if match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$|^ $", line):
            exclusion[idx] = line
    del noslines

    origlines = [line.rstrip('\n') for line in lines if not match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$|^ $", line)]

    newlines = origlines.copy()
    origlines = list(line.rstrip('\n') for line in origlines if line not in ['\n', '', '\0'])

    return newlines, origlines, exclusion


def insert_exc(lines, exclusion):
    finalines = []
    c = 0
    for i in range(0, len(lines) + len(exclusion)):
        if i in exclusion.keys():
            finalines.append(exclusion[i])
        else:
            try:
                finalines.append(lines[c])
                c += 1
            except:
                pass
    return finalines


thispath = px.dirname(__file__)
lexpath = px.join(thispath, "delaf_gpt2.json")
swpath = px.join(thispath, "stopwords.json")
syntmap = px.join(thispath, "mapclass.json")
tt_modelpath = px.join(px.dirname(__file__), "synt-treetagger")
lex_dic = load_lex()
stopwords = load_stopwords()
punctset = set(punctuation)
synt_dic = load_syntmap()
par_path = load_ttmodel()
