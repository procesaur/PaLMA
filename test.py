from transformerworks import inspect, perplexity, ini, initiated_models
from json import load, dump
from random import choice
from tqdm import tqdm


data_dir = "D:/aplikacije_mihailo/gpt2-miks/data/"
mods = ["procesaur/gpt2-srlat-synt", "procesaur/gpt2-srlat", "procesaur/gpt2-srlat-sem"]
files = [data_dir + "synt-testing.json2", data_dir + "testing.json2", data_dir + "sem-testing.json2"]
bads = ["bad-ord", "bad-form", "bad-sem"]
bads_list = [choice(bads) for x in range(27000)]

p = {}
pv = {}

if True:
    for mod, file in zip(mods, files):

        p[mod] = {"sr": [], "google": [], "bad": []}

        with open(file, "r", encoding="utf-8") as jf:
            sentences = load(jf)

        for i, x in enumerate(tqdm(sentences, total=len(sentences))):
            p[mod]["sr"].append(perplexity(mod, x["sr"]))
            p[mod]["google"].append(perplexity(mod, x["google_sr"]))
            p[mod]["bad"].append(perplexity(mod, x["bad"]))

        with open("perplexities.json", "w", encoding="utf-8") as rf:
            dump(p, rf, ensure_ascii=False)


if True:
    for mod, file in zip(mods, files):
        pv[mod] = {"sr": [], "google": [], "bad": []}

        with open(file, "r", encoding="utf-8") as jf:
            sentences = load(jf)

        for i, x in enumerate(tqdm(sentences, total=len(sentences))):
            pv[mod]["sr"].append(inspect(mod, x["sr"], 5)[0])
            pv[mod]["google"].append(inspect(mod, x["google_sr"], 5)[0])
            pv[mod]["bad"].append(inspect(mod, x["bad"], 5)[0])

        with open("pvs.json", "w", encoding="utf-8") as rf:
            dump(pv, rf, ensure_ascii=False)
