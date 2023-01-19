from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import randint as torch_rand
from random import randint
from helper import limit, perp2prob
from math import exp
from preprocessing.preprocessing import get_preprocess
from preprocessing.tokenizer import sr_tokenize
from torchworks import netpass, tensor2device
from json import load


with open("modelname-cache.json", "r", encoding="utf-8") as jf:
    modellist = load(jf)
initiated_models = {}
lock = False


def gengen(text, model, length, temp, alt, cn):
    if alt:
        return gentext_plus(model, alt, text, length, temp, cn)
    else:
        return gentext(model, text, length, temp, cn)


def visualize(text, model, step=5, change=True):
    perp = round(perplexity(model, text), 3)
    vals, tokens = inspect(model, text, step, change)
    return perp, model, vals, tokens


def full_eval(text):
    mods = ["procesaur/gpt2-srlat", "procesaur/gpt2-srlat-sem", "procesaur/gpt2-srlat-synt"]
    perps = {}
    vectors = {}
    tokens = []
    for mod in mods:
        perps[mod] = perplexity(mod, text)
        vectors[mod], tokens = inspect(mod, text, step=5)

    ps = [1/perps[x] for x in perps]
    vs = [[1/y for y in vectors[x]] for x in vectors]

    report = {"general": netpass([vs], "general_cnn.pt", True),
              # "machine": netpass([vs], "google_sr_cnn.pt", True),
              "Semantics": netpass([ps], "bad-sem.pt", False),
              "Syntax (forms)": netpass([ps], "bad-form.pt", False),
              "Syntax (word order)": netpass([ps], "bad-ord.pt", False)
              }

    return report, vectors, tokens, perps


def gentext(modelname, inp="", length=100, temp=0.75, samples=1):
    model, tokenizer, prep = ini(modelname)
    if prep is not None:
        inp = prep(inp)
        if inp == -1:
            return error

    outs = []
    if inp == "":
        tokens = torch_rand(low=260, high=52000, size=(1,))
        inp = tokenizer.decode(tokens, skip_special_tokens=True)

    context = tokenizer(inp, return_tensors="pt")
    cl = context.data["input_ids"].size()[1]

    for x in range(samples):
        output = generate(model, context=context, length=length+cl, temperature=temp)

        decoded_output = []
        for sample in output:
            sample = sample[cl:]
            decoded_output.append(tokenizer.decode(sample, skip_special_tokens=True))

        outs.append("".join(decoded_output))
    label = f"Pieces were generated using {modelname}."
    return outs, None, None, label


def gentext_plus(model, alt, inp="", length=1024,  temp=0.2, samples=1):
    outs, _, __, ___ = gentext(model, inp, length, temp, samples)
    vals = []
    for out in outs:
        vals.append(perplexity(alt, (inp + out).replace("<e>", "").replace("$$", "")))
    best_idx = vals.index(min(vals))
    best = outs[best_idx]
    label = f"Candidates were generated using {model} model, and were evaluated using {alt} model"
    vals = [round(x) for x in vals]
    return outs, vals, best, label


def generate(model, context, length, temperature):
    length = limit(length, 1, 1024)
    encoded_input = tensor2device(context)
    output = model.generate(
        **encoded_input,
        bos_token_id=randint(1, 50000),
        do_sample=True,
        top_k=0,
        max_length=length,
        temperature=temperature,
        no_repeat_ngram_size=3,
        # top_p=0.95,
        num_return_sequences=1,
        pad_token_id=0
        )

    return output


def perplexity(model, text):
    model, tokenizer, prep = ini(model)
    if prep is not None:
        text = prep(text)

    tokens = text2tokentensors(tokenizer, text)
    outputs = model(tokens, labels=tokens)
    loss = outputs[0]
    perp = exp(loss)
    return perp


def inspect(model, text, step, change=True):
    tokens = sr_tokenize(text)
    tokens = [x for x in tokens if x != ""]
    tl = len(tokens)

    if tl < step + 2:
        ini = perplexity(model, "".join(tokens))
        vals = [ini for x in tokens]
        return vals, tokens

    togo = tokens[0:step]
    resto = tokens[step:tl]
    ini = perplexity(model, "".join(togo))
    vals = [ini for x in togo]

    for i, r in enumerate(resto):

        vals.append(0)
        togo.pop(0)
        togo.append(r)
        ini = perplexity(model, "".join(togo))
        n = [ini for x in togo]

        for x in range(step):
            vals[x+i+1] += n[x]

    for i, v in enumerate(vals):
        if i < step:
            ddd = step - i
        elif i == step:
            ddd = 1
        else:
            ddd = step - tl + i + 1
        if ddd < 1:
            ddd = 1
        co = 1+step-ddd

        vals[i] = vals[i]/co

    return vals, tokens


def prepare(model, text):
    model, tokenizer, preprocess = model
    if preprocess is not None:
        text = preprocess(text)
    tokens = text2tokentensors(tokenizer, text)
    return model, tokenizer, tokens


def text2tokentensors(tokenizer, text):
    tokens_tensor = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    tokens_tensor = tensor2device(tokens_tensor)
    return tokens_tensor


def ini(modelname):
    if modelname not in initiated_models:
        if not lock or modelname in modellist:
            initiated_models[modelname] = tensor2device(AutoModelForCausalLM.from_pretrained(modelname)),\
                                          AutoTokenizer.from_pretrained(modelname)
    model, tokenizer = initiated_models[modelname]
    prep = get_preprocess(modelname)
    return model, tokenizer, prep


error = "SOME OF THE REQUIRED PREPROCESSING FILES ARE NOT PRESENT"
