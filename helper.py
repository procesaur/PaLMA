from os import name, remove


def process_req(req):
    query_parameters = req.args
    if len(query_parameters) == 0:
        query_parameters = req.form
    text = query_parameters.get('data')
    model = query_parameters.get('model')
    if "eval" in query_parameters:
        args = [text]
    elif "pv" in query_parameters:
        step = limit(int(query_parameters.get('step')), 2, 10)
        args = [text, model, step]
    else:
        length = limit(int(query_parameters.get('len')), 1, 100)
        temp = limit(float(query_parameters.get('temp')), 0, 1)
        if query_parameters.get('count'):
            cn = limit(int(query_parameters.get('count')), 1, 10)
        else:
            cn = 1
        alt = query_parameters.get('alt')
        args = [text, model, length, temp, alt, cn]
    return args


def limit(x, min, max):
    if x < min:
        x = min
    if x > max:
        x = max
    return x


def isWindows():
    return name == 'nt'


def trydel(file):
    try:
        remove(file)
    except:
        pass


def perp2prob(x):
    if x > 0:
        return 1/x
    else:
        return 1
