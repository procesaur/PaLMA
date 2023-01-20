from flask import Flask, request, Response, render_template, make_response
from helper import limit
from os import environ
from transformerworks import gengen, full_eval, visualize, modellist


app = Flask(__name__)
app.config["DEBUG"] = False


@app.route('/')
def home():
    return render_template('index.html', data=modellist)


@app.route('/4api')
def api_help():
    return render_template('api_help.html', data=request.root_url)


@app.route('/help')
def about():
    return render_template('help.html')


@app.route('/api', methods=['POST', 'GET'])
def api():
    args = process_req(request)
    if len(args) == 1:
        report, vectors, tokens, perps = full_eval(*args)
        return render_template("evaluation_report.html", enum=enumerate, round=round,
                               report=report, vectors=vectors, tokens=tokens, perps=perps)
    elif len(args) == 3:
        perp, model, vals, tokens = visualize(*args)
        return render_template("perplexity_report.html", perp=perp, vals=vals, tokens=tokens, model=model)
    else:
        outs, perplexities, best, label = gengen(*args)
        return render_template("generation_report.html", zip=zip,
                               generated=outs, perplexities=perplexities, best=best, label=label)


def process_req(req):
    query_parameters = req.args
    if len(query_parameters) == 0:
        query_parameters = req.form
    text = limit(query_parameters.get('data'), 1, 300)
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


if __name__ == "__main__":
    port = int(environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
