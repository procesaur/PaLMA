from flask import Flask, request, Response, render_template, make_response
from helper import process_req
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


@app.route('/api', methods=['POST', 'GET'])
def api():
    args = process_req(request)
    if len(args) == 1:
        report, vectors, tokens = full_eval(*args)
        return render_template("evaluation_report.html", report=report, vectors=vectors, tokens=tokens, enum=enumerate)
    elif len(args) == 3:
        perp, model, vals, tokens = visualize(*args)
        return render_template("perplexity_report.html", perp=perp, vals=vals, tokens=tokens, model=model)
    else:
        outs, perplexities, best, label = gengen(*args)
        return render_template("generation_report.html", zip=zip,
                               generated=outs, perplexities=perplexities, best=best, label=label)


if __name__ == "__main__":
    port = int(environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
