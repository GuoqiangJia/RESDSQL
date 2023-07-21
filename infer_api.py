import config
from flask import Flask
from flask_cors import *
from flask_httpauth import HTTPBasicAuth
import json
import logger
import static

app = Flask(__name__)
CORS(app, supports_credentials=True)

LOG = logger.get_logger(__name__, 'api.log')


@app.route('/<date>/tops')
@app.route('/tops', defaults={'date': None})
def tops(date):
    LOG.info('tops requests')
    if date is None:
        date = static.get_today()

    try:
        with open(config.base_dir + date + '-stocks.json') as stocks:
            stocks_dict = json.load(stocks)
    except FileNotFoundError:
        response = app.response_class(response='Not found data', status=403)
        return response

    response = app.response_class(
        response=json.dumps(stocks_dict),
        status=200,
        mimetype='application/json'
    )
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
    LOG.info(stocks_dict)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444, debug=True)
