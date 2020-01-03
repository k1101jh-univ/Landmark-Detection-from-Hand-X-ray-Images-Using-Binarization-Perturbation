import os
import io
import base64
from flask import Flask, request, send_file
from visualization import web_output
app = Flask(__name__)


@app.route('/api/fileUpload', methods=['POST'])
def upload_file():
    f = request.files['file']
    filepath = './uploads/' + f.filename
    filepath = os.path.abspath(filepath)
    resultpath = os.path.abspath('./result/')
    f.save(filepath)

    web_output.get_output(filepath, resultpath)
    with open(resultpath + '/result.png', 'rb') as bites:
        str = base64.b64encode(bites.read())
        return str

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8787, debug=True)
