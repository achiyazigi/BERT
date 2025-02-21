const express = require("express");
const bodyParser = require('body-parser')
const PythonShell = require('python-shell')

const path = require('path');
const app = express();
const router = express.Router();
const PORT = process.env.PORT || 8080;

var ans_en = 'no results yet'
var ans_heb = 'עדיין אין תוצאות...'

app.use(bodyParser.json());

app.use(bodyParser.urlencoded({ extended: true }));

app.use('/', router);

app.get("/heb", (req, res) => {
  res.json({ message: {ans_heb} });
});
app.get("/en", (req, res) => {
  res.json({ message: {ans_en} });
});

const execute_bert = (lang, text, mask, res, original)=>{
  var scriptPath = '';
  if(lang === 'en'){
    scriptPath = '../../AlephBERT-main/models/alephbert-base/english-model.py'
  }
  else if(lang === 'heb'){
    scriptPath = '../../AlephBERT-main/models/alephbert-base/hebrew-model.py'
  }
  const pyshell = new PythonShell.PythonShell(scriptPath, { args: [text, mask, original] })
  pyshell.on('message', (output) => {
    console.log('result:' + output)
    if(lang === 'en'){
      ans_en = output;
    }
    else if(lang === 'heb'){

      ans_heb = output;
    }
    res.send({response: output});
    return output

  })
  pyshell.end((err) => {
    if (err) {
      throw err;
    }
  })
  return 'somthing went wrong...';
}

app.post("/heb", (req, res) => {
  const body = req.body;
  const text = body.text;
  const mask = body.mask;
  const original = body.original;

  console.log('original: ' + original)
  console.log('working on: ' + text)
  console.log('with mask: ' + mask)
  ans_heb = 'loading...';
  ans_heb = execute_bert('heb', text, mask, res, original)
  return ans_heb
})

app.post("/en", (req, res) => {
  const body = req.body;
  const text = body.text;
  const mask = body.mask;
  const original = body.original;

  console.log('original: ' + original)
  console.log('working on: ' + text)
  console.log('with mask: ' + mask)
  ans_en = 'loading...';
  ans_en = execute_bert('en', text, mask, res, original)
  return ans_en;
})

app.use(express.static(__dirname));
app.use(express.static(path.join(__dirname, 'build')));
app.get('/ping', function (req, res) {
 return res.send('pong');
});
app.get('/*', function (req, res) {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server listening on ${PORT}`);
  });
  