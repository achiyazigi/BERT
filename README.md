# AlephBERT Production for sentence completion
- <blockquote class="imgur-embed-pub" lang="en" data-id="a/SrmBBKP" data-context="false" ><a href="//imgur.com/a/SrmBBKP"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
## setup:

If you are using a unix-based machine (aka a mac or a machine with a linux OS),
you dont need to clone anything by yourself! Just run

```
curl -L -o setup.sh https://raw.githubusercontent.com/maxkrakauer/BERT/production/setup.sh
sudo chmod 755
sudo ./setup.sh
```

Note that if you don't download the tune file, you won't be able to use the wiki hebrew-tuned version, only the original model. if you try to use the tuned model without completing the next step, the server will terminate.

### FineTune version setup:

If you want to use the wiki hebrew tuned version, please contact the authors to get access to **tune_6_10.pkl**; store it in the same dir with "setup.sh" and run `./setup.sh`. The rest will be taken care of by the software.

## setup from source (all users including windows users):

1. `git clone -b production https://github.com/maxkrakauer/BERT.git`
2. `cd BERT`
3. `mkdir AlephBERT-main/models/myFirstTune`
4. (optional) store tune_6_10.pkl in AlephBERT-main/models/myFirstTune
5. `cd ui/client`
6. `npm install`
7. `npm start`
8. open "http://localhost:8080"

### note:

If you havn't completed step number 4, you can only use original the model.

## Start up after installation:

If you're using mac OSX, the browser will be opened automaticaly with this [link](http://localhost:8080).
If you're using linux or you closed the browser, open your favorate browser and enter "http://localhost:8080".
If you have already completed the setup but the server port was closed, open terminal at BERT/ui/client
and enter `npm start`.

## Requirements:

All requirement are handled during the "setup.sh" stage, but if you have experience with github, and nodejs, and are experieced with installing various libraries from their source, here are the dependencies that you need:

- python^3.8, git (latest), npm (latest)
- python3.8 modules:
- torch, transformers, pickle\*

### nodejs modules:

- npm
