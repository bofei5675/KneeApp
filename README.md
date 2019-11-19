# Knee Web App

[![](https://img.shields.io/badge/python-2.7%2C%203.5%2B-green.svg)]()
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
------------------
## Update
- [] Working on automating deployment like `gunicorn`
- [x] 2019/11/15, deploy this web app on the Langone server, which supports the access via ssh tunnel 
- [x] 2019/11/11, finish basic detector and classifier pipeline 

------------------

## Getting started in 10 minutes(local)

- Clone this repo 
- Install requirements
- Run the script
- Check http://localhost:5001
- Done! :tada:

## Getting start on the server (will change to a better way later)

- Login the server IP address: `10.189.38.45`
- Git clone this repo, and run the app by `python app.py`
- open a tunnel by the following command
```
ssh -N -L 5001:127.0.0.1:5001 bz1030@10.189.38.45
```
- Open the browser and put `localhost:5001` to see this app

------------------

## Local Installation

### Clone the repo
```shell
$ git clone https://github.com/mtobeiyf/keras-flask-deploy-webapp.git
```

### Install requirements

```shell
$ pip install -r requirements.txt
```

Make sure you have the following installed:
- tensorflow
- keras
- flask
- pillow
- h5py
- gevent

### Run with Python

Python 2.7 or 3.5+ are supported and tested.

```shell
$ python app.py
```

### Play

Open http://localhost:5001 and have fun. :smiley:. Port will be configured inside `app.py`.

------------------

## Customization

### Use your own model

Place your trained `.h5` file saved by `torch.save()` under models directory.


### Use other pre-trained model

Check out `torchvision` for other pre-trained model.

### UI Modification

Modify files in `templates` and `static` directory.

`index.html` for the UI and `main.js` for all the behaviors

## Deployment

To deploy it for public use, you need to have a public **linux server**.

### Run the app

Run the script and hide it in background with `tmux` or `screen`.

```
$ python app.py
```

You can also use gunicorn instead of gevent
```
$ gunicorn -b 127.0.0.1:5001 app:app
```

More deployment options, check [here](http://flask.pocoo.org/docs/0.12/deploying/wsgi-standalone/)

### Set up Nginx

To redirect the traffic to your local app.
Configure your Nginx `.conf` file.
```
server {
    listen  80;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:5000;
    }
}
```

## More resources

Check Siraj's ["How to Deploy a Keras Model to Production"](https://youtu.be/f6Bf3gl4hWY) video. The corresponding [repo](https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production).

[Building a simple Keras + deep learning REST API](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
