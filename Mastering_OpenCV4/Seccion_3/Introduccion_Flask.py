# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:13:36 2021

@author: MBI
"""
from  flask import Flask
#%%
# Introduccion al framework Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

if __name__ == "__main__":
    app.run()
#%%
# Haciendolo accesible  desde cualquier dispositivo

app = Flask(__name__)
@app.route("/")
def hello_external():
    return "Hello World !"

if __name__ ==  "__main__":
    app.run(host = '0.0.0.0')

#%%
app = Flask(__name__)
@app.route("/")
def hello ():
    return "Hello World !"

@app.route("/user") # Usando la ip del server y acontinuacion /user se obtiene este mensaje
def hello_user(): 
    return "User: Hello World !"

if __name__ == "__main__":
    app.run(host='0.0.0.0')

#%%