
import pandas_datareader as web
import numpy as np
import datetime as dt
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

sc1 = MinMaxScaler(feature_range=(0, 1))

df1 = web.DataReader("GOOGL", data_source='yahoo', start='2012-01-01')
df2 = web.DataReader("AAPL", data_source='yahoo', start='2012-01-01')
df3 = web.DataReader("GLD", data_source='yahoo', start='2012-01-03')
df4 = web.DataReader("BTC", data_source='yahoo', start='2020-10-01')
df5 = web.DataReader("DOGE-USD", data_source='yahoo', start='2017-11-09')

model1 = keras.models.load_model('model1.h5')
model2 = keras.models.load_model('model2.h5')
model3 = keras.models.load_model('model3.h5')
model4 = keras.models.load_model('model4.h5')
model5 = keras.models.load_model('model5.h5')

app = Flask(__name__, template_folder="")


def next5days(model, date, ch):
    go_quote1 = web.DataReader(ch, data_source='yahoo', start='2012-01-01', end=date)
    ndf1 = go_quote1.filter(['Close'])
    l60d1 = ndf1[-60:].values
    l60d_sc1 = sc1.fit_transform(l60d1)
    X_test1 = []
    X_test1.append(l60d_sc1)
    X_test1 = np.array(X_test1)
    X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
    pred_price1 = model.predict(X_test1)
    pred_price1 = sc1.inverse_transform(pred_price1)
    pr = pred_price1[0][0]
    return pr

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/anal")
def ana():
    return render_template("analysis.html")


@app.route("/anal2")
def analy():
    ch = request.args.get("txt")
    if ch == "GOOGL":
        tr = df1.tail(10).to_html()
    elif ch == "AAPL":
        tr = df2.tail(10).to_html()
    elif ch == "GLD":
        tr = df3.tail(10).to_html()
    elif ch == "BTC":
        tr = df4.tail(10).to_html()
    elif ch == "DOGE-USD":
        tr = df5.tail(10).to_html()
    return render_template("anal2.html", trade=tr, code=ch)


@app.route("/pred")
def fun():
    return render_template("pred_main.html")

@app.route("/pred2")
def next():
    ch = request.args.get("txt")
    date = request.args.get("date")
    if (ch == "GOOGL"):
        model = model1
    elif (ch == "AAPL"):
        model = model2
    elif (ch == "GLD"):
        model = model3
    elif (ch == "BTC"):
        model = model4
    elif (ch == "DOGE-USD"):
        model = model5

    go_quote1 = web.DataReader(ch, data_source='yahoo', start='2012-01-01', end=date)
    ndf1 = go_quote1.filter(['Close'])
    l60d1 = ndf1[-60:].values
    l60d_sc1 = sc1.fit_transform(l60d1)
    X_test1 = []
    X_test1.append(l60d_sc1)
    X_test1 = np.array(X_test1)
    X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
    pred_price1 = model.predict(X_test1)
    pred_price1 = sc1.inverse_transform(pred_price1)
    pr = pred_price1[0][0]

    tdy = dt.datetime.today()
    d1 = tdy + dt.timedelta(days=1)
    d2 = d1 + dt.timedelta(days=1)
    d3 = d2 + dt.timedelta(days=1)
    d4 = d3 + dt.timedelta(days=1)
    d5 = d4 + dt.timedelta(days=1)

    p1 = next5days(model, d1, ch);

    return render_template("index2.html", pred=pr, p1=p1)


if __name__ == "__main__":
    app.run()
