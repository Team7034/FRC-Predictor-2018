import sqlite3
import tensorflow as tf
import random
import numpy as np
from matches import Matches

con = sqlite3.connect("tba.db")
print("Database Connected")

cur = con.cursor()

cur.execute("SELECT * FROM matches WHERE rowid=2345")

print(cur.fetchone())

batch_size = 2

def getData():      #15373 matches
    samp = random.sample(range(1, 15374), 1)
    team = []
    score = []
    for i, s in enumerate(samp):
        q = (s,)
        cur.execute("SELECT * FROM matches WHERE rowid=?", q)
        r = cur.fetchone()
        team.append(cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED1.value],)).fetchone()[1:] +       #fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
                    cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED2.value],)).fetchone()[1:] +
                    cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED3.value],)).fetchone()[1:])
        score.append(r[Matches.REDSCORE.value])
        team.append(cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE1.value],)).fetchone()[1:] +      #fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
                    cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE2.value],)).fetchone()[1:] + 
                    cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE3.value],)).fetchone()[1:])
        score.append(r[Matches.BLUESCORE.value])

    return team, score

#a mxn b pxq
#ab mxq if n=p

x = tf.placeholder(tf.float32, [batch_size, 30])        #10 features times 3 teams
y_ = tf.placeholder(tf.float32, [batch_size, 1])

#builds the tensorflow graph for the model
weights1 = tf.get_variable("weights1", [30, 60])
bias1 = tf.get_variable("bias1", [batch_size, 60])
preac1 = tf.matmul(x, weights1) + bias1
ac1 = tf.nn.softmax(preac1)

weights2 = tf.get_variable("weigts2", [60, 1])
bias2 = tf.get_variable("bias2", [batch_size, 1])
preac2 = tf.matmul(ac1, weights2) + bias2
y = tf.nn.relu(preac2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)

def train():
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        tf.global_variables_initializer().run()
        for i in range(1537500):       #num epochs (about 100 times over dataset)
            teamData, score = getData()
            score = np.reshape(score, [batch_size, 1])
            feed_dict = {x: teamData, y_: score}
            _, locloss = sess.run([train_step, loss], feed_dict=feed_dict)
            if i % 1500 == 0:
                print("Epoch:", i, "Loss:", locloss)

        saver.save(sess, "C:/Users/Admin/Desktop/Python Scripts/FRC-Predictor-2018/saves/train1.ckpt")

def getTeams(team1, team2, team3):
    x = [   cur.execute("SELECT * FROM teams WHERE teamNum=?", (team1,)).fetchone()[1:] +       #fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
        cur.execute("SELECT * FROM teams WHERE teamNum=?", (team2,)).fetchone()[1:] + 
        cur.execute("SELECT * FROM teams WHERE teamNum=?", (team3,)).fetchone()[1:]     ]
    return x

train()
'''
with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("C:/Users/Admin/Desktop/Python Scripts/FRC-Predictor-2018/saves/train1.ckpt")
    saver.restore(sess, "C:/Users/Admin/Desktop/Python Scripts/FRC-Predictor-2018/saves/train1.ckpt")

    data = getTeams(1511, 1405, 156)# + getTeams(7034, 254, 1425)
    y = sess.run(y, feed_dict={x:data})
    print(y)'''

con.close()
