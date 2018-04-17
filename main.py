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

batch_size = 1

def getScoreData():      #15373 matches
	samp = random.randint(1, 15373)
	alli = random.randint(0,1)
	team = []
	score = []
	cur.execute("SELECT * FROM matches WHERE rowid=?", (samp,))
	r = cur.fetchone()
	if alli == 1:
		team.append(cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED1.value],)).fetchone()[1:] +       #fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED2.value],)).fetchone()[1:] +
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED3.value],)).fetchone()[1:])
		score.append(r[Matches.REDSCORE.value])
	else:
		team.append(cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE1.value],)).fetchone()[1:] +      #fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE2.value],)).fetchone()[1:] + 
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE3.value],)).fetchone()[1:])
		score.append(r[Matches.BLUESCORE.value])

	return team, score

def getMatchData():
	samp = random.randint(1, 15373)
	team = []
	win = []
	cur.execute("SELECT * FROM matches WHERE rowid=?", (samp,))
	r = cur.fetchone()
	team.append(cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED1.value],)).fetchone()[1:] +       #fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
				cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED2.value],)).fetchone()[1:] +
				cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED3.value],)).fetchone()[1:] +
				cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE1.value],)).fetchone()[1:] +      #fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
				cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE2.value],)).fetchone()[1:] + 
				cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE3.value],)).fetchone()[1:])
	red = r[Matches.REDSCORE.value]
	blue = r[Matches.BLUESCORE.value]
	if red > blue:
		win = [1,0,0]
	elif red < blue:
		win = [0,1,0]
	else:
		win = [0,0,1]

	return team, win

#a mxn b pxq
#ab mxq if n=p
def scoreModel():
	x = tf.placeholder(tf.float32, [batch_size, 30])        #10 features times 3 teams
	y_ = tf.placeholder(tf.float32, [batch_size, 1])

	#builds the tensorflow graph for the model
	weights1 = tf.get_variable("weights1", [30, 30])
	bias1 = tf.get_variable("bias1", [batch_size, 30])
	preac1 = tf.matmul(x, weights1) + bias1
	#ac1 = tf.nn.softmax(preac1)

	weights2 = tf.get_variable("weigts2", [30, 1])
	bias2 = tf.get_variable("bias2", [batch_size, 1])
	preac2 = tf.matmul(preac1, weights2) + bias2
	y = tf.nn.relu(preac2)

	loss = abs(y_-y)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	train_step = optimizer.minimize(loss)

x = tf.placeholder(tf.float32, [batch_size, 60])        #10 features times 3 teams
y_ = tf.placeholder(tf.float32, [batch_size, 3])

#builds the tensorflow graph for the model
weights1 = tf.get_variable("weights1", [60, 30])
bias1 = tf.get_variable("bias1", [batch_size, 30])
preac1 = tf.matmul(x, weights1) + bias1
ac1 = tf.nn.softmax(preac1)

weights2 = tf.get_variable("weigts2", [30, 1])
bias2 = tf.get_variable("bias2", [batch_size, 3])
preac2 = tf.matmul(ac1, weights2) + bias2
y = tf.nn.softmax(preac2)

yArg = tf.argmax(preac2, axis=1)

y_Arg = tf.argmax(y_, axis=1)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_Arg, logits=preac2)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss)

#accuracy = tf.metrics.accuracy(labels=y_Arg, predictions=yArg)

def scoreAcc(pred, targ):
	if pred + 20 >= targ and pred - 20 <= targ:
		return 1
	else:
		return 0


def train():
	with tf.Session() as sess:
		saver = tf.train.Saver(tf.global_variables())
		tf.global_variables_initializer().run()
		losses = 0
		for i in range(384375):       #num epochs (about 25 times over dataset)
			teamData, score_ = getMatchData()
			score = np.reshape(score_, [batch_size, 3])
			feed_dict = {x: teamData, y_: score}
			_, locloss = sess.run([train_step, loss], feed_dict=feed_dict)
			losses += locloss
			if i % 2000 == 0:
				print("Epoch:", i, "Loss:", losses/2000)
				losses = 0

		saver.save(sess, "E:/Documents/Programs/FRC-Predictor-2018/saves/train2.ckpt")

def getTeams(team1, team2, team3, team4, team5, team6):
	x = [cur.execute("SELECT * FROM teams WHERE teamNum=?", (team1,)).fetchone()[1:] + 
		cur.execute("SELECT * FROM teams WHERE teamNum=?", (team2,)).fetchone()[1:] + 
		cur.execute("SELECT * FROM teams WHERE teamNum=?", (team3,)).fetchone()[1:] +
		cur.execute("SELECT * FROM teams WHERE teamNum=?", (team4,)).fetchone()[1:] + 
		cur.execute("SELECT * FROM teams WHERE teamNum=?", (team5,)).fetchone()[1:] + 
		cur.execute("SELECT * FROM teams WHERE teamNum=?", (team6,)).fetchone()[1:]]
	return np.reshape(x, [1, 60])

train()

'''with tf.Session() as sess:
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state("E:/Documents/Programs/FRC-Predictor-2018/saves/train1.ckpt")
	saver.restore(sess, "E:/Documents/Programs/FRC-Predictor-2018/saves/train1.ckpt")

	data = getTeams(1511, 1405, 156)# + getTeams(7034, 254, 1425)
	out = sess.run(y, feed_dict={x:data})
	print(out)
	data = getTeams(271, 3173, 6414)
	out = sess.run(y, feed_dict={x:data})
	print(out)
	data = getTeams(7034, 254, 1425)
	out = sess.run(y, feed_dict={x:data})
	print(out)'''

con.close()
