import sqlite3
import tensorflow as tf
import random
from matches import Matches

con = sqlite3.connect("tba.db")
print("Database Connected")

cur = con.cursor()
'''
team=(7034,)

cur.execute("SELECT * FROM teams WHERE teamNum=?", team)

print(cur.fetchone())

con.close()

batch_size = 10
#a mxn b pxq
#ab mxq if n=p

x = tf.placeholder(tf.float32, [batch_size, 30])		#10 features times 3 teams
y_ = tf.placeholder(tf.float32, [batch_size, 1])

#builds the tensorflow graph for the model
weights1 = tf.get_variable("weights1", [30, 30])
bias1 = tf.get_variable("bias1", [batch_size, 30])
preac1 = tf.matmul(x, weights1) + bias1
ac1 = tf.softmax(preac1)

weigts2 = tf.get_variable("weigts2", [30, 1])
bias2 = tf.get_variable("bias2", [batch_size, 1])
y = tf.matmul(ac1, weights2) + bias2

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
	saver = tf.train.Saver(tf.global_variables())
	tf.global_variables_initializer().run()
	for i in range 1000:		#num epochs
		teamData, score = getData()
		feed_dict = {x: x, y_: y}
		sess.run(train_step)
		if i % 10 == 0:
			output, loss = sess.run(y, loss)
			acc = (score - output) / score
			print(score, loss)
'''
batch_size = 2
def getData():		#15373 matches
	samp = random.sample(range(1, 15374), 1)
	team = []
	score = []
	for i, s in enumerate(samp):
		q = (s,)
		cur.execute("SELECT * FROM matches WHERE rowid=?", q)
		r = cur.fetchone()
		print(r)
		team.append(cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED1.value],)).fetchone()[1:],		#fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED2.value],)).fetchone()[1:],
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.RED3.value],)).fetchone()[1:])
		score.append(r[Matches.REDSCORE.value])
		team.append(cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE1.value],)).fetchone()[1:],		#fetches data from team number x, where x is team in slot Matches.Y from the previous sql query
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE2.value],)).fetchone()[1:],
					cur.execute("SELECT * FROM teams WHERE teamNum=?", (r[Matches.BLUE3.value],)).fetchone()[1:])
		score.append(r[Matches.BLUESCORE.value])
		print(team, score)

	return team, score


#cur.execute("SELECT * FROM matches WHERE rowid=1234")
#print(cur.fetchone())
#x, y = getData()
#print(y)
#print(x)

#cur.execute("SELECT COUNT(*) FROM teams")
cur.execute("SELECT * FROM teams WHERE teamNum=?", (254,))
print(cur.fetchone())