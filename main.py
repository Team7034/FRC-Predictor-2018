import sqlite3
from matches import Matches

con = sqlite3.connect("tba.db")
print("Database Connected")

cur = con.cursor()

team=(7034,)

cur.execute("SELECT * FROM teams WHERE teamNum=?", team)

print(cur.fetchone())

con.close()