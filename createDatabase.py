import sqlite3
from matches import Matches
import tbapy

tba = tbapy.TBA("t4yylfKcNhCtrQuwUIPAQSFwSKAJqYy47uDPa2lPjg8TxIOJpcJ7SmilD4cIxdhZ")

con = sqlite3.connect("tba.db")
print("Database Connected")

cur = con.cursor()

#create table
cur.execute("""CREATE TABLE IF NOT EXISTS matches (
			key TEXT PRIMARY KEY,
			red1 INTEGER, 
			red2 INTEGER, 
			red3 INTEGER, 
			redScore INTEGER, 
			blue1 INTEGER, 
			blue2 INTEGER, 
			blue3 INTEGER, 
			blueScore INTEGER)""")

cur.execute("""CREATE TABLE IF NOT EXISTS teams (
			teamNum INTEGER PRIMARY KEY,
			autoLine REAL,
			autoSwitch REAL,
			autoScale REAL,
			teleSwitch REAL,
			teleScale REAL,
			oppoSwitch REAL,
			exchange REAL,
			climb REAL,
			fouls REAL,
			techFouls REAL)""")

#get all teams that played this year
teams = []
for i in range(15):
	teams.append(tba.teams(i, year=2018))
	print(i)

print("teams got")

#fetch every match from 2018 on TBA and insert into db
#also
for i in range(6, 15):
	for team in teams[i]:	#cycle through each team
		matches = tba.team_matches(team["team_number"], year=2018)

		autoLine = 0
		autoSwitch = 0
		autoScale = 0
		teleSwitch = 0
		teleScale = 0
		oppoSwitch = 0
		exchange = 0
		climb = 0
		fouls = 0
		techFouls = 0

		for match in matches:
			#calculate match scores
			alli = None
			slot = None	#1, 2, 3
			oppCol = None

			for j in range(3):
				if int(match["alliances"]["red"]["team_keys"][j][3:]) == team["team_number"]:
					alli = "red"
					oppCol = "blue"
					slot = j + 1
				elif int(match["alliances"]["blue"]["team_keys"][j][3:]) == team["team_number"]:
					alli = "blue"
					oppCol = "red"
					slot = j + 1

			if match["key"] != "2018oncmp_sf1m2":			#this match wasn't played for some reason idk leaving it in for now to save the hassle of debugging
				if match["score_breakdown"][alli]["autoRobot" + str(slot)] != None:
					autoLine += 5

				autoSwitch += match["score_breakdown"][alli]["autoSwitchOwnershipSec"] * 2
				autoScale += match["score_breakdown"][alli]["autoScaleOwnershipSec"] * 2
				teleSwitch += match["score_breakdown"][alli]["teleopSwitchOwnershipSec"] + match["score_breakdown"][alli]["teleopSwitchForceSec"]
				teleScale += match["score_breakdown"][alli]["teleopScaleOwnershipSec"] + match["score_breakdown"][alli]["teleopScaleForceSec"]
				oppoSwitch -= match["score_breakdown"][oppCol]["teleopSwitchOwnershipSec"] - match["score_breakdown"][oppCol]["teleopSwitchForceSec"]
				exchange += match["score_breakdown"][alli]["vaultPoints"]

				if match["score_breakdown"][alli]["endgameRobot" + str(slot)] == "Climbing":
					climb += 30
				elif match["score_breakdown"][alli]["endgameRobot" + str(slot)] == "Parking":
					climb += 5
				else:
					climb += 0

				fouls -= match["score_breakdown"][alli]["foulCount"] * 5
				techFouls -= match["score_breakdown"][alli]["techFoulCount"] * 15

			#add match to database
			q = (match["key"],
				int(match["alliances"]["red"]["team_keys"][0][3:]),		#red1 ([3:] slices string to convert frc7034 to 7034, convert that string to int before table insert)
				int(match["alliances"]["red"]["team_keys"][1][3:]),
				int(match["alliances"]["red"]["team_keys"][2][3:]),
				match["alliances"]["red"]["score"],
				int(match["alliances"]["blue"]["team_keys"][0][3:]),
				int(match["alliances"]["blue"]["team_keys"][1][3:]),
				int(match["alliances"]["blue"]["team_keys"][2][3:]),
				match["alliances"]["red"]["score"])

			cur.execute("INSERT OR IGNORE INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ", q)
		if len(matches) != 0:
			autoLine = autoLine / len(matches)
			autoSwitch = autoSwitch / len(matches)
			autoScale = autoScale / len(matches)
			teleSwitch = teleSwitch / len(matches)
			teleScale = teleScale / len(matches)
			oppoSwitch = oppoSwitch / len(matches)
			exchange = exchange / len(matches)
			climb = climb / len(matches)
			fouls = fouls / len(matches)
			techFouls = techFouls / len(matches)

			a = (team["team_number"], autoLine, autoSwitch, autoScale, teleSwitch, teleScale, oppoSwitch, exchange, climb, fouls, techFouls)

			cur.execute("INSERT OR IGNORE INTO teams VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", a)

	print(100*(i+1)/15, " percent done.")

con.commit()
con.close()