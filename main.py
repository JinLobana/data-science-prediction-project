import ML
import ML_Rookies

import json

if __name__ == "__main__":
    ML.main()
    ML_Rookies.main()

    # Read files 
    with open('Lubina_Jan_all_nba.json', 'r') as file:
        all_nba_teams = json.load(file)

    with open('Lubina_Jan_rookies.json', 'r') as file:
        rookie_teams = json.load(file)

    # Add rookies to variable of all_nba_teams
    all_nba_teams.update(rookie_teams)

    # Save to other file
    with open('Lubina_Jan.json', 'w') as file:
        json.dump(all_nba_teams, file, indent=2)

    