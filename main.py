import ML
import ML_Rookies

import json
import argparse

if __name__ == "__main__":
    ML.main()
    ML_Rookies.main()

    # parser for input parameters
    parser = argparse.ArgumentParser(description="Machine Learning")
    parser.add_argument("output_file", type=str, help="path to output json file")
    args = parser.parse_args()

    # Read files 
    with open('Lubina_Jan_all_nba.json', 'r') as file:
        all_nba_teams = json.load(file)

    with open('Lubina_Jan_rookies.json', 'r') as file:
        rookie_teams = json.load(file)

    # Add rookies to variable of all_nba_teams
    all_nba_teams.update(rookie_teams)

    # Save to other file
    with open(args.output_file, 'w') as file:
        json.dump(all_nba_teams, file, indent=2)

    