import json

class Score():
    def __init__(self, baseline_json):
        with open(baseline_json) as f:
           self.baselineData = json.load(f)
           
    def getScore(self, data_json):
        with open(data_json) as f:
            test_data = json.load(f)

        ### all-nba team score
        all_nba_score = 0

        # First calculate score for correct players in a given team (10 pts for each), in adjacent team (8 pts for each), and 2 teams away (6 pts for each)
        for team_id in range(3):
            test_team = self.getAllNbaTeam(test_data, team_id)

            baseline_team = self.getAllNbaTeam(self.baselineData, team_id)
            adjacent_teams = self.getAllNbaTeam(self.baselineData, team_id - 1) + self.getAllNbaTeam(self.baselineData, team_id + 1)
            far_adjacent_teams = self.getAllNbaTeam(self.baselineData, team_id - 2) + self.getAllNbaTeam(self.baselineData, team_id + 2)
    
            for player in test_team:
                if player in baseline_team:
                    print(player, team_id, 10)
                    all_nba_score += 10
                elif player in adjacent_teams:
                    print(player, team_id, 8)
                    all_nba_score += 8
                elif player in far_adjacent_teams:
                    print(player, team_id, 7)
                    all_nba_score += 6
                else:
                    print(player, team_id, 0)

            # Now add score for correctness of each team (2/5 players - 5 points, 3/5 - 10 points, 4/5 - 20 points, 5/5 - 40 points) 
            correct_players = 0
            for player in test_team:
                if player in baseline_team:
                    correct_players += 1
 
            if correct_players == 2:
                print(team_id, 5)
                all_nba_score += 5
            elif correct_players == 3:
                print(team_id, 10)
                all_nba_score += 10
            elif correct_players == 4:
                print(team_id, 20)
                all_nba_score += 20
            elif correct_players == 5:
                print(team_id, 40)
                all_nba_score += 40
            else:
                print(team_id, 0)

        ### Calculate rookie all-nba team score
        rookie_score = 0
        
        for team_id in range(2):
            test_team = self.getRookieAllNbaTeam(test_data, team_id)

            baseline_team = self.getRookieAllNbaTeam(self.baselineData, team_id)
            adjacent_teams = self.getRookieAllNbaTeam(self.baselineData, team_id - 1) + self.getRookieAllNbaTeam(self.baselineData, team_id + 1)
    
            for player in test_team:
                if player in baseline_team:
                    print(player, team_id, 10)
                    rookie_score += 10
                elif player in adjacent_teams:
                    print(player, team_id, 8)
                    rookie_score += 8
                else:
                    print(player, team_id, 0)

            # Now add score for correctness of each team (2/5 players - 5 points, 3/5 - 10 points, 4/5 - 20 points, 5/5 - 40 points) 
            correct_players = 0
            for player in test_team:
                if player in baseline_team:
                    correct_players += 1
            
            if correct_players == 2:
                print(team_id, 5)
                rookie_score += 5
            elif correct_players == 3:
                print(team_id, 10)
                rookie_score += 10
            elif correct_players == 4:
                print(team_id, 20)
                rookie_score += 20
            elif correct_players == 5:
                print(team_id, 40)
                rookie_score += 40
            else:
                print(team_id, 0)

        return all_nba_score + rookie_score
    

    def getAllNbaTeam(self, data, team_id):
        if team_id == 0:
            res = data['first all-nba team']
        elif team_id == 1:
            res =  data['second all-nba team']
        elif team_id == 2:
            res =  data['third all-nba team']
        else:
            res = []
        return res
    
    def getRookieAllNbaTeam(self, data, team_id):
        if team_id == 0:
            res = data['first rookie all-nba team']
        elif team_id == 1:
            res = data['second rookie all-nba team']
        else:
            res = []
        return res

if __name__ == "__main__":
    score = Score('correct_result.json')
    score_value = score.getScore('Lubina_Jan.json')
    print(f'{score_value} / {900}')