import json

def main():

    rookies = {
    "first rookie all-nba team": [
      "Chet Holmgren",
      "Brandon Miller",
      "Jaime Jaquez Jr.",
      "Victor Wembanyama",
      "Keyonte George"
    ],
    "second rookie all-nba team": [
      "Brandin Podziemski",
      "Scoot Henderson",
      "Toumani Camara",
      "Bilal Coulibaly",
      "Cason Wallace"
    ]}

    with open('Lubina_Jan_rookies.json', 'w') as file:
        json.dump(rookies, file, indent=2)

if __name__ == "__main__":
    main()