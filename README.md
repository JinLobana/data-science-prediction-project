# projekt predykcja drużyn All Nba i Rookies

Uruchomienie skryptu: ```python3 main.py Lubina_Jan.json```  

- Moim pierwotnym pomysłem było wykorzystanie klasyfikatora do trzech klas, pierwszej, drugiej i trzeciej drużyny ALL NBA. Jednak stwarzało to dużo problemów, więc koniec końców zdecydowałem się na prostszą metodę, wykorzystanie klasyfikatora binarnego i wykorzystanie funkcji ```predict_proba_()```, do wyszukania najwyższych prawdopodobieństw i przypisania ich po kolei do drużyn ALL NBA.  
- Chciałem utworzyć zestaw klasyfikatorów, nauczonych z dataset'ów z różnych sezonów i na koniec wykorzystać klasyfikator głosujący. Niestety ostatecznie najlepsze wyniki uzyskałem po połączeniu wszystkich sezonów w jeden duży dataset i nauczeniu klasyfikatora tylko na nim. 
- Użyłem jednego klasyfikatora, *histogram gradient boosting classifier*, ponieważ obsługuje wartości NaN oraz wcześniej z niego nie korzystałem ;) Szybko liczy, a także daje zadowalające wyniki, dlatego później już nie zmieniałem
- Wykorzystałem custom'owy Grid Search. Starałem się wykorzystać już zaimplementowany, jednak walidacja krzyżowa w przypadku danych gdzie próbki w klasach są tak bardzo nie zbalansowane (500 graczy nie dostaje nagrody, a tylko 15 tak) psuje wyniki, model się uczy przypisywać prawie zawsze 0 (czyli brak nagrody). Implementowałem także Random Search, ale nie byłem zadowolony z wyników. Prawdopodbnie były poprawne, aczkolwiek potem już mi się nie chciało ponownie tego pisać. 
  - Mój Grid Search optymalizuje precyzję wyniku (precision score).
- Niewykorzystane funkcje (jeśli takowe są) znajdują się zakomentowane na końcu plików. 

## Przygotowanie danych
### All NBA
Plik ```preparing_data.py```  
      W skrócie jest to zestaw funkcji, który pobiera dane z api, następnie je odpowiednio przetwarza i zapisuje do plików w formacie csv.  

```python
get_players_stats_from_season(Season)
```  
> Powyższa funkcja pobiera z endpointu *PlayerGameLogs* informacje o statystykach w każdym meczu, dla każdego gracza, z całego sezonu zasadniczego. Następnie tworzy dataframe, wybiera odpowiednie kolumny (dane, której uważam za istotne, t.j. usuwałem redundantne, np. FG_PCT (percentage) zawiera w sobie informacje o FG_M (made) i FG_A (attempted)), i pod koniec sortuje od najniższego *Player_id* do najwyższego. Zapisuje wynik do pliku. 
    wcięcie

```python
saving_stats_csv()
```
> W odpowiedni sposób wybiera sezon w pętli for, następnie wywołuje funkcję *get_players_stats_from_season*, która zapisuje dane do plików.

```python
get_unique_id_and_df(Season)
```
> Dla danego sezonu wyodrębnia *Player_id* każdego gracza w sezonie, t.j. lista ids graczy w danym sezonie.

```python
calculate_mean_value(dataframe, PlayerID)
```

> Funkcja liczy średnią z wszystkich statystyk danego gracza z sezonu, dodaje kolumnę w ilu meczach w sezonie zagrał.

```python
saving_avg_values()
```
> Funkcja wykorzystuje *get_unique_id_and_df* oraz *calculate_mea_value*. Zapisuje do plików uśrednione statystyki, używane do późniejszego uczenia. 

```python
main()
``` 
> Tylko wywołuje *saving_stats_csv* oraz *saving_avg_values*.

### Rookies

Plik ```prepating_data_rookies.py```  
Otrzymałem plik "*stats/from_freinds/player_avg_data.csv*", na jego podstawie jedną funkcją wyodrębniam i zapisuję do plików sezon po sezonie dane dla debiutantów oraz graczy, którzy uzyskali nagrodę all rookies. 

```python
add_rookies()
```
> Odczytanie *player_avg_data.csv*, wybranie odpowiednich kolumn, sortowanie i zapisywanie sezon po sezonie wszystkich debiutantów i tych co otrzymali nagrody.

## Uczenie maszynowe
### ALL NBA
Plik ```ML.py```  
Wczytuje statystyki oraz wyniki (plik all_nba.csv), porównuje dane, tworzy wizualizację, przeszukuje hiperparametry customowym **grid search**, 
na koniec implementuje uczenie maszynowe, wybrany zostaje model z najlepszymi wynikami. 


```python
reading_all_nba()
``` 
> Odczytanie nazwisk graczy wybranych do all NBA w każdym roku, zapisanie do pliku w tym samym formacie ich ID. 

```python
reading_from_csv()
``` 
> Odczytuje statystyki, zapisuje do pandas dataframe, odczuca (po analizie) część graczy, którzy nie mają szans na zdobycie nagrody (wyrównanie proporcji klasyfikacji), tworzy listę wszystkich dataframe. 

```python 
searching_through_df(df, ids)
``` 

> Przeszukuje dataframe ze statystykami z danego sezonu, szuka *Player_ID*, graczy którzy dostali nagrodę. Tworzy listę Y o długości dataframe, z informacją  
**1** = wybrany do którejkolwiek drużyny all NBA oraz  
**0** = nie wybrany. 

```python
creating_list_of_all_nba_teams(X_list)
```
 
> wykorzystuje funkcje *searching_through_df* do stworzenia listy list zakwalifikowanych do nagrody graczy dla każdego sezonu.

```python
visualisation(X, Y)
```
> Funkcja wyświetlająca wszystkie interesujące statystyki, ich korelacje itp.

```python
get_names_from_id(ids)
```
> Funkcja zwraca nazwiska z listy id, używa api.

```python
find_15_best(array)
```
> Funkcja otrzymuje macierz prawdopodobieństwa zwróconą przez metodę *predict_proba()* klasyfikatora **HistGradientBoostClassifier**. Tworzy listy 1, 2, 3 drużyny all NBA. 

```python
merge_all_data(X_list, Y_list)
```
> W celu uczenia pojedynczego klasyfikatora funkcja łączy listy dataframe'ów w jeden ogromny zestaw danych wejściowych i wyjściowych.


```python
get_best_hyper_parameters(X_list, Y_list, choose_method)
```
> Przeszukiwanie hiper parametrów. Dla *choose_method* równej
>- 0 - Tworzenie i zapisywanie do słownika najlepszych parametrów klasyfikatorów, które uzyskały najlepszą precyzję, dla danych uczących ograniczonych do pojedynczego sezonu.
>- 1 - To samo co powyżej, ale wykluczam dany sezon i dla niego licze predykcje i precyzję. Liczyło się całą noc 8\)
>- 2 - Po prostu znalezienie najlepszych parametrów dla sezonu 2023-24, dla połączonych danych

```python
ensembling_classifiers(X_list, Y_list, choose_method)
```
> Uczenie maszynowe i inferencja. Zaimplementowane trzy metody do wyboru, ostatecznie wybrana ostatnia (dawała najlepsze rezultaty)
>- 0 - Tworzenie klasyfikatora głosującego z wcześniej wyszkolonych mniejszych *HistGradientBoostingClassifier* z metody 0 w funkcji *get_best_hyper_parameters*
>- 1 - Tworzenie klasyfikatora głosującego z wcześniej wyszkolonych mniejszych *HistGradientBoostingClassifier* z metody 1 w funkcji *get_best_hyper_parameters*
>- 2 - Inferencja dla połączonych danych, tylko jeden *HistGradientBoostingClassifier*

```python
get_names_from_Y_list(X, all_nba_or_not)
```
> Z output naszego uczenia, czyli listy Y (*all_nba_or_not*) ekstrakcja *Player_ID*'s i znalezienie nazwisk zawodników wykorzystując funkcję *get_names_from_id*

```python
main()
```
> Wywołanie odpowiednich funkcji. Ostatecznie aktywne tylko wczytanie danych i inferencja maszynowa. 

### Rookies

Plik ```ML_Rookies.py```  
Funkcjonalność praktycznie identyczna do ```ML.py```, funkcje były tylko kopiowane i dostosowane do danych. 

