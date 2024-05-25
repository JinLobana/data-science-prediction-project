# projekt predykcja drużyn All Nba i Rookies

## Przygotowanie danych
### All NBA
Plik ```python preparing_data.py```  
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

## Uczenie maszynowe
Plik ```python ML.py```  
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
