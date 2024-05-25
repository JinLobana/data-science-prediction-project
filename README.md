# projekt predykcja drużyn All Nba i Rookies

## Przygotowanie danych
### All NBA
Plik preparing_data.py  
W skrócie jest to zestaw funkcji, który pobiera dane z api, następnie je odpowiednio przetwarza i zapisuje do plików w formacie csv.  

```python
get_players_stats_from_season(Season)
```  
> Powyższa funkcja pobiera z endpointu *PlayerGameLogs* informacje o statystykach w każdym meczu, dla każdego gracza, z całego sezonu zasadniczego. Następnie tworzy dataframe, wybiera odpowiednie kolumny (dane, której uważam za istotne, t.j. usuwałem redundantne, np. FG_PCT (percentage) zawiera w sobie informacje o FG_M (made) i FG_A (attempted)), i pod koniec sortuje od najniższego *Player_id* do najwyższego. Zapisuje wynik do pliku. 




### Rookies