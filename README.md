# WSI - LAB 3 - Dwuosobowe gry deterministyczne

Implementacja porownuje `minimax` oraz `minimax` z obcinaniem `alpha-beta` dla gry:

- Na stole znajduje sie `N` zetonow.
- Gracze na przemian zabieraja od `1` do `K` zetonow.
- Przegrywa gracz, ktory zabierze ostatni zeton.

W eksperymentach przyjeto:

- `K = 3`
- `N` losowane jednostajnie z przedzialu `[8, 20]`
- glebokosci przeszukiwania `d in {2, 3, 4, 5}`
- `200` partii dla kazdej konfiguracji
- przeciwnik algorytmu wykonuje ruchy losowe

## Uruchomienie

```powershell
python main.py
```

Skrypt:

- uruchamia serie eksperymentow,
- zapisuje wyniki do `results.csv`,
- generuje tabele w `results.md`,
- wypisuje podsumowanie w terminalu.

## Heurystyka

Dla stanow niekoncowych przy ograniczonej glebokosci zastosowano funkcje oceny oparta o znana wlasnosc tej gry: pozycje `N = 1 mod (K + 1)` sa niekorzystne dla gracza bedacego na ruchu. Dodatkowo dodany jest niewielki skladnik zalezny od liczby pozostalych zetonow, aby preferowac stany blizsze rozstrzygnieciu.

W przypadku kilku ruchow o tej samej ocenie algorytm losuje jeden z najlepszych ruchow.

## Interpretacja wynikow

Wyniki z uruchomienia referencyjnego (`seed = 20260419`):

| Wariant | Glebokosc d | Partie | Wygrane [%] | Sredni czas ruchu [ms] | Srednia liczba wezlow |
|---|---:|---:|---:|---:|---:|
| minimax | 2 | 200 | 99.0 | 0.008 | 10.30 |
| minimax | 3 | 200 | 99.5 | 0.019 | 28.67 |
| minimax | 4 | 200 | 99.0 | 0.047 | 73.01 |
| minimax | 5 | 200 | 99.0 | 0.111 | 169.20 |
| alpha_beta | 2 | 200 | 99.0 | 0.008 | 10.30 |
| alpha_beta | 3 | 200 | 99.5 | 0.018 | 25.69 |
| alpha_beta | 4 | 200 | 99.0 | 0.036 | 51.70 |
| alpha_beta | 5 | 200 | 99.0 | 0.069 | 95.85 |

Krotka interpretacja:

- Skutecznosc obu wariantow jest praktycznie identyczna, co jest zgodne z teoria: obcinanie `alpha-beta` nie zmienia jakosci decyzji, tylko ogranicza liczbe analizowanych stanow.
- Dla malej glebokosci `d = 2` zysk jest pomijalny, bo drzewo gry jest jeszcze bardzo male.
- Im wieksza glebokosc, tym bardziej widoczna przewaga `alpha-beta`: dla `d = 5` srednia liczba odwiedzonych wezlow spadla z `169.20` do `95.85`, a sredni czas ruchu z `0.111 ms` do `0.069 ms`.
- Otrzymane odsetki wygranych sa bardzo wysokie, poniewaz algorytm gra przeciwko przeciwnikowi losowemu, a sama gra ma prosta strukture strategiczna.
