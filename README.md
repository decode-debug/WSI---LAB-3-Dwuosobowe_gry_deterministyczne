# WSI - LAB 3 - Dwuosobowe gry deterministyczne

Implementacja porównuje `minimax` oraz `minimax` z obcinaniem `alpha-beta` dla gry:

- Na stole znajduje się `N` żetonów.
- Gracze na przemian zabierają od `1` do `K` żetonów.
- Przegrywa gracz, który zabierze ostatni żeton.

W eksperymentach przyjęto:

- `K = 3`
- `N` losowane jednostajnie z przedziału `[8, 20]`
- głębokości przeszukiwania `d in {2, 3, 4, 5}`
- `200` partii dla każdej konfiguracji
- przeciwnik algorytmu wykonuje ruchy losowe

### Instalacja:
```bash
git clone https://github.com/decode-debug/WSI---LAB-3-Dwuosobowe_gry_deterministyczne
```

## Uruchomienie

```powershell
python main.py
```

Skrypt:

- uruchamia serię eksperymentów,
- zapisuje wyniki do `results.csv`,
- generuje tabelę w `results.md`,
- wypisuje podsumowanie w terminalu.

## Heurystyka

Dla stanów niekońcowych przy ograniczonej głębokości zastosowano funkcję oceny opartą o znaną własność tej gry: pozycje `N = 1 mod (K + 1)` są niekorzystne dla gracza będącego na ruchu. Dodatkowo dodany jest niewielki składnik zależny od liczby pozostałych żetonów, aby preferować stany bliższe rozstrzygnięciu.

W przypadku kilku ruchów o tej samej ocenie algorytm losuje jeden z najlepszych ruchów.

## Interpretacja wyników

Wyniki z uruchomienia referencyjnego (`seed = 20260419`):

| Wariant | Głębokość d | Partie | Wygrane [%] | Średni czas ruchu [ms] | Średnia liczba węzłów |
|---|---:|---:|---:|---:|---:|
| minimax | 2 | 200 | 99.0 | 0.008 | 10.30 |
| minimax | 3 | 200 | 99.5 | 0.019 | 28.67 |
| minimax | 4 | 200 | 99.0 | 0.047 | 73.01 |
| minimax | 5 | 200 | 99.0 | 0.111 | 169.20 |
| alpha_beta | 2 | 200 | 99.0 | 0.008 | 10.30 |
| alpha_beta | 3 | 200 | 99.5 | 0.018 | 25.69 |
| alpha_beta | 4 | 200 | 99.0 | 0.036 | 51.70 |
| alpha_beta | 5 | 200 | 99.0 | 0.069 | 95.85 |

Krótka interpretacja:

- Skuteczność obu wariantów jest praktycznie identyczna, co jest zgodne z teorią: obcinanie `alpha-beta` nie zmienia jakości decyzji, tylko ogranicza liczbę analizowanych stanów.
- Dla małej głębokości `d = 2` zysk jest pomijalny, bo drzewo gry jest jeszcze bardzo małe.
- Im większa głębokość, tym bardziej widoczna przewaga `alpha-beta`: dla `d = 5` średnia liczba odwiedzonych węzłów spadła z `169.20` do `95.85`, a średni czas ruchu z `0.111 ms` do `0.069 ms`.
- Otrzymane odsetki wygranych są bardzo wysokie, ponieważ algorytm gra przeciwko przeciwnikowi losowemu, a sama gra ma prostą strukturę strategiczną.
