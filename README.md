# KNN Identifikace osob podle obličeje
Cílem tohoto projektu je natrénování neuronové sítě pro verifikaci osob podle obličeje s nasazenou rouškou. 
Projekt vypracovali xnekut00, xplatan00 a xplach08 pro předmět KNN v roce 2021.
## Spuštění
Skript pro trénování a evaluaci spustíte příkazem `python main.py` z adresáře `src`. Běh programu lze ovlivnit následujícími argumenty:
* `-b` nebo `--batch_size` následované kladným celým číslem pro zadání velikosti batche (defaultně 52)
* `-e` nebo `--epochs_num` následované kladným celým číslem pro zadání počtu epoch trénování (defaultně 10)
* `-m` nebo `--model_path` následované cestou k uloženému modelu pro načtení dříve natrénovaných vah
* `-f` nebo `--far` následované maximální povolenou hodnotou FAR (defaultně 0.001) pro evaluaci (má efekt pouze ve spojení s argumentem `-v`)
* `-a` nebo `--arcface` pro použití loss funkce ArcFace (bez tohoto argumentu se defaultně použije Cross Entropy)
* `-v` nebo `--evaluate` pro evaluaci na datasetu LFW
* `-f` nebo `--far` následované maximální povolenou hodnotou FAR (defaultně 0.001) pro evaluaci (má efekt pouze ve spojení s argumentem `--evaluate`)
* `-a` nebo `--arcface` pro použití loss funkce ArcFace (bez tohoto argumentu se defaultně použije Cross Entropy)
* `-r` nebo `--plot_roc` pro spuštění evaluace na LFW a následné vytisknutí křivky ROC 
* `-s` nebo `--visualize_embedings` pro vizualizaci rozmístění embedingů v prostoru (pro dataset CASIA)

Parametry po evaluacy, vykreslení ROC křivky a vizualizaci embedingů jsou exkluzivní. Spuštění s více než jedním z těchto parametrů nevede na chybu, ale provede se jen jeden z nich (s tím, že jsou seřazené s následující prioritou: `--visualize_embedings` > `--evaluate` > `--plot_roc`).

### Závislosti
Pro spuštění programu je třeba Python verze 3.6 a novější s nainstalovanými balíčky torch, torchvision, facenet_pytorch, numpy, scipy, matplotlib a sklearn.
Tyto balíčky lze nainstalovat příkazem 
`pip install torch torchvision facenet_pytorch numpy scipy matplotlib sklearn`

## Výsledný model
Pokud budete chtít použít námi natrénované modely, stačí spustit skript příkazem `python main.py -m ../models/final_model -a` (pozor - pro načtení model natrénovaného na ArcFace je nutné uvést argument `-a`). Námi natrénované modely jsou dostupné na https://drive.google.com/drive/folders/1zD6XL9Tu2tihpTlxOaHFqEB1yX97FNuT?usp=sharing.

## Datasety
Pokud byste chtěl spustit trénování a evaluaci na plnohodnotném datasetu, můžete si stáhnout datasety lfw (http://vis-www.cs.umass.edu/lfw/) a casia (https://drive.google.com/file/d/1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz/view), umístit je do složky `datasets`. Kořenové adresáře datasetů se musejí jmenovat `datasets/lfw` a `datasets/casia`. V případě LFW musí adresář krom podadresářů s fotkami obsahovat soubor `pairs.txt`, který lze také stáhnout z uvedeného odkazu.
### Přidání roušek na obrázky v datasetu
Přidání roušek na obličeje lze provést spuštěním `python add_masks.py`. Skript naplní adresář `datasets/casia_with_masks` upravenými obrázky, obdobně `datasets/lfw_with_masks` včetně `datasets/lfw_with_masks/pairs_with_masks.txt`. Před použitím je třeba stáhnout a rozbalit konfiguraci modelu Dlib, a to spuštěním příkazu `wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2` v adresáři `src`.

## Převzatý kód
Pro detekci obličejových bodů k automatickému mapování roušek na obličeje jsme použili knihovnu Dlib dostupnou na https://github.com/davisking/dlib.
Pro evaluaci modelu na datasetu LFW jsme použili kód z https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb.