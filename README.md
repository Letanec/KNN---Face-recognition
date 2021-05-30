# KNN Face-recognition
Cílem tohoto projektu je natrénování neuronové sítě pro verifikaci osob podle obličeje s nasazenou rouškou. 
Projekt vypracovali xnekut00, xplatan00 a xplach08 pro předmět KNN v roce 2021.
## Spuštění
Skript pro trénování a evaluaci spustíte příkazem `python main.py` z adresáře src. Běh programu lze ovlivnit následujícími argumenty:
* `-b` nebo `--batch_size` následované kladným celým číslem pro zadání velikosti batche (defaultně 52)
* `-e` nebo `--epochs_num` následované kladným celým číslem pro zadání počtu epoch trénování (defaultně 10)
* `-m` nebo `--model_path` následované cestou k uloženým váhám pro načtení dříve natrénovaných vah do modelu
* `-v` nebo `--evaluate` pro evaluaci na datasetu LFW
* `-f` nebo `--far` následované maximální povolenou hodnotou FAR (defaultně 0.001) pro evaluaci (má efekt pouze ve spojení s argumentem `--evaluate`)
* `-a` nebo `--arcface` pro použití loss funkce ArcFace (bez tohoto argumentu se defaultně použije Cross Entropy)
* `-r` nebo `--plot_roc` pro spuštění evaluace na LFW a následné vytisknutí křivky ROC 
* `-s` nebo `--visualize_embedings` pro vizualizaci rozmístění embedingů v prostoru (pro dataset CASIA)

### Závislosti
Pro spuštění programu je třeba Python verze 3.6 a novější s nainstalovanými balíčky torch, torchvision, facenet_pytorch, numpy, scipy, matplotlib a sklearn.
Tyto balíčky lze nainstalovat příkazem 
`pip install torch torchvision facenet_pytorch numpy scipy matplotlib sklearn`

## Výsledný model
Pokud budete chtít použít námi natrénovaný model, stačí spustit skript příkazem `python main.py -m ../models/final_model`.

## Datasety
Pokud byste chtěl spustit trénování a evaluaci na plnohodnotném datasetu, můžete si stáhnout datasety lfw (http://vis-www.cs.umass.edu/lfw/) a casia (https://drive.google.com/file/d/1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz/view), umístit je do složky `datasets`. Kořenové složky datasetů se musejí jmenovat `datasets/lfw` a `datasets/casia_orig` a musejí krom podsložek s fotkami obsahovat soubory `pairs.txt` pro LFW a `pairs.txt` pro CASIA, které lze také stáhnout z uvedených odkazů. Nasazení masek na obličeje v datasetech proběhne spuštěním příkazu `python add_masks.py`.

## Převzatý kód
Pro automatické nasazení masek na obličeje v datasetech jsme použili knihovnu Dlib dostupnou na https://github.com/davisking/dlib.
Pro evaluaci modelu na datasetu LFW jsme použili kód z https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb.