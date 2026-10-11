# HW1 : Introduction and Learnability

**PAC et PAC agnostique : étude expérimentale par ERM sur des classificateurs à seuil**

Cours *Statistical and Machine Learning*, Master 2 IMDS, Université de Haute-Alsace (UHA).
Auteur : Kossi NOUMAGNO.

Ce dépôt contient le notebook rendu pour le devoir HW1. Il compare expérimentalement le cadre PAC (cas réalisable) et le cadre PAC agnostique (étiquettes bruitées) avec une ERM écrite à la main, sans bibliothèque de machine learning. Les résultats détaillés, les tableaux et les figures se trouvent dans le notebook `HW1_PAC_Agnostic_ERM.ipynb`.

## Problème étudié

- Espace d'entrée : X = [0, 1], avec x tiré uniformément.
- Fonction cible : seuil f_θ(x) = 1 si x > θ, 0 sinon.
- Classe d'hypothèses : les seuils h_t(x) = 1 si x > t, pour t dans [0, 1].
- Cas réalisable : y = f_θ(x).
- Cas bruité : chaque étiquette est inversée indépendamment avec probabilité η.

Le notebook distingue deux quantités :

- l'erreur par rapport à la cible, `L_cible = |t - θ|` ;
- le risque sur les étiquettes observées, `L_D = η + (1 - 2η)|t - θ|`, qui ne peut pas descendre sous η, même quand le seuil appris est exactement θ.

## Contenu du notebook

1. Cadre mathématique
2. Imports et paramètres expérimentaux
3. Génération des données et visualisation
4. Implémentation de l'ERM
5. Définition des risques et fonction `evaluate_once`
6. Un essai par taille d'échantillon (illustratif)
7. Vérification expérimentale de PAC par répétitions multiples
8. Comparaison avec la formule théorique du bruit symétrique
9. L'erreur empirique peut-elle être nulle avec du bruit ?
10. Limites de l'expérience
11. Conclusion PAC et PAC agnostique
12. Export des résultats en CSV

## Implémentation de l'ERM

Pour un seuil, les prédictions ne changent que lorsque t traverse une valeur observée. Il suffit donc de tester les milieux entre valeurs distinctes de X, plus les bornes 0 et 1. L'erreur empirique de chaque candidat est calculée, puis `np.argmin` retient le meilleur. En cas d'égalité, le premier candidat (le plus petit seuil) est choisi : la règle de départage est déterministe.

## Paramètres choisis

- θ = 0.60 (seuil cible)
- ε = 0.10 (précision)
- δ = 0.05 (valeur de référence pour la comparaison)
- η ∈ {0, 0.10, 0.20}, où η = 0 correspond au cas réalisable
- m ∈ {5, 10, 20, 50, 100, 200} (tailles imposées par le sujet)
- 300 répétitions par couple (η, m) ; passer à 500 ou 1000 pour l'analyse finale
- jeu de test indépendant de 20 000 points
- graine 2026, pour des résultats reproductibles

## Ce que montrent les expériences

- **Cas réalisable** : l'erreur empirique est toujours nulle, et la probabilité d'échec P(L_cible > ε) diminue rapidement avec m.
- **Cas bruité** : l'erreur empirique n'est plus nulle en moyenne. Elle se rapproche de η quand m augmente, et reste inférieure au risque observé car l'ERM s'ajuste en partie au bruit de l'échantillon.
- **Seuil appris** : il se rapproche de θ malgré le bruit, mais plus lentement quand η est grand, donc il faut plus de données pour que la probabilité d'échec passe sous δ.
- **Risque observé** : il ne tend pas vers 0 mais vers η. C'est pourquoi le critère d'échec utilisé est l'erreur par rapport à la cible et non le risque sur les étiquettes bruitées.

## Exécution

Python 3 avec NumPy, pandas, Matplotlib et Jupyter.

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook HW1_PAC_Agnostic_ERM.ipynb
```

L'exécution complète génère le fichier `hw1_pac_agnostic_results.csv` avec les résultats de l'expérience répétée. Les paramètres se règlent dans la cellule de la section 2 (`THETA`, `EPSILON`, `DELTA`, `NOISE_RATES`, `N_REPETITIONS`, `N_TEST`, `SEED`).

## Limites

- Les fréquences sont des estimations Monte-Carlo, pas des probabilités exactes. Avec 300 répétitions, les valeurs proches de 0 sont incertaines.
- Le jeu de test est fini : le risque observé est estimé avec une petite erreur, alors que l'erreur cible `|t - θ|` est connue exactement.
- Les résultats dépendent de θ, ε, η, du nombre de répétitions et de la taille du test.
- Le bruit symétrique indépendant est un cas particulier du cadre PAC agnostique, et ne représente pas toutes les distributions agnostiques.
- δ est une valeur de comparaison : aucune borne théorique précise (avec ses constantes) n'est appliquée dans ce notebook.

## Structure du dépôt

```
.
├── HW1_PAC_Agnostic_ERM.ipynb
├── hw1_pac_agnostic_results.csv
└── README.md
```
