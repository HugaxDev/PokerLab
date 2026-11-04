# PokerLab

Laboratoire local de simulation de poker (Texas Hold'em No Limit, 6 joueurs, 100 bb) pour savoir,
**avec des preuves statistiques**, quelle stratégie gagne sur la durée, pourquoi elle gagne et où
chaque stratégie perd de l'argent.

Point de départ : 6 phases de 50 mains jouées à la main (≈ 300 mains). Une session de 50 mains se
joue sur une ou deux grosses mains : elle ne prouve rien. PokerLab rejoue des centaines de milliers
de mains en éliminant la chance (duplicate, all-in EV, nombres aléatoires communs) et s'arrête quand
l'intervalle de confiance est assez étroit.

- Moteur de règles : **PokerKit 0.7.x** (actions légales, pots, side pots, showdown).
- 100 % local et déterministe : aucune API, aucun LLM, aucune requête réseau à l'exécution.
- Unité interne : la grosse blind (bb). Résultats en bb et bb/100.

---

## Installation

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

Dépendances : pokerkit 0.7.x, numpy, pandas, scipy, pyyaml, plotly (embarqué hors ligne dans le
rapport), pyarrow (Parquet), pytest. Python ≥ 3.11 (testé en 3.13).

La table d'équité préflop 169×169 est calculée au premier lancement (≈ 10 s) puis mise en cache
dans `cache/preflop_169.npz`. Pour la recalculer plus précisément :

```bash
python -m pokerlab precompute --sims 2000
```

## Démarrage rapide

Démonstration miniature de toutes les expériences (≈ 350 mains, quelques secondes) :

```bash
python -m pokerlab smoke
```

Rapport : `out/smoke/rapport.html` (un fichier autonome, ouvrable hors ligne, sur mobile aussi).
Chaque expérience a aussi son propre rapport dans `out/smoke/<expérience>/`, et une fiche de
démonstration est écrite dans `out/smoke/fiche_joueur.html`.

La vraie campagne (à lancer vous-même, plusieurs heures selon la machine) :

```bash
python -m pokerlab compare
```

## Commandes

Chaque expérience a une commande et une config YAML dans `configs/experiments/`.

| Commande | Ce qu'elle fait | Config |
|---|---|---|
| `compare` | Les 8 stratégies contre chaque composition, en duplicate, arrêt automatique. Matrice stratégie × composition, tests appariés, fuites. | `compare.yaml` |
| `round-robin` | Les stratégies les unes contre les autres à la même table (sous-ensembles de 6 parmi 8). | `round_robin.yaml` |
| `sessions` | 10 000 sessions de 50 mains par stratégie : dispersion, P(session en perte), pire session, drawdown médian. | `sessions.yaml` |
| `bankroll` | Risque de ruine et trajectoires pour un capital (défaut 3,50 en 0,01/0,02), avec et sans stop-loss (25 bb). | `bankroll.yaml` |
| `sweep` | Balayage d'un ou deux paramètres d'une stratégie, heatmap du bb/100. | `sweep.yaml` |
| `ablation` | Retire une règle à la fois pour mesurer ce qu'elle rapporte ou coûte. | `ablation.yaml` |
| `optimize` | Recherche de paramètres sur graines d'entraînement, validation sur graines jamais vues. | `optimize.yaml` |
| `audit` | Audit contrefactuel : EV perdue par décision, par rollouts de chaque alternative légale. | `audit.yaml` |
| `journal` | Import de votre journal CSV réel et comparaison avec la stratégie simulée. | `journal.yaml` |
| `smoke` | Toutes les expériences en miniature (≈ 350 mains) + rapport et fiche de démonstration. | `smoke.yaml` |
| `fiche` | Fiche HTML d'une stratégie : grilles préflop par position, tailles de mise, règles postflop, interdits (audit) et résultat attendu (compare). | `fiche.yaml` |
| `bench` | Mesure le débit réel (mains/s). | `bench.yaml` |
| `precompute` | Calcule la table d'équité préflop 169×169. | `game.yaml` |
| `replay` | Affiche l'historique texte d'une main de campagne. | — |
| `strategies` | Liste les stratégies disponibles. | — |

Options communes : `--config <fichier ou nom>`, `--out <dossier>`, `--workers N` (0 = tous les
cœurs), `--seed S`, `--set chemin=valeur` (surcharge de la config d'expérience),
`--game-set chemin=valeur` (surcharge de `configs/game.yaml`). Exemples :

```bash
python -m pokerlab compare --set autostop.ci_halfwidth_bb100=5 --set "compositions=[micro_play_money]"
```

```bash
python -m pokerlab sweep --set strategy=joueur --set "params=[{path: taille_cbet, values: [0.25, 0.33, 0.5]}]"
```

```bash
python -m pokerlab replay --strategy joueur --composition micro_play_money --deck 12 --rot 3
```

```bash
python -m pokerlab journal --csv journal/mon_journal.csv --phases journal/phases.yaml
```

```bash
python -m pokerlab fiche --strategy serre --campaign out
```

Sorties de chaque campagne (`out/<commande>/`) : `rapport.html`, `mains.csv` + `mains.parquet`
(une ligne par main), `decisions.csv` + `decisions.parquet` (une ligne par décision),
`historiques.txt` (format proche de PokerStars), `resume.json`.

## Architecture

```
pokerlab/
  engine/        boucle de jeu PokerKit (hand_runner), observation/actions (types), table (bouton,
                 tapis, recave), indicateurs par siège (flags), historiques texte (history)
  cards/         encodage des cartes, parseur de ranges, grille 13x13, catégories de mains
  equity/        évaluateur 7 cartes (scalaire + numpy), Monte Carlo, table préflop, all-in EV
  features/      lecture du tableau (sec/mouillé), de la main (niveau, tirages), de la situation
  strategies/    protocole Strategy, moteur à règles YAML, equity_bot, random_legal, modèle de range
  opponents/     archétypes paramétrables, compositions de table
  experiments/   duplicate + arrêt auto, compare, round_robin, sessions, bankroll, sweep/ablation/
                 optimize (variants), audit, smoke, bench
  analysis/      stats (IC, tests, Holm), stats de joueur, décomposition, détecteur de fuites
  journal/       import du journal CSV réel et rapport (dont les mains hors fiche)
  fiche/         fiche de stratégie : grilles préflop, tailles, règles postflop, blocs campagne
  report/        rapport HTML autonome, graphiques plotly, exports CSV/Parquet
  cli.py         python -m pokerlab ...
configs/         game.yaml, fiche.yaml, strategies/, opponents/, tables/, experiments/
journal/         exemple_journal.csv (inventé), phases.yaml
tests/
```

### Interface de stratégie

Toutes les stratégies (règles, robots, archétypes) implémentent le même protocole :

```python
class Strategy(Protocol):
    name: str
    def decide(self, observation: Observation) -> Action: ...
```

`Observation` contient tout ce que voit le joueur : cartes, tableau, street, position, tapis, mises
de la street, contributions, pot, actions légales (montants min/max), historique complet de la main,
nombre d'adversaires actifs, graine de décision. Un futur joueur LLM n'aura qu'à implémenter
`decide` (sérialiser l'observation, renvoyer `Action.fold()`, `Action.check_call()` ou
`Action.raise_to(montant)`) : le moteur n'a rien à changer. Le moteur corrige et compte toute action
illégale (ex. se coucher sans mise à payer devient check).

## Les stratégies

Chaque stratégie est une config YAML lue par un moteur commun (`strategies/rule_engine.py`), pas un
fichier de code par stratégie. `extends:` permet l'héritage (zero_bluff hérite de large, bluff de
position).

| Stratégie | Résumé |
|---|---|
| `serre` | 99+, AK, AQ, AJs, KQs. Ouvre 3 bb, 3-bet 3× QQ+/AK, paie ≤ 4 bb avec 99-JJ/AQ. Postflop : mise 2/3 pot avec top paire+, sinon check, se couche face à une mise sans top paire ; face à un tapis ou une mise > 50 % du pot, ne suit qu'avec deux paires+ ou une overpair QQ+. |
| `large` | Paires, as, deux cartes ≥ T, assorties connectées 54s+. Ouvre 2 bb, paie ≤ 3 bb. Postflop : serré + paie les tirages couleur/quinte ouverte si la cote le justifie. |
| `zero_bluff` | Préflop de large. Postflop : ne mise/relance jamais sans paire ; paie un tirage seulement avec la cote. |
| `position` | BTN/CO : range large, ouvre 3 bb. Ailleurs : 99+, AK, AQ, AJ. Pas de bluff. BB : check gratuit, couché sans main de la range serrée. |
| `bluff` | Position + un bluff 1/2 pot au flop si relanceur préflop, un seul adversaire, flop sec. Un seul bluff par main. |
| `joueur` | ≈ 35-40 % des mains. Ouvre 3 bb, paie ≤ 4/5 bb selon la main. Postflop : mise 1/3-1/2 avec paire ou tirage, c-bet 1/3, garde-fou face aux grosses mises. |
| `equity_bot` | Référence « bon joueur » : équité Monte Carlo contre la range adverse estimée, cote du pot, tailles standard. |
| `random_legal` | Témoin : actions légales au hasard (doit perdre nettement). |

Les couleurs de chaque stratégie sont fixées dans sa config et restent les mêmes dans tout le
rapport (palette catégorielle validée pour le daltonisme).

## Adversaires et compositions

Archétypes (`configs/opponents/archetypes.yaml`) : `calling_station`, `nit`, `TAG`, `LAG`,
`maniac`, `nit_trappeur` (slowplay des grosses mains puis check-relance), `equity_bot`. Paramètres :
VPIP, PFR, 3-bet, agressivité, fréquence de c-bet, fold to c-bet, bluff, semi-bluff, paie faible,
float, relance de value, check-relance, slowplay, taille de mise, seuil de tapis.

Compositions (`configs/tables/`) : `micro_play_money` (défaut : beaucoup de calling stations et de
maniacs, peu de nits), `reguliers` (TAG/LAG), `mixte`. Chaque deck tire ses 5 adversaires selon les
poids, avec une graine fixe : toutes les stratégies affrontent exactement les mêmes adversaires sur
les mêmes cartes.

Le rapport `compare` affiche les stats **mesurées** des archétypes à côté de leurs cibles.

## Réduction de la variance

- **Duplicate** : les cartes sont liées aux sièges (le siège PokerKit `i` reçoit `deck[2i], deck[2i+1]`,
  le tableau est `deck[12:17]`). Chaque deck est rejoué 6 fois, la stratégie testée occupant chaque
  siège, les adversaires gardant le même ordre à sa gauche. L'unité statistique est le deck (moyenne
  des 6 rotations) : la chance des cartes s'annule.
- **All-in EV** : quand l'argent est à tapis avant la river, le résultat est remplacé par
  l'espérance (équité × pot, side pots compris ; énumération exacte au flop et au turn, Monte Carlo
  préflop). Réel et ajusté sont publiés côte à côte.
- **Nombres aléatoires communs** : mêmes graines de deck pour toutes les stratégies comparées ; les
  différences sont appariées deck par deck.
- **Arrêt automatique** : la campagne avance par blocs (défaut 500 decks = 3 000 mains par cellule)
  jusqu'à ce que la demi-largeur de l'IC 95 % du bb/100 soit ≤ 3 bb/100, avec un maximum de
  200 000 mains par cellule (stratégie × composition).

## Métriques (une phrase chacune)

- **bb/100** : grosses blinds gagnées en moyenne pour 100 mains.
- **all-in adjusted** : bb/100 où les tapis avant la river comptent pour leur espérance et non pour le tableau tombé.
- **IC 95 % (±)** : la vraie valeur a 95 % de chances d'être dans cet intervalle (calculé sur les moyennes par deck).
- **σ brut (bb/100)** : écart type d'une série de 100 mains sans réduction de variance (σ par main × 10).
- **σ effectif** : écart type équivalent après duplicate et all-in EV (plus petit = moins de mains nécessaires).
- **mains pour être significatif** : nombre de mains pour que l'IC 95 % exclue 0 au taux observé.
- **test apparié** : test t (ou bootstrap par blocs) sur les différences par deck entre deux stratégies.
- **correction de Holm** : ajuste les p-valeurs quand on fait beaucoup de comparaisons à la fois ; « significatif » = p Holm < 0,05.
- **VPIP** : % de mains où le joueur met de l'argent volontairement préflop.
- **PFR** : % de mains où le joueur relance préflop.
- **3-bet** : % de relances face à une seule relance, quand l'occasion se présente.
- **entrée / relance en pot non ouvert** : % d'entrée (ou de relance) quand personne n'a encore relancé ; c'est la mesure comparable aux cibles des archétypes.
- **AF** : agressivité postflop = (mises + relances) / paies.
- **WTSD** : % des flops vus qui vont jusqu'à l'abattage.
- **W$SD** : % des abattages gagnés.
- **c-bet** : % de mises au flop quand on était le dernier relanceur préflop et que personne n'a misé avant.
- **fold to c-bet** : % d'abandons face à un c-bet.
- **décomposition par position / street / catégorie de main** : d'où viennent les gains et les pertes.
- **grille 13×13** : bb gagnées par main pour chaque combinaison de départ (assorties au-dessus de la diagonale).
- **avec / sans showdown (bleu / rouge)** : gains des mains allées à l'abattage contre gains des mains gagnées ou perdues avant.
- **value bet / bluff / semi-bluff / paie** : ligne jouée selon la dernière action volontaire postflop et la force de la main à ce moment.
- **drawdown max** : plus grosse baisse depuis un sommet de la courbe de gains (en bb).
- **plus longue série perdante** : plus long passage (en mains) sous le dernier sommet de la courbe.
- **EV après décision (fuites)** : gain moyen à partir d'une décision ; négatif = abandonner à ce moment aurait coûté moins cher.
- **EV perdue (audit)** : EV de la meilleure alternative légale moins EV de l'action prise, estimée par rollouts.
- **P(session en perte)** : probabilité qu'une session de 50 mains finisse négative.
- **risque de ruine** : probabilité que la bankroll tombe sous 1 bb sur l'horizon simulé.
- **écart entraînement / validation** : sur-estimation due au sur-ajustement lors de l'optimisation.

## Lire le rapport

Ordre : verdict en 5 lignes, tableau comparatif (la meilleure valeur de chaque ligne prend la couleur
de sa stratégie), graphiques, statistiques mesurées, fuites (avec 3 mains d'exemple rejouées), méthode.
Survol des courbes : main, position, cartes, ligne jouée, montant, cumul. Bouton « clair / sombre »
en haut à droite. Chaque graphique a une table de lecture ou un tableau équivalent.

## Fiche de stratégie

```bash
python -m pokerlab fiche --strategy <nom> [--campaign <dossier>] [--composition <nom>] [--out <fichier.html>]
```

Une page HTML autonome (aucune dépendance, ≈ 60 Ko), en français, lisible sur téléphone, à garder
ouverte pendant que tu joues :

1. **Avant le flop** : une grille 13×13 par position (UTG, MP, CO, BTN, SB, BB), « personne n'a
   relancé » et « face à une relance » : relancer (bleu), payer ou check gratuit (jaune), se coucher
   (gris), avec le % de mains relancées et payées. Les grilles sont obtenues en interrogeant la
   stratégie elle-même dans chaque situation : elles montrent exactement ce que joue le moteur.
2. **Tailles de mise** en bb et en jetons pour les blinds 0,01/0,02 et 0,02/0,05.
3. **Après le flop** : les règles en 10 lignes maximum, générées depuis la config.
4. **Interdits** : les 5 fuites les plus chères de l'audit, formulées comme des interdits
   (« Ne paie pas avec top paire face à une relance au turn en CO : couche-toi »), avec leur coût et
   la mention « à confirmer » si l'échantillon est trop petit.
5. **Résultat attendu** : bb/100 (all-in adjusted, IC 95 %), réel, écart type, et fourchette
   5e-95e centile du résultat réel sur 50, 200 et 1 000 mains.

Les blocs 4 et 5 viennent d'une campagne : `--campaign` pointe vers un dossier qui contient (lui
ou ses sous-dossiers) les sorties de `compare` et d'`audit` (ex. `out` ou `out/smoke`). Sans
campagne, ou si la stratégie n'y figure pas, ces blocs disent « lance compare puis audit » au lieu
d'afficher des chiffres. Pour une fiche complète :

```bash
python -m pokerlab compare
```

```bash
python -m pokerlab audit --set strategy=serre
```

```bash
python -m pokerlab fiche --strategy serre --campaign out
```

Le rapport `journal` utilise ces mêmes grilles pour signaler les **mains jouées hors de la fiche**.

## Mon journal réel

Format CSV (virgule ou point-virgule, lignes commençant par `#` ignorées) :

| Colonne | Contenu |
|---|---|
| `phase` | identifiant de phase (doit exister dans `phases.yaml`) |
| `main` | numéro de main dans la phase |
| `cartes` | `AK`, `T9`, ou cartes exactes `AsKd` |
| `assorties` | `oui` / `non` (ignoré si les couleurs sont données) |
| `position` | `BTN`, `CO`, `MP`, `UTG`, `SB`, `BB` |
| `mise_totale` | montant total mis dans le pot pendant la main (monnaie de la table) |
| `board` | tableau vu (`Kd7c2s`), vide si fini préflop |
| `action_resume` | résumé : `pre: relance 3bb | flop: mise 2/3 | turn: couché`. Mots-clés préflop reconnus : relance, ouvre, 3bet, tapis, mise (= relance) ; paie, suit, call, limp ; couché, check. « couché sur relance » n'est pas une relance. Une recave s'écrit `recave 2.00`. |
| `tapis_apres` | tapis après la main (avant toute recave) |
| `estime` | `oui` si les montants sont approximatifs |
| `hors_regle` | `oui` si la main a été jouée en dehors des règles de la stratégie |

`journal/phases.yaml` décrit chaque phase : stratégie suivie, blinds, tapis de départ. Le résultat
d'une main = variation de tapis (moins les recaves), converti en bb.

Le rapport calcule vos vraies stats (VPIP, PFR, résultats par position et catégorie de main, mains
hors règle), liste les mains jouées hors de la fiche de la stratégie de la phase, et place votre résultat dans la distribution simulée de la même stratégie sur le même
nombre de mains : « 18e centile, dans la distribution normale » ou « sous le 5e centile ».

`journal/exemple_journal.csv` contient **10 mains inventées** pour montrer le format.

## Performance

Mesure réelle (`python -m pokerlab bench`, 1 284 mains au total) dans le conteneur de développement :
Linux x86_64, 4 cœurs, Python 3.13, PokerKit 0.7.7.

| Mesure | Débit |
|---|---|
| Stratégies à règles, 1 processus | ≈ 670 mains/s |
| Stratégies à règles, 4 workers (après démarrage des workers) | ≈ 2 340 mains/s |
| equity_bot (Monte Carlo postflop), 1 processus | ≈ 540 mains/s (petit échantillon de 60 mains) |

Le coût principal est PokerKit (vérifications et évaluation des mains au showdown). L'évaluation
PokerKit est mémoïsée (même ensemble de 7 cartes = même main) : résultats identiques, ≈ 3× plus
rapide ; un test le vérifie. Relancez `bench` sur votre machine : les chiffres dépendent du processeur.

Ordre de grandeur pour `compare` complet (8 stratégies × 3 compositions, jusqu'à 200 000 mains par
cellule) : jusqu'à 4,8 M mains, soit environ 35 min à 2 300 mains/s, moins si l'arrêt automatique
déclenche avant. Estimation pessimiste : comptez 1 h.

## Interprétations

Les règles décrites en français laissent des ambiguïtés. Elles sont tranchées dans les YAML
(commentaires) et listées ici :

1. **Positions 6-max** : SB, BB, UTG, MP, CO, BTN (index PokerKit 0 à 5, le bouton en dernier).
2. **« Paire »** = paire utilisant au moins une carte privée (une paire du tableau seule ne compte pas).
   Top paire = une carte privée appariée à la plus haute carte du tableau ; overpair = paire servie
   au-dessus du tableau (comptée comme « top paire ou mieux ») ; deux paires = les deux cartes privées
   appariées au tableau ; une paire servie + paire au tableau reste une paire.
3. **Limpers** : une ouverture ajoute 1 bb par limper (toutes les stratégies à règles).
4. **serre** : face à un 3-bet après avoir ouvert, tapis avec QQ+/AK, sinon couché ; face à un 4-bet,
   paie avec QQ+/AK. Face à une mise postflop ≤ 50 % du pot : paie avec top paire ou mieux, relance 3×
   avec deux paires ou mieux. **Garde-fou** : face à un tapis ou à une mise/relance > 50 % du pot
   (même lecture que pour joueur : toute mise ou relance, pas seulement les relances), ne suit qu'avec
   deux paires ou mieux, ou une overpair QQ+ (QQ, KK, AA au-dessus du tableau) ; jamais d'exception
   à l'équité. `trigger_scope: raise_only` dans `serre.yaml` limite le garde-fou aux seules relances.
5. **large** : 3-bet QQ+/AK comme serré (non précisé) ; paie ≤ 3 bb avec toute sa range. « Tirage » =
   couleur ou quinte ouverte (double ventrale incluse). « Si la cote le justifie » : équité par les
   outs (1 carte ; 2 si la paie met à tapis au flop) ≥ à payer / (pot + à payer), sans cote implicite
   (`implied_odds_factor: 0`, réglable).
6. **zero_bluff** : la différence avec large est qu'elle mise (2/3 pot) avec **toute** paire ; elle ne
   semi-bluffe jamais ; elle paie une mise avec top paire ou mieux comme large.
7. **position** : au BTN/CO, paie ≤ 3 bb avec la range large (et ≤ 4 bb avec 99-JJ, AQ, AJ) ;
   ailleurs et en BB, paie ≤ 4 bb avec 99-JJ, AQ, AJ, 3-bet QQ+/AK. Postflop identique à serré **sans
   son garde-fou** (ajouté à serré seulement ; bluff, qui hérite de position, ne l'a pas non plus).
8. **bluff** : flop sec = aucune paire de couleur, pas 3 rangs distincts dans un écart ≤ 4 (As haut ou
   bas), pas de paire « mouillée » (paire au tableau avec une autre carte à ≤ 2 rangs). Le bluff
   n'existe qu'au flop, ce qui garantit un seul bluff par main ; payé ou relancé sans main, la
   stratégie abandonne via les règles normales. `frequency` (défaut 1) sert au sweep.
9. **joueur** : range = paires, as, deux cartes ≥ T, rois assortis, connecteurs assortis 32s-T9s,
   un trou J9s, T8s-42s (≈ 35 % des combinaisons ; « environ 40 % » est atteint en comptant la
   défense de BB). Face à une relance ≤ 4 bb : paie avec tout roi, tout as, toute paire, **toute main
   assortie** (règle littérale) ; ≤ 5 bb avec deux cartes ≥ T. 3-bet QQ+/AK. Face à un 3-bet : tapis
   QQ+/AK, paie ≤ 12 bb avec TT-JJ, AQ, AJs, KQs. « 1/3 à 1/2 du pot » : 1/2 avec top paire ou mieux,
   1/3 avec une paire plus faible ou un tirage. Face à une mise ≤ 50 % du pot : paie avec une paire
   ou un tirage ; au-delà ou face à un tapis : garde-fou (deux paires+, sinon équité Monte Carlo
   contre la range estimée ≥ cote). C-bet au flop seulement.
10. **Relance qui engage ≥ 70 % du tapis** : transformée en tapis (`shove_commit_fraction`).
11. **Mode PokerKit** : « tournoi » (un seul déroulé du tableau en cas de tapis, cartes montrées) ;
    sans incidence sur les gains d'une partie cash sans rake.
12. **Tapis** : en duplicate, chaque main repart à 100 bb (sinon les rotations ne rejouent pas la même
    partie). Les sessions utilisent des tapis persistants avec recave à 100 bb (top-up).
13. **Maximum de mains** « par stratégie » : appliqué par cellule (stratégie × composition).
14. **Value bet** : mise avec paire moyenne ou mieux (`analysis.value_min_level`) ; en dessous, semi-bluff
    avec tirage, bluff sinon.
15. **Range adverse estimée** : percentiles de mains selon la ligne préflop (ouverture par position,
    3-bet, paie, limp), puis à chaque street une mise garde les 60 % les plus forts, une relance 35 %,
    une paie retire les 20 % les plus faibles (`range_model` dans le YAML).
16. **equity_bot** : ouvre un pourcentage de mains par position, 3-bet si équité ≥ 60 % contre la
    range de l'ouvreur, mise si équité ≥ part équitable + 25 % du reste, paie si équité × réalisation
    ≥ cote. C'est un étalon simple, pas un solveur.
17. **Fuites** : seules les actions actives (paie, mise, relance) sont évaluées, sur la composition
    principale ; abandonner = 0 bb à partir de la décision.
18. **Audit** : les cartes réelles des adversaires restent celles de la main, les cartes à venir sont
    retirées au hasard, l'aléa des adversaires est resalé ; EV perdue estimée par validation croisée
    (moitié des rollouts pour choisir, l'autre pour mesurer). Les abandons préflop avec une main hors
    du top 50 % sont exclus de l'échantillon (`nontrivial_preflop_pct`).
19. **Bankroll** : sessions tirées avec remise ; quand la bankroll passe sous 100 bb, la perte d'une
    session est plafonnée à ce qui reste.
20. **Journal** : VPIP/PFR déduits des mots-clés du résumé préflop, à défaut de `mise_totale`.
21. **Fiche** : « face à une relance » = une relance à 3 bb juste devant (du BTN quand tu es en blinds),
    les autres couchés ; en BB, « personne n'a relancé » = la SB a complété. Les plafonds de paie
    (« jusqu'à 4 bb ») sont dans le tableau des tailles. Le résultat attendu utilise la composition
    la plus jouée de la campagne pour la stratégie (`--composition` pour en choisir une autre) ; les
    fourchettes tirent au hasard des mains simulées (résultats réels, avec remise).
22. **Mains hors fiche (journal)** : la situation est déduite du résumé préflop (« paie », « 3bet »,
    « couché sur relance »… = face à une relance ; sinon personne n'a relancé), l'action jouée aussi
    (relance / paie / check / couché). Une main est hors fiche si cette action diffère de la grille.
    Un « couché » sans précision est compté comme « personne n'a relancé » : écrivez « couché sur
    relance » pour lever l'ambiguïté. UTG face à une relance n'est pas évalué.

## Limites connues

- Les archétypes ne sont pas de vrais humains : pas d'adaptation, pas de lecture, pas de tilt. Leurs
  VPIP/PFR globaux dépendent de la table (face à beaucoup de relances, ils jouent moins de pots non
  ouverts) ; leurs cibles sont vérifiées en pot non ouvert.
- Les résultats dépendent fortement de la composition : une stratégie gagnante contre des calling
  stations peut perdre contre des réguliers. Regardez la matrice, pas un seul chiffre.
- Profondeur fixe 100 bb, pas de rake (jetons fictifs). Les sessions simulent une petite blind à 0,5 bb :
  pour une phase en 0,02/0,05, lancez avec `--game-set game.small_blind_bb=0.4`.
- equity_bot et le modèle de range sont volontairement simples ; l'équité préflop multiway est
  approchée (équité contre le relanceur × facteur multiway).
- L'audit à peu de rollouts est bruité ; la version smoke ne sert qu'à vérifier le pipeline.
- Les historiques texte arrondissent les montants au centime (les résultats en bb sont exacts).
- Les chiffres du smoke ne sont pas significatifs (quelques dizaines de mains par stratégie).
- Déterminisme : même graine + mêmes versions de Python/numpy/PokerKit = mêmes résultats, quel que
  soit le nombre de workers. Multiprocessing en `forkserver` (Linux) ; mettre `spawn` sous Windows/macOS
  (`mp_start_method` dans `configs/game.yaml`).

## Tests

```bash
pytest -q
```

Couvre : conservation des jetons (side pots, tapis inégaux, recaves), 10 000 décisions aléatoires
uniquement légales, parseur de ranges (cas limites, 1 326 combinaisons, 6/4/12 par classe),
évaluateur identique à PokerKit, équité AA vs KK ≈ 81-82 %, reproductibilité (même graine, 1 ou N
workers), duplicate (somme nulle quand tous les sièges jouent la même stratégie), règles de chaque
stratégie (situations construites et propriétés sur mains jouées), random_legal perd nettement contre
equity_bot, journal d'exemple, statistiques et correction de Holm, garde-fou de serré, fiche (grilles,
blocs avec et sans campagne sur données synthétiques), mains hors fiche du journal. La suite joue
environ 1 000 mains (compteur affiché en fin de `pytest`).
