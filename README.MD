Zaphod is just a hobby!

Maybe at some point it will get to a ELO 2000+, but not today





| Version	| Estimated ELO		| Change log |
| -------	| -------------		| ---------- |
| 1.9		| ~1850				| Optimized make move. <br> Made tt thread safe <br> Introduced killer move in move ordering |
| 1.8		| ~1790				| Reintroduced move ordering in qsearch +60 Elo. <br> Added compiler flags for optimization +60 Elo.|
| 1.7		| ~1670				| Improved calculating PSQT score. Est Elo gain +20 |
| 1.6		| ~1650				| Removed pawn and piece pair evaluation. Only psqt and material left. Est ELO gain +30|
| 1.5		| ~1620				| Adde LMR. - Est ELO gain +60 |
| 1.4		| ~1560				| Check extension in search. Est Elo gain +40. <br><br> Testing: <br> TC 10+0.1 and 120+1, 200 games per engine per tc. <br> Reference engines <br> - Pulse 1.7.2 64-bit ELO 1518 <br> - Cicada 0.1 64-bit ELO 1443 <br> - Zaphod 1.3.1 ELO ~1520 |
| 1.3.1		| ~1520				| Limited quiescence depth from 100 to 10. Est ELO gain +50 |
| 1.3		| ~1470				| - Fixed bug in mate check ( Most of the elo increase comes from this ) <br>- Implemented custom hashtable for TT, configurable with specifying MB (not exposed to UCI) <br> - Pawn hash table <br> - Pair evaluation of minor pieces ( No significant gain )<br> <br> Testing: <br>TC: 10+0.1 <br> 2000 games per engine <br> Reference engines: <br>- Pulse 1.7.2 64-bit ELO 1518 <br>- Cicada 0.1 64-bit ELO 1443 <br> - Zaphod 1.2 ELO 1366
| 1.2		| ~1360				|  - If negamax does not return a move, pick one from the movelist instead of doing an invalid move <br> - Added passed pawn scoring (Low score, max 30)
| 1.1		| ~1360				|- Fixed error in promotion <br>- Added half move clock
| 1.0		| ~1350				| The first functional version which can play games, with a certain confidence it will not do very stupid things.


Features:
- Magic bitboards
- Negamax search
- Check extension
- LMR
- quinesence for checks, captures and promotions
- Basic time control for playing games
- Transposition table
- Move ordering, PV from TTable, captures, MVV-LVA
- Piece square table tapered between mg and eg
- Material count in evaluation
- Decent fast legal move generation, 100M+ NPS using bulk counting at leaf node

