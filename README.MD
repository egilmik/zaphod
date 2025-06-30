Zaphod is just a hobby!

Maybe at some point it will get to a ELO 2000+, but not today

### Version 1.6

Estimated ELO ~1650

Changelog:
- Removed pawn and piece pair evaluation.  Only PSQT and material - Estimated 30+ ELO gain

### Version 1.5

Estimated ELO ~1620

Change log:
- Added LMR - basic implementation - Estimated +60 ELO gain 

### Version 1.4

Estimated ELO ~1560

Change log:
- Check extension in search (not in quiescence) - Estimated +40 ELO gain

Testing:
TC: 10+0.1 and 120+1
200 games per engine per time
Reference engines:
- Pulse 1.7.2 64-bit ELO 1518
- Cicada 0.1 64-bit ELO 1443
- Zaphod 1.3.1 ELO 1520

### Version 1.3.1

Estimated ELO ~1520

Change log:
- Limited quiescence depth to 10 - Estimated +50 ELO gain 

### Version 1.3

Estimated ELO ~1470

Testing:
TC: 10+0.1.
2000 games per engine
Reference engines:
- Pulse 1.7.2 64-bit ELO 1518
- Cicada 0.1 64-bit ELO 1443
- Zaphod 1.2 ELO 1366

Change log
- Fixed bug in mate check ( Most of the elo increase comes from this )
- Implemented custom hashtable for TT, configurable with specifying MB (not exposed to UCI)
- Pawn hash table
- Pair evaluation of minor pieces ( No significant gain )

### Version 1.2

Estimated ELO ~1366

Changelog:
- If negamax does not return a move, pick one from the movelist instead of doing an invalid move
- Added passed pawn scoring (Low score, max 30)

### Version 1.1

Estimated ELO ~1360?

Change log:
- Fixed error in promotion
- Added half move clock

### Version 1.0

The first functional version which can play games, with a certain confidence it will not do very stupid things.

Estimated ELO ~1350?

Features:
- Magic bitboards
- Negamax search
- quinesence for checks, captures and promotions
- Basic time control for playing games
- Transposition table
- Move ordering, PV from TTable, captures, MVV-LVA
- Piece square table tapered between mg and eg
- Material count in evaluation
- Decent fast legal move generation, 100M+ NPS using bulk counting at leaf node

