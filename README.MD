Zaphod is just a hobby!

Maybe at some point it will get to a ELO 2000+, but not today

### Version 1.2

Estimated ELO ~1400

Changelog:
- If negamax does not return a move, pick one from the movelist instead of doing an invalid move
- Added passed pawn scoring (Low score, max 30)

### Version 1.1

Estimated ELO ~1380

Change log:
- Fixed error in promotion
- Added half move clock

### Version 1.0

The first functional version which can play games, with a certain confidence it will not do very stupid things.

Estimated ELO ~1350

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

