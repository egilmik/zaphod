Zaphod is just a hobby!

Maybe at some point it will get to a ELO 2000+, but not today


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