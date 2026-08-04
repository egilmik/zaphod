#ifndef SEE_H
#define SEE_H

#include <algorithm>
#include "board.h"
#include "material.h"

namespace See {
    inline int see(Board& board, int fromSq, int toSq, BitBoardEnum sideToMove) {

        int ply, score[32];

        score[0] = Material::pieceMaterialScoreArray[board.getPieceOnSquare(toSq)];

        BitBoard occupied = board.getBitboard(All);
        BitBoard toSqBB = board.sqBB[toSq];


        //Remove pieces
        occupied &= ~board.sqBB[fromSq];


        BitBoard attackersTo = 0;


        uint64_t magic = ((occupied & board.rookMask[toSq]) * board.magicNumberRook[toSq]) >> board.magicNumberShiftsRook[toSq];
        attackersTo |= (*board.magicMovesRook)[toSq][magic] & (board.getBitboard(Q) | board.getBitboard(R) | board.getBitboard(q) | board.getBitboard(r));

        magic = ((occupied & board.bishopMask[toSq]) * board.magicNumberBishop[toSq]) >> board.magicNumberShiftsBishop[toSq];
        attackersTo |= (*board.magicMovesBishop)[toSq][magic] & (board.getBitboard(Q) | board.getBitboard(B) | board.getBitboard(q) | board.getBitboard(b));

        attackersTo |= board.getKnightMask(toSq) & (board.getBitboard(N) | board.getBitboard(n));

        attackersTo |= ((toSqBB & ~board.FileHMask) >> 7) & board.getBitboard(P);
        attackersTo |= ((toSqBB & ~board.FileAMask) >> 9) & board.getBitboard(P);

        attackersTo |= ((toSqBB & ~board.FileAMask) << 7) & board.getBitboard(p);
        attackersTo |= ((toSqBB & ~board.FileHMask) << 9) & board.getBitboard(p);

        attackersTo |= board.getKingMask(toSq) & (board.getBitboard(K) | board.getBitboard(k));

        //Remove the already capture pieces
        attackersTo &= ~board.sqBB[fromSq];
        ply = 1;

        sideToMove = (sideToMove == White) ? Black : White;

        BitBoard attackerBB = 0;
        BitBoardEnum attacker = board.getPieceOnSquare(fromSq);
        while (attackersTo & board.getBitboard(sideToMove)) {



            score[ply] = -score[ply - 1] + Material::pieceMaterialScoreArray[attacker];

            if ((attackerBB = attackersTo & board.getBitboard(P + sideToMove))) {
                fromSq = board.popLsb(attackerBB);
                attacker = board.getPieceOnSquare(fromSq);

            }
            else if ((attackerBB = attackersTo & board.getBitboard(N + sideToMove))) {
                fromSq = board.popLsb(attackerBB);
                attacker = board.getPieceOnSquare(fromSq);

            }
            else if ((attackerBB = attackersTo & board.getBitboard(B + sideToMove))) {
                fromSq = board.popLsb(attackerBB);
                attacker = board.getPieceOnSquare(fromSq);
            }
            else if ((attackerBB = attackersTo & board.getBitboard(R + sideToMove))) {
                fromSq = board.popLsb(attackerBB);
                attacker = board.getPieceOnSquare(fromSq);

            }
            else if ((attackerBB = attackersTo & board.getBitboard(Q + sideToMove))) {
                fromSq = board.popLsb(attackerBB);
                attacker = board.getPieceOnSquare(fromSq);

            }
            else if ((attackerBB = attackersTo & board.getBitboard(K + sideToMove))) {
                //score[ply++] = 100000000;
                fromSq = board.popLsb(attackerBB);
                attacker = board.getPieceOnSquare(fromSq);

            }
            else {
                break;
            }

            attackersTo ^= board.sqBB[fromSq];
            occupied ^= board.sqBB[fromSq];

            magic = ((occupied & board.rookMask[toSq]) * board.magicNumberRook[toSq]) >> board.magicNumberShiftsRook[toSq];
            attackersTo |= (*board.magicMovesRook)[toSq][magic] & ((board.getBitboard(Q) | board.getBitboard(R) | board.getBitboard(q) | board.getBitboard(r)) & occupied);

            magic = ((occupied & board.bishopMask[toSq]) * board.magicNumberBishop[toSq]) >> board.magicNumberShiftsBishop[toSq];
            attackersTo |= (*board.magicMovesBishop)[toSq][magic] & ((board.getBitboard(Q) | board.getBitboard(B) | board.getBitboard(q) | board.getBitboard(b)) & occupied);

            attackersTo |= ((toSqBB & ~board.FileHMask) >> 7) & (board.getBitboard(P) & occupied);
            attackersTo |= ((toSqBB & ~board.FileAMask) >> 9) & (board.getBitboard(P) & occupied);

            attackersTo |= ((toSqBB & ~board.FileAMask) << 7) & (board.getBitboard(p) & occupied);
            attackersTo |= ((toSqBB & ~board.FileHMask) << 9) & (board.getBitboard(p) & occupied);

            sideToMove = (sideToMove == White) ? Black : White;

            ply++;
        }

        while (--ply) {
            score[ply - 1] = -std::max(-score[ply - 1], score[ply]);
        }
        return score[0];

    }
}

#endif