#ifndef TTABLE_H
#define TTABLE_H


#include <random>
#include <unordered_map>
#include "bitboard.h"
#include "move.h"
#include <iostream>
#include "transpositiontable.h"

enum TType { EXACT, UPPER, LOWER };

struct TTEntry {
    uint64_t key;
    int score = 0;
    Move bestMove;
    TType type;
    int depth;
};



class TTable {
public:
    TTable(int sizeMB) {
        //In MB
        size = (sizeMB * 1000000) / sizeof(TTEntry);
        table = (TTEntry*)malloc(size * sizeof(TTEntry));
    }

    TTEntry* probe(uint64_t key, bool &valid) {
        TTEntry entry = table[key % size];
        valid = false;
        if (entry.key == key) {
            valid = true;
            return &entry;
        }
    };

    void put(uint64_t key, int score,int depth = 0, Move move = Move(), TType type = EXACT) {
        TTEntry entry = { key, score, move, type, depth };
        table[key % size] = entry;
    }

    void clear() {
        for (int i = 0; i < size; i++) {
            table[i].key = 0;
            table[i].score = 0;
        }
    }

    
private:
    TTEntry* table;
    int size = 0;

};

#endif