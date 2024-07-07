#ifndef TTABLE_H
#define TTABLE_H


#include <random>
#include <unordered_map>
#include "bitboard.h"
#include "move.h"
#include <iostream>
#include "transpositiontable.h"

struct TTEntry {
    uint64_t key;
    int score = 0; 

};

class TTable {
public:
    TTable() {
        size = (sizeMB * 1000000) / sizeof(TTEntry);
        table = (TTEntry*)malloc(size * sizeof(TTEntry));
        
    }

    void probe(uint64_t key, bool &valid, int &score) {
        TTEntry entry = table[key % size];
        valid = false;
        if (entry.key == key) {
            valid = true;
            score = entry.score;
        } 
        
    };

    void put(uint64_t key, int score) {
        TTEntry entry = { key, score };
        table[key % size] = entry;
    }

    
private:
    TTEntry* table;
    int sizeMB = 16;
    int size = 0;

};

#endif