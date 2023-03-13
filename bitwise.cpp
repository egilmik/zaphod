#include <iostream>
#include <bitset>

int main() {
    unsigned long long piceses = 0b1111111111111111000000000000000000000000000000001111111111111111;
    unsigned long long white = 0b1111111111111111000000000000000000000000000000000000000000000000;
    unsigned long long rooks = 0b1000000100000000000000000000000000000000000000000000000010000001;
    unsigned long long knights = 0b0100001000000000000000000000000000000000000000000000000001000010;
    unsigned long long queens = 0;
    unsigned long long kings = 0;
    unsigned long long pawns = 0;
    unsigned long long bishops = 0b0010010000000000000000000000000000000000000000000000000000100100;
    unsigned long long boardWhite = piceses & white & rooks;
    std::cout << "piceses & white = " << std::bitset<64>(boardWhite) << std::endl;
}