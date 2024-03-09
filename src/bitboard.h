#ifndef BITBOARD_H
#define BITBOARD_H

#ifdef LINUX
	#define BitBoard __UINT64_TYPE__
#elif WIN32
	#define BitBoard int64_t
#endif


//#define BitBoard __UINT64_TYPE__
enum BitBoardEnum {White,R,N,B,Q,K,P,Black,r,n,b,q,k,p,All};

#endif