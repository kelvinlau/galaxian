#ifndef _guid_h_
#define _guid_h_

#include <string>
#include "fceu/types.h"
#include "fceu/utils/valuearray.h"

struct FCEU_Guid : public ValueArray<uint8,16>
{
	void newGuid();
	string toString();
	static FCEU_Guid fromString(string str);
	static uint8 hexToByte(char** ptrptr);
	void scan(string& str);
};


#endif
