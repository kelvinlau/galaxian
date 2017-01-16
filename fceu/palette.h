#ifndef PALETTE_H_
#define PALETTE_H_

#include "fceu/types.h"

struct pal {
	uint8 r,g,b;
};

extern pal *palo;
void FCEU_ResetPalette(void);

void FCEU_ResetPalette(void);
void FCEU_ResetMessages();
void FCEU_LoadGamePalette(void);
void FCEU_DrawNTSCControlBars(uint8 *XBuf);

#endif
