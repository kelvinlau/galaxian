#ifndef __GALAXIAN_H
#define __GALAXIAN_H

#include "fceu/utils/md5.h"
#include "fceu/driver.h"
#include "fceu/drivers/common/args.h"
#include "fceu/state.h"
#include "fceu/fceu.h"
#include "fceu/types.h"

using namespace std;

int GetScore() {
  int score = 0;
  for (int addr = 0x6A0; addr <= 0x6A5; ++addr) {
    score = 10 * score + (RAM[addr] & 0xF);
  }
  return score;
}

int GetX() {
  return (RAM[0xE5]) % 256;
}

string GetAction(uint8 input) {
  // RLDUTSBA
  if (input & 0x01) return "A";
  if ((input & 0xB0) == 0xB0) return "_";
  if (input & 0x40) return "L";
  if (input & 0x80) return "R";
  if (input & ~0xB1) return "?";
  return "_";
}

#endif
