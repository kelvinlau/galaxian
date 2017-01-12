#ifndef __GALAXIAN_H
#define __GALAXIAN_H

#include "fceu/utils/md5.h"
#include "fceu/driver.h"
#include "fceu/drivers/common/args.h"
#include "fceu/state.h"
#include "fceu/fceu.h"
#include "fceu/types.h"

using namespace std;

namespace galaxian {

struct Point {
  int x, y;
};

struct State {
  Point galaxian;
  //vector<Point> still_enemies;
  vector<int> still_enemies_encoded;
  vector<Point> incoming_enemies;
  vector<Point> bullets;
  Point missile;
  int score;
  int lifes;
};

// Incoming enemies.
vector<Point> GetIncomingEnemies() {
  vector<Point> ret;
  for (int addr = 0x203; addr <= 0x253; addr += 0x10) {
    int x = RAM[addr];
    int y = RAM[addr + 1];
    if (x > 0 and y > 0) {
      ret.push_back({(x+8)%0xFF, y+6});
    }
  }
  return ret;
}

// Enemies standing still.
vector<Point> GetStillEnemies() {
  vector<Point> ret;
  int dx = RAM[0xE5];
  for (int i = 0; i < 10; ++i) {
    int x = (dx + 48 + 16 * i + 8) % 0xFF;
    int y = 108;
    int mask = RAM[0xC3 + i];
    while (mask > 0) {
      if (mask % 2) {
        ret.push_back({x, y});
      }
      mask >>= 1;
      y -= 12;
    }
  }
  return ret;
}

vector<int> GetStillEnemiesEncoded() {
  vector<int> ret;
  ret.reserve(11);
  int dx = RAM[0xE5];
  ret.push_back(dx);
  for (int i = 0; i < 10; ++i) {
    int mask = RAM[0xC3 + i];
    ret.push_back(mask);
  }
  return ret;
}

// Incoming enemy bullets.
vector<Point> GetBullets() {
  vector<Point> ret;
  for (int addr = 0x28B; addr <= 0x29F; addr += 4) {
    int x = RAM[addr];
    int y = RAM[addr - 3];
    if (x > 0 and y > 0) {
      ret.push_back({x + 4, y + 8});
    }
  }
  return ret;
}

// Our missile. nil if not fired.
Point GetMissile() {
  int x = RAM[0x283];
  int y = RAM[0x280];
  return {x, y};
}

int GetScore() {
  int score = 0;
  for (int addr = 0x6A0; addr <= 0x6A5; ++addr) {
    score = 10 * score + (RAM[addr] & 0xF);
  }
  return score;
}

State GetState() {
  State s;
  s.galaxian.x = (RAM[0xE4] + 128) % 256;
  s.galaxian.y = 222;
  //s.still_enemies = GetStillEnemies();
  s.still_enemies_encoded = GetStillEnemiesEncoded();
  s.incoming_enemies = GetIncomingEnemies();
  s.bullets = GetBullets();
  s.missile = GetMissile();
  s.score = GetScore();
  s.lifes = RAM[0x42];
  return s;
}

bool IsDead() {
  return RAM[0x41] != 0;
}

char ToAction(uint8 input) {
  // RLDUTSBA
  if (input & 0x01) return 'A';
  if ((input & 0xB0) == 0xB0) return '_';
  if (input & 0x40) return 'L';
  if (input & 0x80) return 'R';
  if (input & ~0xB1) return '?';
  return '_';
}

uint8 ToInput(char action) {
  switch (action) {
    case 'L': return 0x40;
    case 'R': return 0x80;
    case 'A': return 0x01;
  }
  return 0;
}

void SkipFrames(int frames) {
  for (int i = 0; i < frames; ++i) {
    Emulator::Step(0);
  }
}

void SkipMenu() {
  for (int k = 0; k < 2; ++k) {
    SkipFrames(60);
    for (int i = 0; i < 10; ++i) {
      Emulator::Step(0x08);  // Start button.
    }
    for (int i = 0; i < 10; ++i) {
      Emulator::Step(0);
    }
  }
  SkipFrames(240);
}

}  // namespace galaxian

#endif
