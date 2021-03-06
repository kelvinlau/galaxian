#ifndef __GALAXIAN_H
#define __GALAXIAN_H

#include <vector>
#include <map>
#include "fceu/utils/md5.h"
#include "fceu/driver.h"
#include "fceu/drivers/common/args.h"
#include "fceu/state.h"
#include "fceu/fceu.h"
#include "fceu/types.h"
#include "emulator.h"

using std::vector;
using std::map;

namespace galaxian {

struct Point {
  int x, y;
};

struct IncomingEnemy : Point {
  int row;
};

struct State {
  Point galaxian;
  //vector<Point> still_enemies;
  vector<int> still_enemies_encoded;
  map<int, IncomingEnemy> incoming_enemies;
  map<int, Point> bullets;
  Point missile;
  int score;
  int lifes;
};

// Incoming enemies.
map<int, IncomingEnemy> GetIncomingEnemies() {
  map<int, IncomingEnemy> ret;
  for (int i = 0; i <= 6; ++i) {
    int x = RAM[0x203 + 0x10 * i];
    int y = RAM[0x203 + 0x10 * i + 1];
    if (x > 0 and y > 0) {
      IncomingEnemy e;
      e.x = (x + 8) % 0xFF;
      e.y = y + 3;
      e.row = RAM[0x718 + i];
      ret[i] = e;
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
    int y = 105;
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
  int dx = (RAM[0xE5] + 56) % 256;
  ret.push_back(dx);
  for (int i = 0; i < 10; ++i) {
    int mask = RAM[0xC3 + i];
    ret.push_back(mask);
  }
  return ret;
}

// Incoming enemy bullets.
map<int, Point> GetBullets() {
  map<int, Point> ret;
  int id = 0;
  for (int addr = 0x287; addr <= 0x29F; addr += 4) {
    int x = RAM[addr];
    int y = RAM[addr - 3];
    if (x > 0 and y > 0) {
      ret[id] = Point{x + 4, y + 5};
    }
    ++id;
  }
  return ret;
}

// Our missile. nil if not fired.
Point GetMissile() {
  int x = RAM[0x283]+4;
  int y = RAM[0x280]+5;
  return {x, y};
}

int GetScore() {
  int score = 0;
  for (int addr = 0x6A0; addr <= 0x6A5; ++addr) {
    score = 10 * score + (RAM[addr] & 0xF);
  }
  return score;
}

int GetLifes() {
  return RAM[0x42];
}

State GetState() {
  State s;
  s.galaxian.x = (RAM[0xE4] + 128) % 256;
  s.galaxian.y = 209;
  //s.still_enemies = GetStillEnemies();
  s.still_enemies_encoded = GetStillEnemiesEncoded();
  s.incoming_enemies = GetIncomingEnemies();
  s.bullets = GetBullets();
  s.missile = GetMissile();
  s.score = GetScore();
  s.lifes = GetLifes();
  return s;
}

bool IsDead() {
  return RAM[0x41] != 0;
}

int GetLevel() {
  return RAM[0x59];
}

char ToAction(uint8 input) {
  // RLDUTSBA
  const bool a = (input & 0x01);
  const bool l = (input & 0x40);
  const bool r = (input & 0x80);
  if (a) {
    if (l) return 'l';
    if (r) return 'r';
    return 'A';
  } else {
    if (l) return 'L';
    if (r) return 'R';
    return '_';
  }
}

uint8 ToInput(char action) {
  switch (action) {
    case 'L': return 0x40;
    case 'R': return 0x80;
    case 'A': return 0x01;
    case 'l': return 0x41;
    case 'r': return 0x81;
  }
  return 0x00;
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
