#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "simplefm2.h"
#include "emulator.h"
#include "basis-util.h"
#include "galaxian.h"

int main(int argc, char *argv[]) {
  Emulator::Initialize("galaxian.nes");

  vector<uint8> inputs = SimpleFM2::ReadInputs("galaxian.fm2");
  vector<uint8> basis = BasisUtil::LoadOrComputeBasis(inputs, 4935, "galaxian.basis");
  // The nth savestate is from before issuing the nth input.
  vector<vector<uint8>> savestates;

  vector<uint8> beginning;
  Emulator::Save(&beginning);

  int prev_score = -1;
  string prev_action;
  fprintf(stderr, "Running %zd steps...\n", inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    vector<uint8> v;
    Emulator::SaveEx(&v, &basis);
    savestates.push_back(v);

    Emulator::Step(inputs[i]);

    const int score = GetScore();
    const string action = GetAction(inputs[i]);
    if (score != prev_score || action != prev_action) {
      fprintf(stderr, "%d %s %d %d\n", i + 510-409, action.c_str(), score, GetX());
      prev_action = action;
      prev_score = score;
    }
  }

  fprintf(stderr, "Checksum %lx\n", Emulator::RamChecksum());

  Emulator::Shutdown();

  // exit the infrastructure
  FCEUI_Kill();

  fprintf(stderr, "SUCCESS.\n");
  return 0;
}
