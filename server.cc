#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "cc-lib/base/logging.h"
#include "simplefm2.h"
#include "emulator.h"
#include "galaxian.h"

namespace galaxian {

class Server {
 public:
  void Loop() {
    cout << "Running Galaxian server\n";
    cout.flush();

    InitSocket();

    SkipMenu();

    vector<uint8> beginning;
    Emulator::Save(&beginning);

    CHECK(GetState().lifes == 2);

    int prev_score = 0;
    int max_score = 0;

    for (int step = 0; ; ++step) {
      uint8 input;
      int seq;
      RecvInput(&input, &seq);

      for (int i = 0; i < 12; ++i) {
        Emulator::Step(input);
      }

      const State s = GetState();
      const int reward = s.score - prev_score;
      Respond(seq, s, reward);

      prev_score = s.score;
      max_score = std::max(max_score, s.score);

      if (s.lifes < 2) {
        Emulator::Load(&beginning);
        prev_score = 1000;  // Next respond will have reward = -1000.
      }

      if (step % 1 == 0) {
        cout << "Step " << step << " Max score: " << max_score << "\n";
      }
    }
  }

 private:
  void InitSocket() {
    sockaddr_in sin;
    sin.sin_addr.s_addr = htonl(INADDR_ANY);
    sin.sin_family = AF_INET;
    sin.sin_port = htons(62343);
    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd_ == -1) {
      cerr << "INVALID socket\n";
      abort();
    }
    if (bind(sockfd_, (sockaddr *)&sin, sizeof(sin))) {
      cerr << "Failed to bind\n";
      abort();
    }
    listen(sockfd_, 1);

    socklen_t sizeof_sin = sizeof(sin);
    sockfd_ = accept(sockfd_, (sockaddr*)&sin, &sizeof_sin);
    if (sockfd_ == -1) {
      cerr << "Sock accept failed\n";
      abort();
    }
  }

  void RecvInput(uint8* input, int* seq) {
    ssize_t size = recv(sockfd_, buffer_, sizeof(buffer_), 0);
    if (size <= 0) {
      cerr << "Recv size: " << size << "\n";
      abort();
    }
    buffer_[size] = 0;
    // fprintf(stderr, "Recv: %s", buffer_);
    char action;
    sscanf(buffer_, "%c %d", &action, seq);
    *input = ToInput(action);
  }

  void Respond(int seq, const State& s, int reward) {
    SendInt(seq);
    SendInt(reward);
    SendPoint(s.galaxian);
    SendPoint(s.missile);
    for (int e : s.still_enemies_encoded) {
      SendInt(e);
    }
    SendInt(s.incoming_enemies.size());
    for (const Point& e : s.incoming_enemies) {
      SendPoint(e);
    }
    SendInt(s.bullets.size());
    for (const Point& b : s.bullets) {
      SendPoint(b);
    }
  }

  void SendInt(int i) {
    SendBuffer(sprintf(buffer_, "%d\n", i));
  }

  void SendPoint(const Point& p) {
    SendBuffer(sprintf(buffer_, "%d %d\n", p.x, p.y));
  }

  void SendBuffer(int size) {
    // fprintf(stderr, "Send: %s", buffer_);
    if (send(sockfd_, buffer_, size, 0) < 0) {
      fprintf(stderr, "Send failed\n");
      abort();
    }
  }

  int sockfd_;
  char buffer_[256];
};

}  // namespace galaxian

int main(int argc, char *argv[]) {
  //cout.sync_with_stdio(false);

  Emulator::Initialize("galaxian.nes");

  galaxian::Server server;
  server.Loop();

  Emulator::Shutdown();

  // exit the infrastructure
  FCEUI_Kill();

  fprintf(stderr, "SUCCESS.\n");
  return 0;
}
